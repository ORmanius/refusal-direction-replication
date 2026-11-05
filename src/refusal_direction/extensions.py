"""Extensions beyond the core replication. Each one either tightens the evidence or shows
the mechanism; all reuse the direction fitted by `pipeline.run_experiment`.

1. Direction similarity: cosine similarity between the per-layer difference-in-means
   directions. The paper's "single direction" claim predicts a block of near-identical
   directions across the middle layers.
2. Capability check (the paper's KL criterion plus a loss on reference answers): for each
   layer's direction, ablate it everywhere and measure (a) the KL divergence from the
   baseline next-token distribution at the last prompt token on harmless prompts and
   (b) the mean negative log-likelihood of Alpaca reference answers. A direction that
   merely damages the model shows up here.
3. Token-by-token dynamics: the projection of the residual stream onto the chosen
   direction at every token of prompt + completion, under each intervention.
4. Transfer: the direction fitted on AdvBench applied to HarmBench, JailbreakBench harmful
   (minus its AdvBench-sourced rows) and JailbreakBench benign prompts.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import get_transfer_sets, load_harmless_pairs
from .intervention import _batches, _walk, capture_residuals, generate
from .model import ModelBundle, format_prompts, tokenize, value
from .pipeline import REPO_ROOT, ExperimentConfig, _write_jsonl, set_seed
from .scoring import is_refusal, refusal_rate

N_PROMPT_TOKENS_SHOWN = 8  # prompt tokens kept (before the first generated token) in the dynamics plots


def load_run(out_dir: Path) -> dict:
    """Directions and chosen layer written by `run_experiment` into `out_dir`."""
    d = torch.load(out_dir / "directions.pt")
    with open(out_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    return {"r": d["r"], "r_hat": d["r_hat"], "chosen": int(summary["chosen_layer"]), "summary": summary}


# 1. Direction similarity ---------------------------------------------------------------


def direction_cosine(r_hat: torch.Tensor) -> np.ndarray:
    """[n_layers, n_layers] cosine similarity between unit directions."""
    return (r_hat.float() @ r_hat.float().T).cpu().numpy()


# 2. Capability check -------------------------------------------------------------------


@torch.no_grad()
def last_token_logprobs(
    bundle: ModelBundle,
    prompts: list[str],
    batch_size: int = 16,
    ablate_dir: torch.Tensor | None = None,
    add_vec: torch.Tensor | None = None,
    add_layer: int | None = None,
    ablate_scale: float = 1.0,
    add_scale: float = 1.0,
) -> torch.Tensor:
    """Log-probabilities of the next token at the last prompt position, [n, vocab] float32 CPU."""
    out = []
    for batch in _batches(prompts, batch_size):
        enc = tokenize(bundle, batch).to(bundle.device)
        with bundle.lm.trace(enc):
            _walk(bundle, ablate_dir=ablate_dir, add_vec=add_vec, add_layer=add_layer,
                  ablate_scale=ablate_scale, add_scale=add_scale)
            lp = torch.log_softmax(bundle.lm.output.logits[:, -1, :].float(), dim=-1).cpu().save()
        out.append(value(lp))
    return torch.cat(out, dim=0)


def kl_divergence(p_logprobs: torch.Tensor, q_logprobs: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per row, in nats."""
    return (p_logprobs.exp() * (p_logprobs - q_logprobs)).sum(-1)


def _reference_texts(bundle: ModelBundle, pairs: list[tuple[str, str]]) -> tuple[list[str], list[int]]:
    """Chat-formatted (instruction + reference answer) texts and the number of answer tokens in each."""
    tok = bundle.tokenizer
    if bundle.chat:
        prompts = format_prompts(bundle, [i for i, _ in pairs])
        full = [
            tok.apply_chat_template(
                [{"role": "user", "content": i}, {"role": "assistant", "content": o}], tokenize=False
            )
            for i, o in pairs
        ]
    else:
        prompts = [i + "\n" for i, _ in pairs]
        full = [i + "\n" + o for i, o in pairs]
    add = not bundle.chat
    n_answer = [
        len(tok(f, add_special_tokens=add)["input_ids"]) - len(tok(p, add_special_tokens=add)["input_ids"])
        for f, p in zip(full, prompts)
    ]
    return full, n_answer


LOGITS_BUDGET_BYTES = 2**29  # keep one batch of full logits under 512 MB


def logits_budget_batch(bundle: ModelBundle, n_tokens: int, requested: int,
                        budget_bytes: int = LOGITS_BUDGET_BYTES) -> int:
    """Largest batch size (at most `requested`) whose [batch, n_tokens, vocab] logits fit the budget.

    The full-sequence logits are the largest tensor a scoring pass creates, and cross-entropy
    copies them once more. Gemma-2 has a 256k vocabulary, so four 330-token texts in float32
    are 1.3 GB of logits, which is what pushes a 15 GB T4 over the edge next to the 10.5 GB of
    weights. Qwen's 152k vocabulary in float16 fits the requested batch unchanged.
    """
    vocab = getattr(bundle.lm.config, "vocab_size", None) or len(bundle.tokenizer)
    itemsize = torch.tensor([], dtype=bundle.dtype).element_size()
    return max(1, min(requested, budget_bytes // (vocab * max(n_tokens, 1) * itemsize)))


@torch.no_grad()
def reference_nll(
    bundle: ModelBundle,
    pairs: list[tuple[str, str]],
    batch_size: int = 4,
    ablate_dir: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean negative log-likelihood per answer token of each reference answer, [n] float32."""
    full, n_answer = _reference_texts(bundle, pairs)
    lengths = [len(bundle.tokenizer(t, add_special_tokens=not bundle.chat)["input_ids"]) for t in full]
    out = []
    i = 0
    while i < len(full):
        bs = logits_budget_batch(bundle, max(lengths[i : i + batch_size]), batch_size)
        batch, n_ans = full[i : i + bs], n_answer[i : i + bs]
        i += bs
        enc = tokenize(bundle, batch).to(bundle.device)
        ids = enc["input_ids"]
        # Left padding puts the answer tokens at the end of every row, so only the logits that
        # predict the last max(n_ans) tokens are needed.
        n_max = max(n_ans)
        with bundle.lm.trace(enc):
            _walk(bundle, ablate_dir=ablate_dir)
            logits = bundle.lm.output.logits
            nll = F.cross_entropy(
                logits[:, -(n_max + 1) : -1, :].float().transpose(1, 2), ids[:, -n_max:], reduction="none"
            ).cpu().save()
        nll = value(nll)
        out.extend(float(nll[j, -n:].mean()) for j, n in enumerate(n_ans))
    return torch.tensor(out)


def capability_by_layer(
    bundle: ModelBundle,
    harmless_prompts: list[str],
    pairs: list[tuple[str, str]],
    r_hat: torch.Tensor,
    layers: list[int],
    batch_size: int = 16,
    nll_batch_size: int = 4,
) -> pd.DataFrame:
    """KL to baseline and reference-answer NLL when each layer's direction is ablated everywhere.

    Row `layer == -1` is the baseline (no ablation)."""
    base_lp = last_token_logprobs(bundle, harmless_prompts, batch_size)
    base_nll = reference_nll(bundle, pairs, nll_batch_size)
    rows = [{"layer": -1, "kl_mean": 0.0, "kl_max": 0.0, "reference_nll": float(base_nll.mean())}]
    for layer in layers:
        lp = last_token_logprobs(bundle, harmless_prompts, batch_size, ablate_dir=r_hat[layer])
        kl = kl_divergence(base_lp, lp)
        nll = reference_nll(bundle, pairs, nll_batch_size, ablate_dir=r_hat[layer])
        rows.append(
            {"layer": layer, "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
             "reference_nll": float(nll.mean())}
        )
        print(f"layer {layer:3d}: KL {rows[-1]['kl_mean']:.4f}  reference NLL {rows[-1]['reference_nll']:.4f}",
              flush=True)
    return pd.DataFrame(rows)


# 3. Token-by-token dynamics ------------------------------------------------------------


@torch.no_grad()
def token_projections(
    bundle: ModelBundle,
    texts: list[str],
    direction: torch.Tensor,
    batch_size: int = 8,
    ablate_dir: torch.Tensor | None = None,
    add_vec: torch.Tensor | None = None,
    add_layer: int | None = None,
) -> list[np.ndarray]:
    """Projection of every layer's residual stream onto `direction` at every token.

    Returns one [n_layers, n_tokens] array per text (padding removed). The intervention,
    if any, is applied during the forward pass, so a teacher-forced pass over prompt +
    greedy completion reproduces the activations seen during generation."""
    d = direction.to(bundle.device, bundle.dtype)
    out: list[np.ndarray] = []
    for batch in _batches(texts, batch_size):
        enc = tokenize(bundle, batch).to(bundle.device)
        with bundle.lm.trace(enc):
            cols = _walk(bundle, ablate_dir=ablate_dir, add_vec=add_vec, add_layer=add_layer, position=slice(None))
            proj = (torch.stack(cols, dim=1) * d).sum(-1).float().cpu().save()  # [B, L, T]
        proj = value(proj).numpy()
        mask = enc["attention_mask"].cpu().numpy().astype(bool)
        for j in range(proj.shape[0]):
            out.append(proj[j][:, mask[j]])
    return out


def _token_strings(bundle: ModelBundle, text: str) -> list[str]:
    ids = tokenize(bundle, [text])["input_ids"][0].tolist()
    return [bundle.tokenizer.decode([i]) for i in ids]


def _prompt_len(bundle: ModelBundle, prompt: str) -> int:
    return int(tokenize(bundle, [prompt])["attention_mask"].sum())


def _aligned_mean(projs: list[np.ndarray], prompt_lens: list[int], layer: int, max_new_tokens: int) -> pd.DataFrame:
    """Mean projection at `layer` by token index relative to the first generated token."""
    ks = range(-N_PROMPT_TOKENS_SHOWN, max_new_tokens)
    rows = []
    for k in ks:
        vals = []
        for p, n in zip(projs, prompt_lens):
            t = n + k
            if 0 <= t < p.shape[1]:
                vals.append(p[layer, t])
        if vals:
            rows.append({"k": k, "mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)})
    return pd.DataFrame(rows)


def dynamics_study(
    bundle: ModelBundle,
    harmful: list[str],
    harmless: list[str],
    r: torch.Tensor,
    r_hat: torch.Tensor,
    chosen: int,
    max_new_tokens: int = 64,
    batch_size: int = 8,
    example_index: int = 0,
) -> dict:
    """Projections onto the chosen direction, token by token, for four conditions.

    harmful prompts: baseline and direction ablated; harmless prompts: baseline and direction
    added at the chosen layer. Also returns the full layer x token map for one harmless example
    (its tokens are safe to display)."""
    d = r_hat[chosen]
    conds = {
        "harmful_baseline": (harmful, {}),
        "harmful_ablated": (harmful, {"ablate_dir": d}),
        "harmless_baseline": (harmless, {}),
        "harmless_added": (harmless, {"add_vec": r[chosen], "add_layer": chosen}),
    }
    curves, projs_all, completions_all = {}, {}, {}
    for name, (prompts, kw) in conds.items():
        comps = generate(bundle, prompts, max_new_tokens, batch_size, **kw)
        texts = [p + c for p, c in zip(prompts, comps)]
        projs = token_projections(bundle, texts, d, batch_size, **kw)
        lens = [_prompt_len(bundle, p) for p in prompts]
        curves[name] = _aligned_mean(projs, lens, chosen, max_new_tokens)
        projs_all[name], completions_all[name] = projs, comps
        print(f"{name}: refusal rate {refusal_rate(comps):.3f}", flush=True)

    i = example_index
    example = {}
    for name in ("harmless_baseline", "harmless_added"):
        text = harmless[i] + completions_all[name][i]
        example[name] = {
            "tokens": _token_strings(bundle, text),
            "prompt_len": _prompt_len(bundle, harmless[i]),
            "proj": projs_all[name][i],  # [L, T]
            "refusal": is_refusal(completions_all[name][i]),
        }
    return {"curves": curves, "example": example, "chosen": chosen, "max_new_tokens": max_new_tokens}


# 4. Transfer ---------------------------------------------------------------------------


def transfer_study(
    bundle: ModelBundle,
    sets: dict,
    r: torch.Tensor,
    r_hat: torch.Tensor,
    chosen: int,
    out_dir: Path,
    max_new_tokens: int = 64,
    batch_size: int = 16,
    tag: str = "",
) -> pd.DataFrame:
    """Refusal rates on the transfer sets with the direction of layer `chosen` ablated (all
    three sets) or added (benign set). Completions of the ablated model are written to files
    that .gitignore keeps out of the repository; `tag` is appended to the file names."""
    plan = [("harmbench_standard", "ablation"), ("jbb_harmful", "ablation"),
            ("jbb_benign", "ablation"), ("jbb_benign", "addition")]
    base_cache: dict[str, list[str]] = {}
    rows = []
    for name, kind in plan:
        instr = sets[name]
        prompts = format_prompts(bundle, instr)
        if name not in base_cache:
            base_cache[name] = generate(bundle, prompts, max_new_tokens, batch_size)
            _write_jsonl(out_dir / f"completions_{name}_baseline.jsonl", instr, base_cache[name])
        base = base_cache[name]
        if kind == "addition":
            inter = generate(bundle, prompts, max_new_tokens, batch_size, add_vec=r[chosen], add_layer=chosen)
        else:
            inter = generate(bundle, prompts, max_new_tokens, batch_size, ablate_dir=r_hat[chosen])
        _write_jsonl(out_dir / f"completions_{name}_{kind}{tag}.jsonl", instr, inter)
        rows.append(
            {"set": name, "n": len(instr), "layer": chosen, "intervention": kind,
             "baseline_refusal_rate": refusal_rate(base), "intervention_refusal_rate": refusal_rate(inter)}
        )
        print(f"{name}: baseline {rows[-1]['baseline_refusal_rate']:.3f} -> {kind} (layer {chosen}) "
              f"{rows[-1]['intervention_refusal_rate']:.3f}", flush=True)
    return pd.DataFrame(rows)


# 5. Dose-response -----------------------------------------------------------------------

ADD_COEFS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
ABL_COEFS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def refusal_first_token_ids(bundle: ModelBundle, completions: list[str]) -> list[int]:
    """Token ids that start the refusals among `completions` (the model's own refusal openers)."""
    tok = bundle.tokenizer
    ids = set()
    for c in completions:
        if is_refusal(c) and c.strip():
            ids.add(tok.encode(c, add_special_tokens=False)[0])
    return sorted(ids)


def dose_response_study(
    bundle: ModelBundle,
    harmful: list[str],
    harmless: list[str],
    r: torch.Tensor,
    r_hat: torch.Tensor,
    layer: int,
    harmless_example: tuple[str, str],
    benign_example: tuple[str, str],
    add_coefs: list[float] = ADD_COEFS,
    abl_coefs: list[float] = ABL_COEFS,
    max_new_tokens: int = 64,
    batch_size: int = 16,
) -> dict:
    """Refusal as a function of intervention strength.

    Addition: x <- x + a r at `layer`, on the harmless validation prompts, a in `add_coefs`.
    Ablation: x <- x - a (x . r_hat) r_hat at every layer, on the harmful validation prompts,
    a in `abl_coefs` (a = 1 is the paper's full ablation). For every strength: the refusal rate,
    the probability mass the model puts on its own refusal-opening tokens at the first generated
    position, and one example completion. `harmless_example` is (instruction, formatted prompt)
    for the addition side; `benign_example` is a harmless-but-over-refused prompt for the ablation
    side, so that no harmful completion needs to be displayed."""
    base_harmful = generate(bundle, harmful, max_new_tokens, batch_size)
    ref_ids = refusal_first_token_ids(bundle, base_harmful)
    rows, examples = [], []

    def mass(lp: torch.Tensor) -> float:
        return float(lp[:, ref_ids].exp().sum(-1).mean()) if ref_ids else float("nan")

    for a in add_coefs:
        kw = dict(add_vec=r[layer], add_layer=layer, add_scale=a)
        comps = generate(bundle, harmless, max_new_tokens, batch_size, **kw)
        lp = last_token_logprobs(bundle, harmless, batch_size, **kw)
        ex = generate(bundle, [harmless_example[1]], max_new_tokens, 1, **kw)[0]
        rows.append({"kind": "addition_harmless", "layer": layer, "coef": a,
                     "refusal_rate": refusal_rate(comps), "p_refusal_first_token": mass(lp)})
        examples.append({"kind": "addition_harmless", "coef": a, "instruction": harmless_example[0],
                         "completion": ex, "refusal": is_refusal(ex)})
        print(f"addition a={a:.1f}: refusal {rows[-1]['refusal_rate']:.3f}  p(refusal token) {rows[-1]['p_refusal_first_token']:.3f}", flush=True)
    for a in abl_coefs:
        kw = dict(ablate_dir=r_hat[layer], ablate_scale=a)
        comps = generate(bundle, harmful, max_new_tokens, batch_size, **kw)
        lp = last_token_logprobs(bundle, harmful, batch_size, **kw)
        ex = generate(bundle, [benign_example[1]], max_new_tokens, 1, **kw)[0]
        rows.append({"kind": "ablation_harmful", "layer": layer, "coef": a,
                     "refusal_rate": refusal_rate(comps), "p_refusal_first_token": mass(lp)})
        examples.append({"kind": "ablation_benign_example", "coef": a, "instruction": benign_example[0],
                         "completion": ex, "refusal": is_refusal(ex)})
        print(f"ablation a={a:.1f}: refusal {rows[-1]['refusal_rate']:.3f}  p(refusal token) {rows[-1]['p_refusal_first_token']:.3f}", flush=True)
    return {"rows": pd.DataFrame(rows), "examples": examples, "layer": layer,
            "refusal_first_tokens": [bundle.tokenizer.decode([i]) for i in ref_ids]}


# 6. Geometry at the chosen layer, and which MLP neurons separate the two prompt sets ------


def geometry_study(
    bundle: ModelBundle,
    harmful: list[str],
    harmless: list[str],
    r_hat: torch.Tensor,
    layer: int,
    batch_size: int = 16,
) -> dict:
    """PCA of the pooled last-token residuals after `layer` (harmful + harmless prompts), the
    same prompts with the direction ablated, and where the refusal direction sits among the
    principal components."""
    hv = capture_residuals(bundle, harmful, -1, batch_size)[:, layer].numpy()
    hl = capture_residuals(bundle, harmless, -1, batch_size)[:, layer].numpy()
    ha = capture_residuals(bundle, harmful, -1, batch_size, ablate_dir=r_hat[layer])[:, layer].numpy()
    pool = np.concatenate([hv, hl])
    mu = pool.mean(0)
    _, s, vt = np.linalg.svd(pool - mu, full_matrices=False)
    pcs = vt[:3].T  # [d_model, 3]
    explained = (s[:3] ** 2 / (s ** 2).sum()).tolist()
    u = r_hat[layer].numpy()
    rows = []
    for label, pts in (("harmful", hv), ("harmless", hl), ("harmful_ablated", ha)):
        proj = (pts - mu) @ pcs
        along = pts @ u
        for i in range(len(pts)):
            rows.append({"label": label, "index": i, "pc1": float(proj[i, 0]), "pc2": float(proj[i, 1]),
                         "pc3": float(proj[i, 2]), "proj_r_hat": float(along[i])})
    points = pd.DataFrame(rows)
    summary = {
        "layer": layer,
        "explained_variance_pc1_3": explained,
        "abs_cos_r_hat_with_pc1_3": np.abs(u @ pcs).tolist(),
        "mean_proj_r_hat": {k: float(points[points["label"] == k]["proj_r_hat"].mean())
                            for k in ("harmful", "harmless", "harmful_ablated")},
        "r_hat_in_pc_space": (u @ pcs).tolist(),
        "harmless_centroid_pc": ((hl - mu) @ pcs).mean(0).tolist(),
    }
    return {"points": points, "summary": summary}


@torch.no_grad()
def neuron_effects(
    bundle: ModelBundle, harmful: list[str], harmless: list[str], batch_size: int = 16
) -> np.ndarray:
    """Cohen's d (harmful minus harmless) of every MLP neuron's activation at the last prompt
    token, [n_layers, n_neurons]. Neuron activation = the input to the MLP output projection."""

    def acts(prompts):
        outs = []
        for batch in _batches(prompts, batch_size):
            enc = tokenize(bundle, batch).to(bundle.device)
            with bundle.lm.trace(enc):
                cols = [bundle.mlp_out(l).input[:, -1, :] for l in range(bundle.n_layers)]
                stacked = torch.stack(cols, dim=1).float().cpu().save()
            outs.append(value(stacked))
        return torch.cat(outs)

    a, b = acts(harmful), acts(harmless)
    sd = torch.sqrt(0.5 * (a.var(0) + b.var(0))) + 1e-6
    return ((a.mean(0) - b.mean(0)) / sd).numpy()


def neuron_effects_tables(d: np.ndarray, top_k: int = 200, threshold: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(ranked |d| of the top_k neurons per layer, count of neurons above `threshold` per layer)."""
    absd = np.abs(d)
    ranked = -np.sort(-absd, axis=1)[:, :top_k]
    top = pd.DataFrame(ranked, columns=[f"rank_{i}" for i in range(ranked.shape[1])])
    top.insert(0, "layer", range(d.shape[0]))
    counts = pd.DataFrame({"layer": range(d.shape[0]), "n_neurons": d.shape[1],
                           f"n_abs_d_over_{threshold:g}": (absd > threshold).sum(1),
                           "max_abs_d": absd.max(1), "mean_abs_d": absd.mean(1)})
    return top, counts


# 7. Layer selection with the paper's KL constraint ---------------------------------------


def select_layer_with_kl(
    sweep: pd.DataFrame, cap: pd.DataFrame, n_layers: int, skip_last_frac: float = 0.2, kl_max: float = 0.1
) -> dict:
    """The paper filters candidate directions by KL divergence on harmless prompts (< 0.1 nats)
    before picking the one that best removes refusal. Applied to this repo's sweep: among the
    layers below the late-layer cutoff whose ablation KL is under `kl_max`, take the best
    combined validation score (as `pipeline.choose_layer`), ties to the lowest KL. If no layer
    passes the KL filter, the filter is dropped and `kl_filter_satisfied` is False."""
    cutoff = int(round(n_layers * (1 - skip_last_frac)))
    df = sweep.merge(cap[cap["layer"] >= 0], on="layer")
    df = df[df["layer"] < cutoff].copy()
    df["score"] = (1 - df["ablation_refusal_rate_harmful"]) + df["addition_refusal_rate_harmless"]
    ok = df[df["kl_mean"] < kl_max]
    satisfied = len(ok) > 0
    cand = ok if satisfied else df
    best = cand.sort_values(["score", "kl_mean", "layer"], ascending=[False, True, True]).iloc[0]
    return {
        "layer": int(best["layer"]),
        "kl_mean": float(best["kl_mean"]),
        "reference_nll": float(best["reference_nll"]),
        "sweep_ablation_refusal_rate_harmful": float(best["ablation_refusal_rate_harmful"]),
        "sweep_addition_refusal_rate_harmless": float(best["addition_refusal_rate_harmless"]),
        "kl_max": kl_max,
        "kl_filter_satisfied": bool(satisfied),
    }


# Driver --------------------------------------------------------------------------------


def _reference_pairs(splits: dict, n: int, seed: int) -> list[tuple[str, str]]:
    """Alpaca (instruction, output) pairs disjoint from the train/val instructions, fixed seed."""
    used = set(splits["harmless_train"]) | set(splits["harmless_val"])
    pool = [p for p in load_harmless_pairs() if p[0] not in used]
    rng = random.Random(seed)
    return [pool[i] for i in sorted(rng.sample(range(len(pool)), n))]


def run_extensions(
    cfg: ExperimentConfig,
    bundle: ModelBundle,
    r: torch.Tensor,
    r_hat: torch.Tensor,
    chosen: int,
    splits: dict,
    transfer_path: Path = REPO_ROOT / "data" / "transfer_seed0.json",
    n_transfer: int = 64,
    n_reference: int = 64,
    layers: list[int] | None = None,
) -> dict:
    from .plot import (plot_capability_by_layer, plot_direction_similarity, plot_dose_response, plot_dynamics,
                       plot_neuron_effects, plot_pca_clouds)

    t0 = time.time()
    set_seed(cfg.seed)
    out = cfg.out_dir / "extensions"
    out.mkdir(parents=True, exist_ok=True)
    short = cfg.model_name.split("/")[-1]
    layers = layers if layers is not None else list(range(bundle.n_layers))
    fmt = lambda xs: format_prompts(bundle, xs)  # noqa: E731

    # 1. Direction similarity.
    cos = direction_cosine(r_hat)
    pd.DataFrame(cos).to_csv(out / "direction_cosine.csv", index_label="layer")
    plot_direction_similarity(cos, chosen, f"{short}: cosine similarity between per-layer directions",
                              out / "direction_similarity.png")

    # 2. Capability check.
    pairs = _reference_pairs(splits, n_reference, cfg.seed)
    with open(out / "reference_pairs.json", "w", encoding="utf-8") as f:
        json.dump([{"instruction": i, "output": o} for i, o in pairs], f, indent=1, ensure_ascii=False)
    cap = capability_by_layer(bundle, fmt(splits["harmless_val"]), pairs, r_hat, layers, cfg.batch_size)
    cap.to_csv(out / "capability_by_layer.csv", index=False)
    plot_capability_by_layer(cap, chosen, f"{short}: cost of ablating each layer's direction",
                             out / "capability_by_layer.png")

    # 3. Token-by-token dynamics.
    dyn = dynamics_study(bundle, fmt(splits["harmful_val"]), fmt(splits["harmless_val"]), r, r_hat, chosen,
                         cfg.max_new_tokens, min(cfg.batch_size, 8))
    curves = pd.concat([df.assign(condition=k) for k, df in dyn["curves"].items()])
    curves.to_csv(out / "projection_dynamics.csv", index=False)
    ex_rows = []
    for name, ex in dyn["example"].items():
        for t, tok in enumerate(ex["tokens"]):
            ex_rows.append({"condition": name, "t": t, "token": tok, "is_prompt": t < ex["prompt_len"],
                            **{f"layer_{l}": float(ex["proj"][l, t]) for l in range(ex["proj"].shape[0])}})
    pd.DataFrame(ex_rows).to_csv(out / "projection_example.csv", index=False)
    plot_dynamics(dyn, f"{short}: residual-stream projection onto the refusal direction (layer {chosen})",
                  out / "projection_dynamics.png")

    # 4. Transfer.
    sets = get_transfer_sets(transfer_path, n_transfer, cfg.seed)
    tr = transfer_study(bundle, sets, r, r_hat, chosen, out, cfg.max_new_tokens, cfg.batch_size)
    tr.to_csv(out / "transfer.csv", index=False)

    # 5. Dose-response at the chosen layer. The ablation-side example is the first benign
    # JailbreakBench prompt the unmodified model refused (if any; else the first one).
    benign_base = [json.loads(l) for l in open(out / "completions_jbb_benign_baseline.jsonl", encoding="utf-8")]
    benign_instr = next((b["instruction"] for b in benign_base if is_refusal(b["completion"])), benign_base[0]["instruction"])
    harmless_ex = (splits["harmless_val"][0], fmt([splits["harmless_val"][0]])[0])
    benign_ex = (benign_instr, fmt([benign_instr])[0])

    def dose_at(layer: int, tag: str) -> dict:
        d = dose_response_study(bundle, fmt(splits["harmful_val"]), fmt(splits["harmless_val"]), r, r_hat, layer,
                                harmless_ex, benign_ex, max_new_tokens=cfg.max_new_tokens, batch_size=cfg.batch_size)
        d["rows"].to_csv(out / f"dose_response{tag}.csv", index=False)
        with open(out / f"dose_response_examples{tag}.jsonl", "w", encoding="utf-8") as f:
            for e in d["examples"]:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        plot_dose_response(d, f"{short}: refusal as a function of intervention strength (layer {layer})",
                           out / f"dose_response{tag}.png", out / f"dose_response{tag}.gif")
        return d

    dose = dose_at(chosen, "")

    # 6. Geometry at the chosen layer (PCA clouds) and MLP neuron effects.
    geo = geometry_study(bundle, fmt(splits["harmful_train"]), fmt(splits["harmless_train"]), r_hat, chosen, cfg.batch_size)
    geo["points"].to_csv(out / "pca_clouds.csv", index=False)
    plot_pca_clouds(geo, f"{short}: residual stream after layer {chosen} (last prompt token), PCA of 128 + 128 training prompts",
                    out / "pca_clouds.png", out / "pca_clouds.gif")
    d_neuron = neuron_effects(bundle, fmt(splits["harmful_train"]), fmt(splits["harmless_train"]), cfg.batch_size)
    top, counts = neuron_effects_tables(d_neuron)
    top.to_csv(out / "neuron_effects_top.csv", index=False)
    counts.to_csv(out / "neuron_effects_counts.csv", index=False)
    plot_neuron_effects(top, counts, chosen, f"{short}: MLP neurons that separate harmful from harmless prompts",
                        out / "neuron_effects.png")

    # 7. Re-select the layer with the paper's KL filter; if it differs, repeat 3, 4 and 5 there.
    sweep = pd.read_csv(cfg.out_dir / "refusal_by_layer.csv")
    kl_sel = select_layer_with_kl(sweep, cap, bundle.n_layers)
    kl_layer = kl_sel["layer"]
    print(f"layer selected with the KL filter: {kl_layer} (main run chose {chosen})", flush=True)
    if kl_layer != chosen:
        tag = f"_layer{kl_layer}"
        tr_kl = transfer_study(bundle, sets, r, r_hat, kl_layer, out, cfg.max_new_tokens, cfg.batch_size, tag=tag)
        tr_kl.to_csv(out / f"transfer{tag}.csv", index=False)
        dyn_kl = dynamics_study(bundle, fmt(splits["harmful_val"]), fmt(splits["harmless_val"]), r, r_hat,
                                kl_layer, cfg.max_new_tokens, min(cfg.batch_size, 8))
        pd.concat([df.assign(condition=k) for k, df in dyn_kl["curves"].items()]).to_csv(
            out / f"projection_dynamics{tag}.csv", index=False)
        plot_dynamics(dyn_kl, f"{short}: residual-stream projection onto the refusal direction (layer {kl_layer})",
                      out / f"projection_dynamics{tag}.png")
        dose_kl = dose_at(kl_layer, tag)
        kl_sel["transfer"] = tr_kl.to_dict(orient="records")
        kl_sel["example_refusal"] = {k: v["refusal"] for k, v in dyn_kl["example"].items()}
        kl_sel["dose_response"] = dose_kl["rows"].to_dict(orient="records")

    chosen_cap = cap[cap["layer"] == chosen].iloc[0]
    base_cap = cap[cap["layer"] == -1].iloc[0]
    summary = {
        "model": cfg.model_name,
        "chosen_layer": chosen,
        "direction_cosine_chosen_vs_neighbours": {
            str(l): float(cos[chosen, l]) for l in range(bundle.n_layers)
        },
        "capability": {
            "baseline_reference_nll": float(base_cap["reference_nll"]),
            "chosen_ablation_reference_nll": float(chosen_cap["reference_nll"]),
            "chosen_ablation_kl_mean": float(chosen_cap["kl_mean"]),
            "n_reference_pairs": n_reference,
            "n_kl_prompts": len(splits["harmless_val"]),
        },
        "transfer": tr.to_dict(orient="records"),
        "example_refusal": {k: v["refusal"] for k, v in dyn["example"].items()},
        "dose_response": dose["rows"].to_dict(orient="records"),
        "dose_response_refusal_first_tokens": dose["refusal_first_tokens"],
        "dose_response_examples": {"harmless": harmless_ex[0], "benign": benign_ex[0]},
        "geometry": geo["summary"],
        "neuron_effects": {"n_neurons_per_layer": int(d_neuron.shape[1]),
                           "n_abs_d_over_1_per_layer": counts["n_abs_d_over_1"].tolist(),
                           "max_abs_d_per_layer": [round(float(x), 3) for x in counts["max_abs_d"]]},
        "kl_selected": kl_sel,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return {"summary": summary, "cosine": cos, "capability": cap, "dynamics": dyn, "transfer": tr,
            "dose_response": dose, "geometry": geo, "neuron_top": top, "neuron_counts": counts,
            "kl_selected": kl_sel}
