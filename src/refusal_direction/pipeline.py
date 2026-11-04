"""End-to-end experiment: data -> activations -> directions -> per-layer sweep -> CSV + plot.

The notebooks call `run_experiment` (or its pieces) rather than re-implementing anything.
"""

from __future__ import annotations

import json
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import __version__
from .data import get_splits
from .direction import difference_in_means, unit
from .intervention import capture_residuals, generate
from .model import ModelBundle, format_prompts, load_model
from .scoring import refusal_rate

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ExperimentConfig:
    model_name: str
    out_dir: Path
    dtype: str = "float16"
    n_train: int = 128
    n_val: int = 32
    seed: int = 0
    position: int = -1
    max_new_tokens: int = 64
    batch_size: int = 16
    layers: list[int] | None = None  # None = sweep every layer
    attn_implementation: str | None = None
    device_map: str = "auto"
    splits_path: Path = REPO_ROOT / "data" / "splits_seed0.json"
    hf_token: str | None = None


def _rel(path: Path) -> str:
    """Path relative to the repo root (or to the current directory), never an absolute local path."""
    path = Path(path)
    for base in (REPO_ROOT, Path.cwd()):
        try:
            return path.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return path.name


def config_summary(cfg: "ExperimentConfig") -> dict:
    """The run configuration as plain values: paths made relative, the HF token left out."""
    out = {}
    for k, v in asdict(cfg).items():
        if k == "hf_token":
            continue
        out[k] = _rel(v) if isinstance(v, Path) else v
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_layer(df: pd.DataFrame, n_layers: int, skip_last_frac: float = 0.2) -> int:
    """Pick the layer whose direction best satisfies both tests on the validation sets.

    Candidates exclude the last `skip_last_frac` of layers (the paper does the same, since
    late-layer directions tend not to transfer). Score = (1 - ablation refusal on harmful)
    + (addition refusal on harmless); ties go to the earlier layer.
    """
    cutoff = int(round(n_layers * (1 - skip_last_frac)))
    cand = df[df["layer"] < cutoff].copy()
    cand["score"] = (1 - cand["ablation_refusal_rate_harmful"]) + cand["addition_refusal_rate_harmless"]
    best = cand.sort_values(["score", "layer"], ascending=[False, True]).iloc[0]
    return int(best["layer"])


def _write_jsonl(path: Path, prompts: list[str], completions: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p, c in zip(prompts, completions):
            f.write(json.dumps({"instruction": p, "completion": c}, ensure_ascii=False) + "\n")


def run_experiment(cfg: ExperimentConfig, bundle: ModelBundle | None = None) -> dict:
    t0 = time.time()
    set_seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    splits = get_splits(cfg.splits_path, cfg.n_train, cfg.n_val, cfg.seed)

    if bundle is None:
        bundle = load_model(
            cfg.model_name,
            dtype=getattr(torch, cfg.dtype),
            device_map=cfg.device_map,
            attn_implementation=cfg.attn_implementation,
            hf_token=cfg.hf_token,
        )
    fmt = lambda xs: format_prompts(bundle, xs)  # noqa: E731

    # 1. Activations at the chosen position on the training sets, every layer.
    harmful_acts = capture_residuals(bundle, fmt(splits["harmful_train"]), cfg.position, cfg.batch_size)
    harmless_acts = capture_residuals(bundle, fmt(splits["harmless_train"]), cfg.position, cfg.batch_size)

    # 2. Candidate directions, one per layer.
    r = difference_in_means(harmful_acts, harmless_acts)  # [n_layers, d_model]
    r_hat = unit(r)
    torch.save({"r": r, "r_hat": r_hat, "position": cfg.position}, cfg.out_dir / "directions.pt")

    # 3. Baselines on the held-out sets.
    harmful_val, harmless_val = fmt(splits["harmful_val"]), fmt(splits["harmless_val"])
    gen = lambda prompts, **kw: generate(bundle, prompts, cfg.max_new_tokens, cfg.batch_size, **kw)  # noqa: E731
    base_harmful = gen(harmful_val)
    base_harmless = gen(harmless_val)
    baseline_harmful = refusal_rate(base_harmful)
    baseline_harmless = refusal_rate(base_harmless)
    _write_jsonl(cfg.out_dir / "completions_harmful_baseline.jsonl", splits["harmful_val"], base_harmful)
    _write_jsonl(cfg.out_dir / "completions_harmless_baseline.jsonl", splits["harmless_val"], base_harmless)

    # 4. Per-layer sweep: ablate r_hat[l] everywhere; add r[l] at layer l.
    layers = cfg.layers if cfg.layers is not None else list(range(bundle.n_layers))
    rows = []
    for layer in layers:
        abl = gen(harmful_val, ablate_dir=r_hat[layer])
        add = gen(harmless_val, add_vec=r[layer], add_layer=layer)
        rows.append(
            {
                "layer": layer,
                "ablation_refusal_rate_harmful": refusal_rate(abl),
                "addition_refusal_rate_harmless": refusal_rate(add),
                "direction_norm": float(r[layer].norm()),
            }
        )
        print(f"layer {layer:3d}: ablation {rows[-1]['ablation_refusal_rate_harmful']:.3f}  "
              f"addition {rows[-1]['addition_refusal_rate_harmless']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.insert(1, "baseline_refusal_rate_harmful", baseline_harmful)
    df.insert(2, "baseline_refusal_rate_harmless", baseline_harmless)
    df.to_csv(cfg.out_dir / "refusal_by_layer.csv", index=False)

    # 5. Chosen layer: re-generate and keep the completions.
    chosen = choose_layer(df, bundle.n_layers)
    abl = gen(harmful_val, ablate_dir=r_hat[chosen])
    add = gen(harmless_val, add_vec=r[chosen], add_layer=chosen)
    _write_jsonl(cfg.out_dir / "completions_harmful_ablation.jsonl", splits["harmful_val"], abl)
    _write_jsonl(cfg.out_dir / "completions_harmless_addition.jsonl", splits["harmless_val"], add)

    from .plot import plot_refusal_by_layer

    short = cfg.model_name.split("/")[-1]
    plot_refusal_by_layer(
        df, baseline_harmful, baseline_harmless, chosen,
        title=f"{short}: refusal rate by candidate layer (n={cfg.n_val} per set, {cfg.max_new_tokens} tokens)",
        out_path=cfg.out_dir / "refusal_by_layer.png",
    )

    summary = {
        "package_version": __version__,
        "model": cfg.model_name,
        "dtype": cfg.dtype,
        "n_layers": bundle.n_layers,
        "d_model": bundle.d_model,
        "position": cfg.position,
        "position_note": "last token of the chat-templated prompt (left padding)",
        "chosen_layer": chosen,
        "baseline_refusal_rate_harmful": baseline_harmful,
        "baseline_refusal_rate_harmless": baseline_harmless,
        "chosen_ablation_refusal_rate_harmful": refusal_rate(abl),
        "chosen_addition_refusal_rate_harmless": refusal_rate(add),
        "config": config_summary(cfg),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "nnsight": __import__("nnsight").__version__,
        },
        "device": str(bundle.device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(cfg.out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return {"summary": summary, "df": df, "r": r, "r_hat": r_hat, "bundle": bundle}
