"""One plot per model: refusal rate on the validation sets as a function of layer."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_refusal_by_layer(
    df: pd.DataFrame,
    baseline_harmful: float,
    baseline_harmless: float,
    chosen_layer: int,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.plot(
        df["layer"], df["ablation_refusal_rate_harmful"],
        marker="o", ms=3.5, lw=1.4, color="#1f77b4",
        label="harmful val, direction ablated (all layers)",
    )
    ax.plot(
        df["layer"], df["addition_refusal_rate_harmless"],
        marker="s", ms=3.5, lw=1.4, color="#d62728",
        label="harmless val, direction added (this layer)",
    )
    ax.axhline(baseline_harmful, ls="--", lw=1, color="#1f77b4", alpha=0.6,
               label="harmful val, no intervention")
    ax.axhline(baseline_harmless, ls="--", lw=1, color="#d62728", alpha=0.6,
               label="harmless val, no intervention")
    ax.axvline(chosen_layer, color="grey", lw=1, ls=":", label=f"chosen layer {chosen_layer}")
    ax.set_xlabel("layer whose output defines the direction (resid_post, last prompt token)")
    ax.set_ylabel("refusal rate (substring match)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(df["layer"][:: max(1, len(df) // 14)])
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# Extensions ----------------------------------------------------------------------------


def plot_direction_similarity(cos, chosen_layer: int, title: str, out_path: Path) -> None:
    import numpy as np

    n = cos.shape[0]
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    im = ax.imshow(cos, vmin=-1, vmax=1, cmap="RdBu_r", origin="lower")
    ax.axhline(chosen_layer, color="k", lw=0.8, ls=":")
    ax.axvline(chosen_layer, color="k", lw=0.8, ls=":")
    step = max(1, n // 14)
    ax.set_xticks(np.arange(0, n, step))
    ax.set_yticks(np.arange(0, n, step))
    ax.set_xlabel("layer")
    ax.set_ylabel("layer")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_capability_by_layer(df: pd.DataFrame, chosen_layer: int, title: str, out_path: Path) -> None:
    base = df[df["layer"] == -1].iloc[0]
    d = df[df["layer"] >= 0]
    fig, ax1 = plt.subplots(figsize=(7, 4), dpi=150)
    ax1.plot(d["layer"], d["kl_mean"], marker="o", ms=3.5, lw=1.4, color="#2ca02c",
             label="KL(baseline || ablated), harmless val, last prompt token")
    ax1.set_ylabel("mean KL divergence (nats)", color="#2ca02c")
    ax1.set_xlabel("layer whose direction is ablated (at every layer)")
    ax1.set_ylim(bottom=0)
    ax2 = ax1.twinx()
    ax2.plot(d["layer"], d["reference_nll"], marker="s", ms=3.5, lw=1.4, color="#9467bd",
             label="NLL of Alpaca reference answers (per token)")
    ax2.axhline(base["reference_nll"], ls="--", lw=1, color="#9467bd", alpha=0.6, label="NLL, no ablation")
    ax2.set_ylabel("reference-answer NLL (nats / token)", color="#9467bd")
    ax1.axvline(chosen_layer, color="grey", lw=1, ls=":", label=f"chosen layer {chosen_layer}")
    ax1.set_xticks(d["layer"][:: max(1, len(d) // 14)])
    ax1.grid(alpha=0.25)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    ax1.set_title(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _dynamics_figure(dyn: dict, title: str):
    """Build the static dynamics figure; returns (fig, artists, n_frames) for optional animation."""
    import numpy as np
    from matplotlib.gridspec import GridSpec

    curves, ex, chosen = dyn["curves"], dyn["example"], dyn["chosen"]
    ks = np.arange(-8, dyn["max_new_tokens"])
    fig = plt.figure(figsize=(11, 6.2), dpi=100)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.35], hspace=0.55, wspace=0.28)
    axes = {"harmful": fig.add_subplot(gs[0, 0]), "harmless": fig.add_subplot(gs[1, 0]),
            "harmless_baseline": fig.add_subplot(gs[0, 1]), "harmless_added": fig.add_subplot(gs[1, 1])}
    lines = {}
    specs = [("harmful", "harmful_baseline", "no intervention", "#1f77b4", "-"),
             ("harmful", "harmful_ablated", "direction ablated", "#1f77b4", "--"),
             ("harmless", "harmless_baseline", "no intervention", "#d62728", "-"),
             ("harmless", "harmless_added", f"direction added at layer {chosen}", "#d62728", "--")]
    for panel, cond, label, color, ls in specs:
        df = curves[cond]
        (ln,) = axes[panel].plot(df["k"], df["mean"], color=color, ls=ls, lw=1.6, label=label)
        lines[cond] = (ln, df["k"].to_numpy(), df["mean"].to_numpy())
    for panel, label in (("harmful", "harmful prompts (AdvBench val, n={n})"),
                         ("harmless", "harmless prompts (Alpaca val, n={n})")):
        ax = axes[panel]
        name = label.format(n=int(curves[f"{panel}_baseline"]["n"].max()))
        ax.axvline(-0.5, color="grey", lw=1, ls=":")
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)
        ax.set_xlabel("token index (0 = first generated token)")
        ax.set_ylabel(f"projection, layer {chosen}")
        ax.set_title(f"mean projection onto the refusal direction, {name}", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)
        lo = min(np.nanmin(v[2]) for c, v in lines.items() if c.startswith(panel))
        hi = max(np.nanmax(v[2]) for c, v in lines.items() if c.startswith(panel))
        pad = 0.08 * (hi - lo + 1e-6)
        ax.set_ylim(lo - pad, hi + pad)

    images = {}
    # Colour scale from the displayed window only, robust to the few very large values that
    # early tokens (BOS, template tokens) carry in every layer.
    shown = []
    for c in ("harmless_baseline", "harmless_added"):
        n_p = ex[c]["prompt_len"]
        shown.append(np.abs(ex[c]["proj"][:, max(0, n_p - 8):]))
    vmax = float(np.percentile(np.concatenate([s.ravel() for s in shown]), 99)) or 1.0
    for cond, label in (("harmless_baseline", "no intervention"),
                        ("harmless_added", f"direction added at layer {chosen}")):
        e, ax = ex[cond], axes[cond]
        proj, n_p = e["proj"], e["prompt_len"]
        start = max(0, n_p - 8)
        sub = proj[:, start:]
        im = ax.imshow(sub, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower",
                       interpolation="nearest")
        ax.axvline(n_p - start - 0.5, color="k", lw=1, ls=":")
        toks = [t.replace("\n", "\\n") for t in e["tokens"][start:]]
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels(toks, rotation=90, fontsize=5.5)
        ax.set_ylabel("layer")
        ax.set_title(f"one harmless prompt, {label} (refusal = {e['refusal']})", fontsize=9)
        images[cond] = (im, sub, n_p - start)
    fig.colorbar(images["harmless_added"][0], ax=[axes["harmless_baseline"], axes["harmless_added"]],
                 fraction=0.03, pad=0.02, label="projection onto refusal direction")
    fig.suptitle(title, fontsize=11)
    return fig, lines, images, len(ks)


def plot_dynamics(dyn: dict, title: str, out_png: Path) -> None:
    fig, _, _, _ = _dynamics_figure(dyn, title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# Dose-response: the animated variable (intervention strength) is not on either axis, so every
# frame shows a different completion and a different point on the curves.

_DOSE_ROWS = {
    "addition_harmless": {
        "example": "addition_harmless",
        "title": "harmless prompts (Alpaca val): add a * r at layer {layer}",
        "formula": "x <- x + a * r   (a = {a:.1f}, layer {layer})",
        "xlabel": "a (1.0 = the raw difference-in-means vector)",
    },
    "ablation_harmful": {
        "example": "ablation_benign_example",
        "title": "harmful prompts (AdvBench val): remove a * (x . r_hat) r_hat, all layers",
        "formula": "x <- x - a * (x . r_hat) r_hat   (a = {a:.1f}, every layer)",
        "xlabel": "a (1.0 = full ablation, the paper's intervention)",
    },
}


def _wrap(text: str, width: int = 74, limit: int = 420) -> str:
    import textwrap

    text = " ".join(text.split())
    if len(text) > limit:
        text = text[:limit].rstrip() + " ..."
    return textwrap.fill(text, width=width)


def _dose_curves(ax, rows: pd.DataFrame, kind: str, layer: int) -> None:
    d = rows[rows["kind"] == kind]
    spec = _DOSE_ROWS[kind]
    ax.plot(d["coef"], d["refusal_rate"], marker="o", ms=4, lw=1.6, color="#d62728",
            label="refusal rate (substring match)")
    ax.plot(d["coef"], d["p_refusal_first_token"], marker="s", ms=3.5, lw=1.2, color="#9467bd",
            label="P(first token is a refusal opener)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(d["coef"].min() - 0.05, d["coef"].max() + 0.05)
    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel("rate / probability")
    ax.set_title(spec["title"].format(layer=layer), fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="center right")


def plot_dose_response(dose: dict, title: str, out_png: Path, out_gif: Path | None = None, fps: float = 1.0) -> None:
    """Static: the two dose-response curves. GIF: the same curves with a moving marker and, beside
    them, the example completion generated at that strength."""
    from matplotlib import animation
    from matplotlib.gridspec import GridSpec

    rows, examples, layer = dose["rows"], dose["examples"], dose["layer"]
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    for ax, kind in zip(axes, _DOSE_ROWS):
        _dose_curves(ax, rows, kind, layer)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    if out_gif is None:
        return

    fig = plt.figure(figsize=(12, 6.6), dpi=100)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.45], hspace=0.6, wspace=0.15)
    state = {}
    for i, kind in enumerate(_DOSE_ROWS):
        spec = _DOSE_ROWS[kind]
        ax = fig.add_subplot(gs[i, 0])
        _dose_curves(ax, rows, kind, layer)
        (m1,) = ax.plot([], [], "o", ms=11, mfc="none", mec="#d62728", mew=2)
        (m2,) = ax.plot([], [], "s", ms=10, mfc="none", mec="#9467bd", mew=2)
        tx = fig.add_subplot(gs[i, 1])
        tx.axis("off")
        exs = [e for e in examples if e["kind"] == spec["example"]]
        # Header: prompt (at most two wrapped lines), blank line, formula. Body starts below it.
        head = tx.text(0.0, 1.0, "", va="top", ha="left", fontsize=8.5, family="monospace", transform=tx.transAxes)
        body = tx.text(0.0, 0.62, "", va="top", ha="left", fontsize=8.5, family="monospace", transform=tx.transAxes)
        state[kind] = {"m1": m1, "m2": m2, "head": head, "body": body, "exs": exs, "spec": spec}

    def show(kind: str, a: float):
        st = state[kind]
        r = rows[(rows["kind"] == kind) & (rows["coef"] == a)].iloc[0]
        st["m1"].set_data([a], [r["refusal_rate"]])
        st["m2"].set_data([a], [r["p_refusal_first_token"]])
        e = next(e for e in st["exs"] if e["coef"] == a)
        st["head"].set_text("PROMPT: " + _wrap(e["instruction"], 74, 140) + "\n\n"
                            + st["spec"]["formula"].format(a=a, layer=layer)
                            + f"   refusal rate on the set: {r['refusal_rate']:.2f}")
        st["body"].set_text(("REFUSAL\n" if e["refusal"] else "COMPLIES\n") + _wrap(e["completion"]))
        st["body"].set_color("#b71c1c" if e["refusal"] else "#1b5e20")

    frames = [(k, a) for k in _DOSE_ROWS for a in rows[rows["kind"] == k]["coef"].tolist()]
    frames += [frames[-1]] * 3  # hold the last frame

    def update(frame):
        kind, a = frame
        if frame == frames[0]:
            for k in _DOSE_ROWS:
                show(k, rows[rows["kind"] == k]["coef"].min())
        show(kind, a)
        return []

    fig.suptitle(title, fontsize=11)
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(out_gif, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
