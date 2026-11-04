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


def plot_dynamics(dyn: dict, title: str, out_png: Path, out_gif: Path | None = None, fps: int = 8) -> None:
    import numpy as np
    from matplotlib import animation

    fig, lines, images, n_frames = _dynamics_figure(dyn, title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    if out_gif is None:
        plt.close(fig)
        return

    def show_upto(k):
        # Reveal the curves up to token index k and the heatmaps up to the same generated token.
        for ln, xs, ys in lines.values():
            m = xs <= k
            ln.set_data(xs[m], ys[m])
        for im, sub, gen_start in images.values():
            masked = sub.copy()
            cut = gen_start + k + 1
            if cut < masked.shape[1]:
                masked[:, cut:] = np.nan
            im.set_data(masked)
        return [v[0] for v in lines.values()] + [v[0] for v in images.values()]

    ks = list(range(-8, dyn["max_new_tokens"]))
    hold = [ks[-1]] * (2 * fps)  # pause on the final frame
    anim = animation.FuncAnimation(fig, show_upto, frames=ks + hold, blit=False)
    anim.save(out_gif, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
