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
