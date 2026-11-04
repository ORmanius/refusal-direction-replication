"""Difference-in-means refusal directions, one candidate per layer."""

from __future__ import annotations

import torch


def difference_in_means(harmful_acts: torch.Tensor, harmless_acts: torch.Tensor) -> torch.Tensor:
    """Return r[l] = mean_harmful[l] - mean_harmless[l] for every layer.

    Inputs have shape [n_prompts, n_layers, d_model] (one activation per prompt at the
    chosen token position). Output has shape [n_layers, d_model] and is *not* normalised:
    the paper adds this raw vector for activation addition.
    """
    if harmful_acts.ndim != 3 or harmless_acts.ndim != 3:
        raise ValueError("expected [n_prompts, n_layers, d_model] tensors")
    return harmful_acts.float().mean(0) - harmless_acts.float().mean(0)


def unit(vectors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalise each row to unit L2 norm (the direction used for ablation)."""
    return vectors / (vectors.norm(dim=-1, keepdim=True) + eps)
