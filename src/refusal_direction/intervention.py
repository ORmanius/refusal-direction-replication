"""Activation capture and interventions (directional ablation, activation addition).

Both interventions follow Arditi et al. (2024):

* Directional ablation removes the component along the unit direction r_hat from the
  residual stream at every layer and every token position:
      x <- x - (x . r_hat) r_hat
  Applied to the embedding output and to every decoder layer output, so the residual
  stream entering every layer is orthogonal to r_hat (equivalent to the paper's
  orthogonalisation of the embedding, attention-output and MLP-output weights).

* Activation addition adds the raw difference-in-means vector r (not normalised) at
  a single layer, at every token position:
      x <- x + r
"""

from __future__ import annotations

import torch

from .model import ModelBundle, tokenize, value


def _resid(bundle: ModelBundle, module) -> torch.Tensor:
    """Proxy for the residual-stream tensor produced by a module."""
    out = module.output
    return out[0] if (bundle.output_is_tuple and module is not bundle.embed) else out


def _set_resid(bundle: ModelBundle, module, new: torch.Tensor) -> None:
    if bundle.output_is_tuple and module is not bundle.embed:
        module.output[0][:] = new
    else:
        module.output[:] = new


def _apply_ablation(bundle: ModelBundle, direction: torch.Tensor) -> None:
    d = direction.to(bundle.device, bundle.dtype)
    for module in [bundle.embed, *bundle.layers]:
        x = _resid(bundle, module)
        proj = (x * d).sum(-1, keepdim=True)
        _set_resid(bundle, module, x - proj * d)


def _apply_addition(bundle: ModelBundle, vec: torch.Tensor, layer: int) -> None:
    v = vec.to(bundle.device, bundle.dtype)
    module = bundle.layers[layer]
    _set_resid(bundle, module, _resid(bundle, module) + v)


def _apply(bundle, ablate_dir, add_vec, add_layer) -> None:
    if ablate_dir is not None:
        _apply_ablation(bundle, ablate_dir)
    if add_vec is not None:
        if add_layer is None:
            raise ValueError("add_layer is required with add_vec")
        _apply_addition(bundle, add_vec, add_layer)


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


@torch.no_grad()
def capture_residuals(
    bundle: ModelBundle,
    prompts: list[str],
    position: int = -1,
    batch_size: int = 16,
    ablate_dir: torch.Tensor | None = None,
) -> torch.Tensor:
    """Residual stream at `position` after every decoder layer.

    Returns a float32 CPU tensor of shape [n_prompts, n_layers, d_model]. Index l is the
    output of decoder layer l (resid_post of layer l, i.e. the input to layer l+1).
    With left padding, position -1 is the last real token of every prompt.
    """
    out_batches = []
    for batch in _batches(prompts, batch_size):
        enc = tokenize(bundle, batch).to(bundle.device)
        with bundle.lm.trace(enc):
            _apply(bundle, ablate_dir, None, None)
            saved = [
                _resid(bundle, layer)[:, position, :].float().cpu().save()
                for layer in bundle.layers
            ]
        out_batches.append(torch.stack([value(s) for s in saved], dim=1))
    return torch.cat(out_batches, dim=0)


@torch.no_grad()
def generate(
    bundle: ModelBundle,
    prompts: list[str],
    max_new_tokens: int = 64,
    batch_size: int = 16,
    ablate_dir: torch.Tensor | None = None,
    add_vec: torch.Tensor | None = None,
    add_layer: int | None = None,
) -> list[str]:
    """Greedy generation with an optional intervention active at every generated token."""
    tok = bundle.tokenizer
    completions: list[str] = []
    for batch in _batches(prompts, batch_size):
        enc = tokenize(bundle, batch).to(bundle.device)
        n_in = enc["input_ids"].shape[1]
        with bundle.lm.generate(
            enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tok.pad_token_id,
        ) as tracer:
            with tracer.all():
                _apply(bundle, ablate_dir, add_vec, add_layer)
            out = bundle.lm.generator.output.save()
        ids = value(out)[:, n_in:]
        completions.extend(tok.batch_decode(ids, skip_special_tokens=True))
    return completions
