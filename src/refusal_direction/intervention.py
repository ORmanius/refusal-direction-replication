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

NNsight runs the body of a `with model.trace(...)` block in lockstep with the forward
pass and requires module outputs to be requested in execution order, so every
intervention here is a single ordered walk over [embedding, layer 0, ..., layer L-1]
that reads, edits and (optionally) records each module's output exactly once.
"""

from __future__ import annotations

import warnings

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


def _walk(
    bundle: ModelBundle,
    ablate_dir: torch.Tensor | None = None,
    add_vec: torch.Tensor | None = None,
    add_layer: int | None = None,
    position: int | slice | None = None,
    ablate_scale: float = 1.0,
    add_scale: float = 1.0,
) -> list[torch.Tensor] | None:
    """One in-order pass over the residual stream; must be called inside a trace.

    Applies the requested interventions and, if `position` is given, returns the
    (post-intervention) residual at that token position (or slice of positions) for
    every decoder layer. `ablate_scale` a removes only a fraction of the component
    (x <- x - a (x . r_hat) r_hat; a = 1 is full ablation) and `add_scale` scales the
    added vector (x <- x + a r); both exist for the dose-response extension.
    """
    if add_vec is not None and add_layer is None:
        raise ValueError("add_layer is required with add_vec")
    d = ablate_dir.to(bundle.device, bundle.dtype) if ablate_dir is not None else None
    v = add_vec.to(bundle.device, bundle.dtype) if add_vec is not None else None
    collected: list[torch.Tensor] = []

    modules = [bundle.embed, *bundle.layers]
    for i, module in enumerate(modules):
        layer = i - 1  # -1 is the embedding
        needs_edit = d is not None or (v is not None and layer == add_layer)
        needs_read = needs_edit or (position is not None and layer >= 0)
        if not needs_read:
            continue
        x = _resid(bundle, module)
        if d is not None:
            x = x - ablate_scale * (x * d).sum(-1, keepdim=True) * d
        if v is not None and layer == add_layer:
            x = x + add_scale * v
        if needs_edit:
            _set_resid(bundle, module, x)
        if position is not None and layer >= 0:
            collected.append(x[:, position, :])
    return collected if position is not None else None


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
            cols = _walk(bundle, ablate_dir=ablate_dir, position=position)
            # NNsight hands back the values bound by name in this block: one tensor.
            stacked = torch.stack(cols, dim=1).float().cpu().save()
        out_batches.append(value(stacked))
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
    ablate_scale: float = 1.0,
    add_scale: float = 1.0,
) -> list[str]:
    """Greedy generation with an optional intervention active at every generated token."""
    tok = bundle.tokenizer
    completions: list[str] = []
    for batch in _batches(prompts, batch_size):
        enc = tokenize(bundle, batch).to(bundle.device)
        n_in = enc["input_ids"].shape[1]
        with warnings.catch_warnings(), bundle.lm.generate(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tok.pad_token_id,
        ) as tracer:
            # The intervention loop is cut short when generation stops early (EOS),
            # so the result is read in a separate invoke block, as NNsight advises.
            # NNsight warns about the cut-short loop every time; that is expected here.
            warnings.filterwarnings("ignore", message=".*was never reached.*")
            with tracer.invoke(enc):
                if ablate_dir is not None or add_vec is not None:
                    for _ in tracer.all():
                        _walk(bundle, ablate_dir=ablate_dir, add_vec=add_vec, add_layer=add_layer,
                              ablate_scale=ablate_scale, add_scale=add_scale)
            with tracer.invoke():
                out = tracer.result.save()
        ids = value(out)[:, n_in:]
        completions.extend(tok.batch_decode(ids, skip_special_tokens=True))
    return completions
