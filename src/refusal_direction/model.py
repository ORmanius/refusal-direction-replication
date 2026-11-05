"""Model loading via NNsight and prompt formatting.

NNsight wraps the Hugging Face model unchanged (no weight re-folding), so a 1.5B model
in fp16 fits a 6 GB GPU and the same code runs Qwen2.5, Gemma-2 and Pythia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from nnsight import TransformersModel

# Where the decoder layers and token embedding live, per Hugging Face model_type.
_PATHS: dict[str, tuple[str, str]] = {
    "qwen2": ("model.layers", "model.embed_tokens"),
    "gemma2": ("model.layers", "model.embed_tokens"),
    "gemma": ("model.layers", "model.embed_tokens"),
    "llama": ("model.layers", "model.embed_tokens"),
    "mistral": ("model.layers", "model.embed_tokens"),
    "gpt_neox": ("gpt_neox.layers", "gpt_neox.embed_in"),
    "gpt2": ("transformer.h", "transformer.wte"),
}

# The MLP output projection inside a decoder layer; its input is the vector of neuron
# activations (used by the neuron-effects extension).
_MLP_OUT: dict[str, str] = {
    "qwen2": "mlp.down_proj",
    "gemma2": "mlp.down_proj",
    "gemma": "mlp.down_proj",
    "llama": "mlp.down_proj",
    "mistral": "mlp.down_proj",
    "gpt_neox": "mlp.dense_4h_to_h",
    "gpt2": "mlp.c_proj",
}


def value(x: Any) -> Any:
    """Return the concrete value of a saved NNsight proxy (works across nnsight versions)."""
    if isinstance(x, (torch.Tensor, tuple, list)):
        return x
    return getattr(x, "value", x)


def _resolve(root: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        root = getattr(root, part)
    return root


@dataclass
class ModelBundle:
    lm: TransformersModel
    name: str
    n_layers: int
    d_model: int
    layer_path: str
    embed_path: str
    output_is_tuple: bool
    chat: bool
    model_type: str = ""

    @property
    def tokenizer(self):
        return self.lm.tokenizer

    def mlp_out(self, layer: int):
        """The MLP output projection module of decoder layer `layer` (its input = neuron activations)."""
        return _resolve(self.layers[layer], _MLP_OUT[self.model_type])

    @property
    def layers(self):
        return _resolve(self.lm, self.layer_path)

    @property
    def embed(self):
        return _resolve(self.lm, self.embed_path)

    @property
    def hf_model(self) -> torch.nn.Module:
        return self.lm.pipeline.model

    @property
    def device(self) -> torch.device:
        return next(self.hf_model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.hf_model.parameters()).dtype


def load_model(
    name: str,
    dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    attn_implementation: str | None = None,
    hf_token: str | None = None,
) -> ModelBundle:
    kwargs: dict[str, Any] = {"device_map": device_map, "dtype": dtype, "dispatch": True}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if hf_token:
        kwargs["token"] = hf_token
    lm = TransformersModel(name, task="text-generation", **kwargs)
    tok = lm.tokenizer
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = lm.config
    model_type = getattr(cfg, "model_type", "")
    if model_type not in _PATHS:
        raise ValueError(f"unsupported model_type {model_type!r}; add it to model._PATHS")
    layer_path, embed_path = _PATHS[model_type]
    layers = _resolve(lm, layer_path)
    n_layers = len(layers)
    d_model = getattr(cfg, "hidden_size", None) or cfg.n_embd
    chat = tok.chat_template is not None

    # Decoder layers return either a tensor or a tuple whose first element is the
    # residual stream, depending on the transformers version. Probe once.
    with torch.no_grad():
        with lm.trace("probe"):
            probe = layers[0].output.save()
    output_is_tuple = isinstance(value(probe), tuple)

    return ModelBundle(
        lm=lm,
        name=name,
        n_layers=n_layers,
        d_model=int(d_model),
        layer_path=layer_path,
        embed_path=embed_path,
        output_is_tuple=output_is_tuple,
        chat=chat,
        model_type=model_type,
    )


def format_prompts(bundle: ModelBundle, instructions: list[str]) -> list[str]:
    """Wrap each instruction in the model's chat template with an empty assistant turn.

    No system prompt is passed. Note that Qwen2.5-Instruct's template inserts its own
    default system message when none is given; we keep the template as shipped.
    Models without a chat template (Pythia, GPT-2) get the raw instruction.
    """
    tok = bundle.tokenizer
    if not bundle.chat:
        return list(instructions)
    return [
        tok.apply_chat_template(
            [{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True
        )
        for x in instructions
    ]


def tokenize(bundle: ModelBundle, texts: list[str]):
    """Left-padded batch encoding. Chat-templated text already contains any BOS token."""
    return bundle.tokenizer(
        texts, return_tensors="pt", padding=True, add_special_tokens=not bundle.chat
    )
