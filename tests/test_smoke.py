"""CPU smoke test on EleutherAI/pythia-70m: every stage runs and the ablation is exact.

Uses neutral toy prompts (no harmful content) in two arbitrary groups; the point is to
exercise shapes and the intervention mechanics, not to measure refusal.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from refusal_direction import data, scoring
from refusal_direction.direction import difference_in_means, unit
from refusal_direction.intervention import capture_residuals, generate
from refusal_direction.model import format_prompts, load_model

GROUP_A = ["Describe a sunrise over the ocean.", "Explain how bread rises.", "Name three primary colours.", "Write a haiku about rain."]
GROUP_B = ["List the planets in order.", "What is a prime number?", "Translate 'hello' to French.", "Summarise the water cycle."]


@pytest.fixture(scope="module")
def bundle():
    return load_model("EleutherAI/pythia-70m", dtype=torch.float32, device_map="cpu")


def test_scoring():
    assert scoring.is_refusal("I'm sorry, but I can't help with that.")
    assert not scoring.is_refusal("Sure, here is a short poem about rain.")
    assert scoring.refusal_rate(["I cannot", "ok"]) == 0.5


def test_hash_check_rejects_tampered_file(tmp_path, monkeypatch):
    bad = tmp_path / "x.csv"
    bad.write_text("goal,target\nfoo,bar\n")
    monkeypatch.setitem(
        data.SOURCES, "tampered", data.Source(url="file:///unused", sha256="0" * 64, filename="x.csv")
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        data.download("tampered", raw_dir=tmp_path)
    assert not bad.exists()
    assert data.sha256_of(tmp_path / "x.csv") if (tmp_path / "x.csv").exists() else True


def test_sha256_of(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert data.sha256_of(p) == hashlib.sha256(b"abc").hexdigest()


def test_pipeline_stages(bundle):
    a = capture_residuals(bundle, format_prompts(bundle, GROUP_A), batch_size=2)
    b = capture_residuals(bundle, format_prompts(bundle, GROUP_B), batch_size=2)
    assert a.shape == (4, bundle.n_layers, bundle.d_model)
    assert b.shape == a.shape

    r = difference_in_means(a, b)
    r_hat = unit(r)
    assert r.shape == (bundle.n_layers, bundle.d_model)
    assert torch.allclose(r_hat.norm(dim=-1), torch.ones(bundle.n_layers), atol=1e-4)

    # Ablation removes the direction from every layer's output (projection ~ 0).
    layer = bundle.n_layers // 2
    ablated = capture_residuals(bundle, format_prompts(bundle, GROUP_A), batch_size=2, ablate_dir=r_hat[layer])
    proj = (ablated @ r_hat[layer])  # [n_prompts, n_layers]
    scale = ablated.norm(dim=-1).max()
    assert proj.abs().max() < 1e-3 * scale, proj

    # Generation runs with and without interventions and returns one string per prompt.
    plain = generate(bundle, GROUP_A[:2], max_new_tokens=5, batch_size=2)
    abl = generate(bundle, GROUP_A[:2], max_new_tokens=5, batch_size=2, ablate_dir=r_hat[layer])
    add = generate(bundle, GROUP_B[:2], max_new_tokens=5, batch_size=2, add_vec=r[layer], add_layer=layer)
    for out in (plain, abl, add):
        assert len(out) == 2 and all(isinstance(s, str) for s in out)
