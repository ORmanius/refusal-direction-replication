"""CPU smoke test for the extensions on pythia-70m: shapes, invariants, and that the plots
(including the GIF) are written. No dataset download is needed."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from refusal_direction.direction import difference_in_means, unit
from refusal_direction.extensions import (
    capability_by_layer,
    direction_cosine,
    dynamics_study,
    kl_divergence,
    last_token_logprobs,
    reference_nll,
    token_projections,
)
from refusal_direction.intervention import capture_residuals
from refusal_direction.model import load_model
from refusal_direction.plot import plot_capability_by_layer, plot_direction_similarity, plot_dynamics

GROUP_A = ["The weather today is", "My favourite colour is", "The train leaves at", "She opened the"]
GROUP_B = ["Please list three fruits", "Describe a quiet morning", "Name a famous river", "Write one sentence about tea"]
PAIRS = [("Name a colour.", "Blue."), ("Name a fruit.", "An apple."), ("Count to three.", "One, two, three.")]


@pytest.fixture(scope="module")
def bundle():
    return load_model("EleutherAI/pythia-70m", dtype=torch.float32, device_map="cpu")


@pytest.fixture(scope="module")
def directions(bundle):
    a = capture_residuals(bundle, GROUP_A, batch_size=4)
    b = capture_residuals(bundle, GROUP_B, batch_size=4)
    r = difference_in_means(a, b)
    return r, unit(r)


def test_direction_cosine(directions):
    _, r_hat = directions
    cos = direction_cosine(r_hat)
    assert cos.shape == (r_hat.shape[0],) * 2
    assert np.allclose(np.diag(cos), 1.0, atol=1e-4)
    assert np.allclose(cos, cos.T, atol=1e-5)


def test_logprobs_kl_and_nll(bundle, directions):
    _, r_hat = directions
    base = last_token_logprobs(bundle, GROUP_B, batch_size=4)
    assert base.shape[0] == len(GROUP_B)
    assert torch.allclose(base.exp().sum(-1), torch.ones(len(GROUP_B)), atol=1e-3)
    assert torch.all(kl_divergence(base, base).abs() < 1e-5)
    abl = last_token_logprobs(bundle, GROUP_B, batch_size=4, ablate_dir=r_hat[2])
    assert torch.all(kl_divergence(base, abl) >= -1e-5)
    nll = reference_nll(bundle, PAIRS, batch_size=2)
    assert nll.shape == (len(PAIRS),) and torch.all(nll > 0)


def test_capability_table_and_plot(bundle, directions, tmp_path):
    _, r_hat = directions
    df = capability_by_layer(bundle, GROUP_B, PAIRS, r_hat, layers=[0, 1], batch_size=4, nll_batch_size=2)
    assert list(df["layer"]) == [-1, 0, 1]
    assert df.loc[0, "kl_mean"] == 0.0
    plot_capability_by_layer(df, chosen_layer=1, title="t", out_path=tmp_path / "cap.png")
    plot_direction_similarity(direction_cosine(r_hat), 1, "t", tmp_path / "cos.png")
    assert (tmp_path / "cap.png").exists() and (tmp_path / "cos.png").exists()


def test_token_projections_ablation_is_zero(bundle, directions):
    _, r_hat = directions
    projs = token_projections(bundle, GROUP_A[:2], r_hat[2], batch_size=2, ablate_dir=r_hat[2])
    assert len(projs) == 2
    for p in projs:
        assert p.shape[0] == bundle.n_layers
        assert np.abs(p).max() < 1e-3 * float(r_hat.shape[1]) ** 0.5


def test_dynamics_and_gif(bundle, directions, tmp_path):
    r, r_hat = directions
    dyn = dynamics_study(bundle, GROUP_A, GROUP_B, r, r_hat, chosen=2, max_new_tokens=6, batch_size=4)
    assert set(dyn["curves"]) == {"harmful_baseline", "harmful_ablated", "harmless_baseline", "harmless_added"}
    ex = dyn["example"]["harmless_added"]
    assert ex["proj"].shape == (bundle.n_layers, len(ex["tokens"]))
    plot_dynamics(dyn, "t", tmp_path / "dyn.png")
    assert (tmp_path / "dyn.png").stat().st_size > 0


def test_dose_response_and_gif(bundle, directions, tmp_path):
    from refusal_direction.extensions import dose_response_study
    from refusal_direction.plot import plot_dose_response

    r, r_hat = directions
    dose = dose_response_study(
        bundle, GROUP_A, GROUP_B, r, r_hat, layer=2,
        harmless_example=(GROUP_B[0], GROUP_B[0]), benign_example=(GROUP_A[0], GROUP_A[0]),
        add_coefs=[0.0, 1.0], abl_coefs=[0.0, 1.0], max_new_tokens=5, batch_size=4,
    )
    rows = dose["rows"]
    assert list(rows["kind"]) == ["addition_harmless"] * 2 + ["ablation_harmful"] * 2
    assert len(dose["examples"]) == 4 and all(0 <= x <= 1 for x in rows["refusal_rate"])
    plot_dose_response(dose, "t", tmp_path / "dose.png", tmp_path / "dose.gif", fps=4)
    assert (tmp_path / "dose.png").stat().st_size > 0 and (tmp_path / "dose.gif").stat().st_size > 0


def test_select_layer_with_kl():
    import pandas as pd
    from refusal_direction.extensions import select_layer_with_kl

    sweep = pd.DataFrame({
        "layer": [0, 1, 2, 3, 4],
        "ablation_refusal_rate_harmful": [1.0, 0.0, 0.0, 0.0, 0.0],
        "addition_refusal_rate_harmless": [0.0, 1.0, 1.0, 1.0, 1.0],
    })
    cap = pd.DataFrame({"layer": [-1, 0, 1, 2, 3, 4], "kl_mean": [0, 0.01, 0.5, 0.05, 0.02, 0.01],
                        "kl_max": [0] * 6, "reference_nll": [1.0] * 6})
    sel = select_layer_with_kl(sweep, cap, n_layers=5, skip_last_frac=0.2, kl_max=0.1)
    assert sel["layer"] == 3 and sel["kl_filter_satisfied"]  # layer 4 is cut as a late layer
    sel = select_layer_with_kl(sweep, cap, n_layers=5, skip_last_frac=0.2, kl_max=0.001)
    assert sel["layer"] == 3 and not sel["kl_filter_satisfied"]
