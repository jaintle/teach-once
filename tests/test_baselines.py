"""Tests for the six Sec. V-A transportation baselines (Phase 6)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from gpt_repro.baselines import (
    BASELINES,
    EnsembleNFBaseline,
    EnsembleNNBaseline,
    EnsembleRFBaseline,
    GPTransportBaseline,
    KMPBaseline,
    LaplacianEditingBaseline,
)
from gpt_repro.transport.nonlinear_gp import GPNonlinearResidual
from gpt_repro.utils import set_global_seed


def _toy_data(n: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    S = rng.uniform(-1.0, 1.0, size=(n, 2))
    T = S + 0.1 * np.stack([np.sin(2 * S[:, 1]), np.cos(2 * S[:, 0])], axis=1)
    return S, T


def _make(name: str):
    cls = BASELINES[name]
    if name == "gp":
        return cls(n_iter_default=120)
    if name in ("enn", "enf"):
        return cls(n_members=3, n_epochs=120)
    if name == "erf":
        return cls(n_estimators=5)
    return cls()


# ---------------------------------------------------------------------------
def test_all_baselines_fit_predict_shape():
    """Every baseline produces (M, d) outputs from (10, 2) training data."""
    set_global_seed(0)
    S, T = _toy_data(n=10)
    Xq = np.random.default_rng(1).uniform(-1, 1, size=(5, 2))
    for name in BASELINES:
        b = _make(name).fit(S, T)
        mean = b.transform(Xq)
        m2, std = b.predict_with_std(Xq)
        assert mean.shape == (5, 2), f"{name}: bad transform shape {mean.shape}"
        assert m2.shape == (5, 2),  f"{name}: bad predict_with_std mean shape {m2.shape}"
        if std is not None:
            assert std.shape == (5, 2), f"{name}: bad std shape {std.shape}"


# ---------------------------------------------------------------------------
def test_kmp_le_have_no_uncertainty():
    set_global_seed(0)
    S, T = _toy_data(n=10)
    Xq = np.random.default_rng(1).uniform(-1, 1, size=(5, 2))
    for cls in (KMPBaseline, LaplacianEditingBaseline):
        b = cls().fit(S, T)
        mean, std = b.predict_with_std(Xq)
        assert mean.shape == (5, 2)
        assert std is None
        assert cls.has_uncertainty is False
        assert cls.uncertainty_type == "none"


# ---------------------------------------------------------------------------
def test_ensemble_std_positive():
    """E-RF, E-NN, E-NF must have non-zero per-prediction std (member
    diversity)."""
    set_global_seed(0)
    S, T = _toy_data(n=15)
    Xq = np.random.default_rng(1).uniform(-1, 1, size=(7, 2))
    for cls, kwargs in (
        (EnsembleRFBaseline, dict(n_estimators=5)),
        (EnsembleNNBaseline, dict(n_members=4, n_epochs=120)),
        (EnsembleNFBaseline, dict(n_members=4, n_epochs=120)),
    ):
        b = cls(**kwargs).fit(S, T)
        _, std = b.predict_with_std(Xq)
        assert std is not None
        # Average std across all entries must be strictly positive.
        assert float(std.mean()) > 0.0
        assert np.all(std >= 0.0)


# ---------------------------------------------------------------------------
def test_gp_baseline_matches_nonlinear_residual():
    """GPTransportBaseline.transform must reproduce the GPNonlinearResidual
    posterior mean (same object under the hood)."""
    set_global_seed(0)
    S, T = _toy_data(n=20)
    Xq = np.random.default_rng(1).uniform(-1, 1, size=(6, 2))
    base = GPTransportBaseline(n_iter_default=150).fit(S, T)
    nres_mean = base.residual.transform(Xq)
    base_mean = base.transform(Xq)
    np.testing.assert_allclose(base_mean, nres_mean, atol=1e-6)


# ---------------------------------------------------------------------------
def test_ood_behavior_rf_vs_gp():
    """RF std plateaus (or shrinks) far OOD; GP std grows.

    The paper highlights this asymmetry in Table I's discussion: random-
    forest ensemble disagreement does not increase out-of-distribution, so
    the predictive band is overconfident.
    """
    set_global_seed(0)
    S, T = _toy_data(n=20)
    rf = EnsembleRFBaseline(n_estimators=8).fit(S, T)
    gp = GPTransportBaseline(n_iter_default=200).fit(S, T)
    far = np.array([[10.0, -10.0], [-12.0, 9.0]])
    near = S[:3]

    _, std_rf_near = rf.predict_with_std(near)
    _, std_rf_far = rf.predict_with_std(far)
    _, std_gp_near = gp.predict_with_std(near)
    _, std_gp_far = gp.predict_with_std(far)

    # RF does NOT grow far OOD (paper's overconfidence claim).
    assert float(std_rf_far.mean()) <= 2.0 * float(std_rf_near.mean()) + 1e-6
    # GP DOES grow far OOD (Sec. IV-E zero-mean fall-back).
    assert float(std_gp_far.mean()) > 10.0 * float(std_gp_near.mean())


# ---------------------------------------------------------------------------
def test_nf_bijection():
    """Real NVP members must be exactly invertible: inverse(forward(x)) ≈ x."""
    set_global_seed(0)
    S, T = _toy_data(n=15)
    b = EnsembleNFBaseline(n_members=2, n_epochs=80).fit(S, T)
    rng = np.random.default_rng(5)
    x = torch.tensor(rng.uniform(-1.0, 1.0, size=(20, 2)), dtype=torch.float64)
    for flow in b.members:
        with torch.no_grad():
            y = flow(x)
            x_back = flow.inverse(y)
        err = float((x - x_back).abs().max())
        assert err < 1e-3, f"bijection error too large: {err}"


@pytest.fixture(autouse=True)
def _silence_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
