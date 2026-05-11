"""Tests for transportation + epistemic uncertainty propagation (Sec. IV-E).

Covers:

* Analytical Eq. (16) gradient std vs a Monte-Carlo estimate from
  posterior samples.
* Analytical Eq. (16) gradient mean vs the autograd path.
* Transportation variance Σ_x̂ is much smaller at source points than
  far from the source distribution.
* Epistemic variance of f̂ grows away from the transported demo.
* Eq. (18) bookkeeping: Σ_total ≡ Σ_x̂ + Σ_f̂.
* Output shapes are consistent across the API.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from gpt_repro.gp import ExactGPRegressor
from gpt_repro.policies import (
    GPDynamicalSystem,
    make_letter_C_demo,
    make_surface_2d,
)
from gpt_repro.transport import (
    PolicyTransport,
    total_velocity_variance,
    transportation_velocity_variance,
)
from gpt_repro.utils import set_global_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fit_toy_gp(seed: int = 0, n: int = 30) -> ExactGPRegressor:
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    X = np.linspace(-3.0, 3.0, n).reshape(-1, 1)
    y = np.sin(X[:, 0]) + 0.05 * rng.standard_normal(n)
    return ExactGPRegressor(n_iter_default=200, lr=0.1).fit(X, y)


def _mc_gradient_std(
    gp: ExactGPRegressor,
    X_star: np.ndarray,
    h: float = 1e-2,
    n_samples: int = 500,
    rng_seed: int = 0,
) -> np.ndarray:
    """Monte-Carlo estimate of std(∂f/∂x) by sampling correlated posterior
    function values at X_star ± h·e_d, then central-differencing each sample.

    Uses an explicit Cholesky decomposition of the joint posterior covariance
    instead of gpytorch's built-in :meth:`sample` — that path can fall back
    to a low-rank approximation under ``fast_pred_var`` in some torch/gpytorch
    builds, which destroys the tight cross-correlations between f(x+h) and
    f(x-h) needed for finite-differencing to recover the true gradient std.

    Note on h: we use h=1e-2 (truncation error O(h²) ≈ 1e-4, well below the
    1e-2 test tolerance). Going smaller is *worse* numerically — the joint
    covariance at adjacent test points agrees in the first ~7 significant
    digits, so cancellation from float64 Cholesky destroys the FD signal.
    """
    M, D = X_star.shape
    # Joint test set ordering: (m=0,d=0,+h), (m=0,d=0,-h), (m=0,d=1,+h), ...
    pts = []
    for m in range(M):
        for d in range(D):
            ep = np.zeros(D); ep[d] = h
            pts.append(X_star[m] + ep)
            pts.append(X_star[m] - ep)
    pts = np.asarray(pts)
    pts_t = torch.tensor(pts, dtype=gp._dtype, device=gp._device)
    gp.model.eval(); gp.likelihood.eval()
    with torch.no_grad():
        f_pred = gp.model(pts_t)
        mean = f_pred.mean
        # Full joint posterior covariance (no fast-path approximations).
        cov = f_pred.covariance_matrix
        # Symmetrize and add jitter for Cholesky robustness.
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        N = cov.shape[0]
        jitter = 1e-8 * torch.eye(N, dtype=cov.dtype, device=cov.device)
        L = torch.linalg.cholesky(cov + jitter)
        # Manual Cholesky sampling with explicit seeding (independent of
        # gpytorch's internal RNG path).
        g = torch.Generator(device="cpu").manual_seed(int(rng_seed))
        eps = torch.randn(n_samples, N, dtype=cov.dtype, generator=g)
        samples = mean.unsqueeze(0) + eps @ L.transpose(-1, -2)  # (n_samples, N)
    samples = samples.detach().cpu().numpy().reshape(n_samples, M, D, 2)
    grads = (samples[..., 0] - samples[..., 1]) / (2.0 * h)  # (n_samples, M, D)
    return grads.std(axis=0)


# ---------------------------------------------------------------------------
# Eq. (16) — analytical vs MC and autograd
# ---------------------------------------------------------------------------
def test_gp_derivative_variance_matches_finite_diff():
    """Analytical std(∂f/∂x) from Eq. (16) matches the MC estimate from
    posterior samples within atol=1e-2."""
    gp = _fit_toy_gp(seed=0, n=30)
    X_star = np.array([[-2.0], [-1.0], [-0.2], [0.4], [1.5], [2.5]])
    _, dsigma_ana = gp.predict_derivative(X_star)
    dsigma_mc = _mc_gradient_std(gp, X_star, h=1e-2, n_samples=600, rng_seed=1)
    np.testing.assert_allclose(dsigma_ana, dsigma_mc, atol=1e-2)


def test_gp_derivative_mean_matches_autograd():
    """Analytical and autograd paths for the GP gradient mean must agree."""
    gp = _fit_toy_gp(seed=0, n=40)
    rng = np.random.default_rng(7)
    X_star = rng.uniform(-2.5, 2.5, size=(15, 1))
    dmu_ana, _ = gp.predict_derivative(X_star)
    dmu_ag = gp.predict_derivative_autograd(X_star)
    np.testing.assert_allclose(dmu_ana, dmu_ag, atol=1e-4)


# ---------------------------------------------------------------------------
# Transportation variance behavior
# ---------------------------------------------------------------------------
def _fit_transport_2d(seed: int = 0, n_s: int = 20):
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    S = rng.uniform(-1.0, 1.0, size=(n_s, 2))
    R = np.array([[np.cos(0.4), -np.sin(0.4)], [np.sin(0.4), np.cos(0.4)]])
    T = S @ R.T + np.array([0.5, -0.2]) + 0.05 * np.stack(
        [np.sin(2 * S[:, 1]), np.cos(2 * S[:, 0])], axis=1
    )
    pt = PolicyTransport(n_iter_default=250, lr=0.1).fit(S, T)
    return S, T, pt


def test_transport_variance_zero_at_source_points():
    """At training source points, transportation variance is much smaller
    than at points far outside the source distribution."""
    S, _, pt = _fit_transport_2d(seed=0, n_s=20)
    Xdot = np.ones_like(S)
    Sigma_at_S = transportation_velocity_variance(pt, S, Xdot)
    far = np.array([[5.0, -5.0], [-6.0, 4.0]])
    far_v = np.ones_like(far)
    Sigma_far = transportation_velocity_variance(pt, far, far_v)

    mean_S = float(Sigma_at_S.mean())
    mean_far = float(Sigma_far.mean())
    assert mean_far > 10.0 * mean_S, (
        f"transportation variance does not grow OOD: "
        f"mean@S={mean_S}, mean@far={mean_far}"
    )


# ---------------------------------------------------------------------------
# Epistemic variance grows OOD for f̂
# ---------------------------------------------------------------------------
def test_epistemic_variance_grows_with_distance_from_demo():
    """f̂ epistemic std grows away from the transported demo trajectory."""
    set_global_seed(0)
    demo = make_letter_C_demo(n_points=40, duration=1.0, radius=0.8)
    S = make_surface_2d("flat", n_points=12, x_range=(-1.4, 1.4))
    T = make_surface_2d("curved", n_points=12, x_range=(-1.4, 1.4),
                        amplitude=0.4)
    pt = PolicyTransport(n_iter_default=200, lr=0.1).fit(S, T)
    x_hat = pt.transform(demo["x"])
    xdot_hat = pt.transform_velocity(demo["x"], demo["xdot"])
    f_hat = GPDynamicalSystem(n_iter_default=150).fit(x_hat, xdot_hat)

    _, std_on_demo = f_hat.predict_with_std(x_hat)
    on_demo = float(np.linalg.norm(std_on_demo, axis=1).mean())

    off_demo_pts = np.array([[3.0, 3.0], [-3.0, -3.0], [4.0, 0.0]])
    _, std_off = f_hat.predict_with_std(off_demo_pts)
    off_demo = float(np.linalg.norm(std_off, axis=1).min())

    assert off_demo > on_demo, (
        f"epistemic std not growing OOD: on_demo={on_demo}, off={off_demo}"
    )


# ---------------------------------------------------------------------------
# Eq. (18) bookkeeping
# ---------------------------------------------------------------------------
def test_total_variance_is_sum():
    """Σ_total == Σ_x̂ + Σ_f̂ to floating-point precision (Eq. 18)."""
    set_global_seed(0)
    demo = make_letter_C_demo(n_points=30, duration=1.0, radius=0.7)
    S = make_surface_2d("flat", n_points=12, x_range=(-1.2, 1.2))
    T = make_surface_2d("curved", n_points=12, x_range=(-1.2, 1.2),
                        amplitude=0.3)
    pt = PolicyTransport(n_iter_default=180, lr=0.1).fit(S, T)
    x_hat = pt.transform(demo["x"])
    xdot_hat = pt.transform_velocity(demo["x"], demo["xdot"])
    f_hat = GPDynamicalSystem(n_iter_default=150).fit(x_hat, xdot_hat)

    rng = np.random.default_rng(0)
    X = rng.uniform(-0.6, 0.6, size=(8, 2))
    Xdot = rng.standard_normal((8, 2))
    Sigma_x = transportation_velocity_variance(pt, X, Xdot)
    Sigma_total = total_velocity_variance(f_hat, pt, X, Xdot)
    # Σ_total - Σ_x̂ should equal Σ_f̂ = std(f̂(ϕ(X)))²; verify recomputed Σ_f̂.
    x_hat_q = pt.transform(X)
    _, std_fh = f_hat.predict_with_std(x_hat_q)
    Sigma_fhat_recomputed = std_fh ** 2
    np.testing.assert_allclose(
        Sigma_total - Sigma_x, Sigma_fhat_recomputed, atol=1e-10
    )


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
def test_uncertainty_shapes_are_consistent():
    """For (M, 2) inputs, all three variance fields are (M, 2); L2-norm
    reduction across output dims yields (M,)."""
    set_global_seed(0)
    S, _, pt = _fit_transport_2d(seed=0, n_s=15)
    demo = make_letter_C_demo(n_points=12, duration=1.0, radius=0.5)
    x_hat = pt.transform(demo["x"])
    xdot_hat = pt.transform_velocity(demo["x"], demo["xdot"])
    f_hat = GPDynamicalSystem(n_iter_default=100).fit(x_hat, xdot_hat)

    X = np.random.default_rng(0).uniform(-0.5, 0.5, size=(7, 2))
    Xdot = np.random.default_rng(1).standard_normal((7, 2))
    Sxh = transportation_velocity_variance(pt, X, Xdot)
    Stot = total_velocity_variance(f_hat, pt, X, Xdot)
    assert Sxh.shape == (7, 2)
    assert Stot.shape == (7, 2)
    scalar_xhat = np.linalg.norm(np.sqrt(Sxh), axis=1)
    scalar_total = np.linalg.norm(np.sqrt(Stot), axis=1)
    assert scalar_xhat.shape == (7,)
    assert scalar_total.shape == (7,)
    assert np.all(Sxh >= 0.0)
    assert np.all(Stot >= 0.0)


@pytest.fixture(autouse=True)
def _silence_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
