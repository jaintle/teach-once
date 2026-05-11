"""Tests for the full policy transportation ϕ = γ + ψ ∘ γ (Sec. IV).

Covers:

* Identity recovery when ``T == S``.
* ϕ correctly maps each source point to its paired target after fit.
* Analytical Jacobian (autograd through ψ + chain rule with γ) matches
  a central finite-difference Jacobian of ``transform``.
* Velocity transport: ``ẋ̂ = J(x) ẋ`` matches the closed-form chain
  rule on a handcrafted ϕ (pure rotation in this test).
* Orientation transport produces a proper rotation
  (orthogonal, det = +1).
* Stiffness transport ``K̂ = J K J^T`` preserves symmetry and
  positive-definiteness.
* Out-of-distribution test points fall back to the linear γ
  (zero-mean GP residual property, stated after Eq. 12).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from gpt_repro.transport import PolicyTransport
from gpt_repro.utils import set_global_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rot_2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _finite_diff_jacobian(fn, X: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """Central-difference Jacobian of a vector-valued ``fn`` on a batch."""
    d = X.shape[1]
    J = np.zeros((X.shape[0], d, d))
    for j in range(d):
        e = np.zeros(d)
        e[j] = h
        J[:, :, j] = (fn(X + e) - fn(X - e)) / (2.0 * h)
    return J


# ---------------------------------------------------------------------------
# Identity recovery
# ---------------------------------------------------------------------------
def test_transport_recovers_identity_when_T_equals_S():
    """If T == S, ϕ should be the identity (up to GP fitting tolerance),
    and its Jacobian at every point should be approximately I."""
    set_global_seed(0)
    rng = np.random.default_rng(0)
    S = rng.uniform(-1.0, 1.0, size=(20, 2))
    pt = PolicyTransport(n_iter_default=300, lr=0.1).fit(S, S)
    X = rng.uniform(-0.8, 0.8, size=(8, 2))
    np.testing.assert_allclose(pt.transform(X), X, atol=1e-4)
    J = pt.jacobian(X)
    for m in range(X.shape[0]):
        np.testing.assert_allclose(J[m], np.eye(2), atol=1e-4)


# ---------------------------------------------------------------------------
# ϕ matches T at the source points
# ---------------------------------------------------------------------------
def test_transport_matches_target_at_source_points():
    """ψ should absorb the residual that γ leaves at the source points."""
    set_global_seed(1)
    rng = np.random.default_rng(1)
    S = rng.uniform(-1.0, 1.0, size=(24, 2))
    R = _rot_2d(0.4)
    nonlinearity = 0.05 * np.stack([np.sin(2 * S[:, 1]), np.cos(2 * S[:, 0])], axis=1)
    T = S @ R.T + np.array([0.6, -0.3]) + nonlinearity
    pt = PolicyTransport(n_iter_default=300, lr=0.1).fit(S, T)
    np.testing.assert_allclose(pt.transform(S), T, atol=1e-3)


# ---------------------------------------------------------------------------
# Jacobian vs finite differences (the most important correctness test)
# ---------------------------------------------------------------------------
def test_jacobian_matches_finite_difference():
    """Analytical Jacobian must match a central FD Jacobian of `transform`."""
    set_global_seed(2)
    rng = np.random.default_rng(2)
    S = rng.uniform(-1.0, 1.0, size=(24, 2))
    R = _rot_2d(0.3)
    nonlinearity = 0.05 * np.stack([np.sin(1.8 * S[:, 1]), np.cos(2.1 * S[:, 0])], axis=1)
    T = S @ R.T + np.array([0.2, 0.4]) + nonlinearity
    pt = PolicyTransport(n_iter_default=300, lr=0.1).fit(S, T)

    X = rng.uniform(-0.8, 0.8, size=(10, 2))
    J_ana = pt.jacobian(X)
    J_fd = _finite_diff_jacobian(pt.transform, X, h=1e-4)
    np.testing.assert_allclose(J_ana, J_fd, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Velocity chain rule — pure-rotation closed form
# ---------------------------------------------------------------------------
def test_velocity_transport_chain_rule():
    """With T = R·S (pure rotation), ϕ ≡ γ, so ẋ̂ = R ẋ analytically.
    transport_velocity must match this closed-form chain rule."""
    set_global_seed(3)
    rng = np.random.default_rng(3)
    R = _rot_2d(0.7)
    S = rng.uniform(-1.0, 1.0, size=(30, 2))
    T = S @ R.T + np.array([0.5, -0.2])
    pt = PolicyTransport(n_iter_default=200, lr=0.1).fit(S, T)

    X = rng.uniform(-0.8, 0.8, size=(8, 2))
    Xdot = rng.standard_normal((8, 2))
    Xdot_hat = pt.transform_velocity(X, Xdot)
    expected = Xdot @ R.T
    np.testing.assert_allclose(Xdot_hat, expected, atol=5e-3)

    # And sanity check at the einsum level: transport_velocity must equal
    # the per-point J @ ẋ.
    J = pt.jacobian(X)
    np.testing.assert_allclose(
        Xdot_hat,
        np.einsum("mij,mj->mi", J, Xdot),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Orientation transport: proper rotations
# ---------------------------------------------------------------------------
def test_orientation_transport_is_proper_rotation():
    """For arbitrary input rotations and points, the orientation
    transport must produce orthogonal matrices with det = +1."""
    set_global_seed(4)
    rng = np.random.default_rng(4)
    S = rng.uniform(-1.0, 1.0, size=(24, 2))
    R_align = _rot_2d(0.4)
    T = S @ R_align.T + np.array([0.3, 0.1]) + 0.05 * np.stack(
        [np.sin(2 * S[:, 1]), np.cos(2 * S[:, 0])], axis=1
    )
    pt = PolicyTransport(n_iter_default=200, lr=0.1).fit(S, T)

    X = rng.uniform(-0.8, 0.8, size=(10, 2))
    R_in = np.array([_rot_2d(rng.uniform(-np.pi, np.pi)) for _ in range(10)])
    R_out = pt.transform_orientation(X, R_in)
    assert R_out.shape == (10, 2, 2)
    for m in range(10):
        # Orthogonality.
        np.testing.assert_allclose(
            R_out[m] @ R_out[m].T, np.eye(2), atol=1e-6
        )
        # Proper (det = +1) rotation.
        assert abs(np.linalg.det(R_out[m]) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Stiffness transport: symmetry + PSD
# ---------------------------------------------------------------------------
def test_stiffness_transport_preserves_symmetry_and_psd():
    """K̂ = J K J^T must be symmetric and positive-definite when K is."""
    set_global_seed(5)
    rng = np.random.default_rng(5)
    S = rng.uniform(-1.0, 1.0, size=(20, 2))
    T = S @ _rot_2d(0.5).T + np.array([0.4, 0.1]) + 0.05 * np.stack(
        [np.sin(2 * S[:, 1]), np.cos(2 * S[:, 0])], axis=1
    )
    pt = PolicyTransport(n_iter_default=200, lr=0.1).fit(S, T)

    X = rng.uniform(-0.8, 0.8, size=(6, 2))
    # Build a batch of symmetric PSD K matrices.
    K = np.empty((6, 2, 2))
    for m in range(6):
        M = rng.standard_normal((2, 2))
        K[m] = M @ M.T + 0.5 * np.eye(2)  # PSD + a positive shift
    K_hat = pt.transform_stiffness(X, K)
    assert K_hat.shape == (6, 2, 2)
    for m in range(6):
        # Symmetric.
        np.testing.assert_allclose(K_hat[m], K_hat[m].T, atol=1e-10)
        # Positive-definite.
        eigvals = np.linalg.eigvalsh((K_hat[m] + K_hat[m].T) / 2)
        assert np.all(eigvals > 0.0), f"K̂ not PSD at m={m}: eigvals={eigvals}"

    # Damping uses the same transform; sanity check on a single matrix.
    D = np.broadcast_to(np.diag([5.0, 7.0]), (6, 2, 2))
    D_hat = pt.transform_damping(X, D)
    for m in range(6):
        np.testing.assert_allclose(D_hat[m], D_hat[m].T, atol=1e-10)


# ---------------------------------------------------------------------------
# OOD fall-back to γ
# ---------------------------------------------------------------------------
def test_ood_falls_back_to_linear():
    """Far from the source distribution, the zero-mean GP residual ψ
    must vanish, so ϕ should equal γ at that point."""
    set_global_seed(6)
    rng = np.random.default_rng(6)
    S = rng.uniform(-1.0, 1.0, size=(20, 2))
    T = S @ _rot_2d(0.6).T + np.array([0.3, 0.2]) + 0.05 * np.stack(
        [np.sin(2 * S[:, 1]), np.cos(2 * S[:, 0])], axis=1
    )
    pt = PolicyTransport(n_iter_default=200, lr=0.1).fit(S, T)
    x_far = np.array([[20.0, -20.0], [-15.0, 25.0]])
    phi_far = pt.transform(x_far)
    gamma_far = pt.gamma.transform(x_far)
    # GP lengthscales are O(1), so 20+ units away from S the residual ψ
    # has decayed to numerical zero.
    np.testing.assert_allclose(phi_far, gamma_far, atol=1e-6)


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------
def test_rejects_non_zero_mean_residual():
    """The residual GP must use a zero-mean prior (Sec. IV-B requirement)."""
    with pytest.raises(ValueError):
        PolicyTransport(mean="constant")


@pytest.fixture(autouse=True)
def _silence_warnings():
    """gpytorch emits NumericalWarning when interp-mode variances dip
    slightly below zero at training points; harmless here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
