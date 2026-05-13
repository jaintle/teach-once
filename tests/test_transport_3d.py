"""Tests for 3-D transport math — Phase 9.

Covers:
1. LinearTransport rotation in 3-D (Rodrigues rotation).
2. Velocity transport shapes and finiteness in 3-D.
3. Orientation transport preserves SO(3) structure.
4. Stiffness transport preserves symmetry and PSD.
5. Analytical Jacobian matches central finite differences in 3-D.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpt_repro.transport.linear import LinearTransport, kabsch_svd_rotation
from gpt_repro.transport.policy_transport import PolicyTransport, _nearest_proper_rotation
from gpt_repro.utils.seeding import set_global_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula — exact 3×3 SO(3) matrix."""
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _make_3d_cloud(n: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3))


# ---------------------------------------------------------------------------
# 1. LinearTransport rotation in 3-D
# ---------------------------------------------------------------------------

def test_3d_linear_transport_rotation():
    """LinearTransport.A should recover Rodrigues rotation (atol=1e-6)."""
    set_global_seed(0)
    axis = np.array([0.0, 0.0, 1.0])
    angle = np.pi / 4
    R_true = _rodrigues(axis, angle)  # 3×3 SO(3)

    S = _make_3d_cloud(n=20, seed=1)
    T = (R_true @ S.T).T

    lt = LinearTransport()
    lt.fit(S, T)

    np.testing.assert_allclose(
        lt.A, R_true, atol=1e-6,
        err_msg="LinearTransport.A did not recover Rodrigues rotation",
    )


# ---------------------------------------------------------------------------
# 2. 3-D velocity transport shapes and finiteness
# ---------------------------------------------------------------------------

def test_3d_velocity_transport():
    """transform_velocity in 3-D returns (N,3) finite array."""
    set_global_seed(0)
    from gpt_repro.policies.demos_3d import make_3d_trajectory

    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([1.0, 0.5, 0.3])
    demo = make_3d_trajectory(start, goal, n_points=30, seed=2)
    x_demo = demo["x"]      # (30, 3)
    xdot_demo = demo["xdot"]  # (30, 3)

    S = _make_3d_cloud(n=8, seed=3)
    T = S + np.array([0.5, 0.2, -0.1])

    transport = PolicyTransport(n_iter_default=30)
    transport.fit(S, T)
    xdot_hat = transport.transform_velocity(x_demo, xdot_demo)

    assert xdot_hat.shape == (30, 3), f"Expected (30,3) got {xdot_hat.shape}"
    assert np.all(np.isfinite(xdot_hat)), "Transported velocities contain non-finite values"


# ---------------------------------------------------------------------------
# 3. Orientation transport preserves SO(3)
# ---------------------------------------------------------------------------

def test_3d_orientation_transport_proper_rotation():
    """transform_orientation must produce proper rotation matrices (Rᵀ R ≈ I, det ≈ +1)."""
    set_global_seed(0)
    rng = np.random.default_rng(7)

    S = _make_3d_cloud(n=10, seed=4)
    T = S + np.array([0.2, -0.1, 0.3])

    transport = PolicyTransport(n_iter_default=30)
    transport.fit(S, T)

    # Generate 10 random proper rotation matrices
    for i in range(10):
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0, np.pi)
        R = _rodrigues(axis, angle)
        x = rng.standard_normal(3)

        R_hat = transport.transform_orientation(x[None], R[None])[0]  # (3,3)

        # Check Rᵀ R ≈ I
        np.testing.assert_allclose(
            R_hat.T @ R_hat, np.eye(3), atol=1e-6,
            err_msg=f"Transported R not orthogonal at trial {i}",
        )
        # Check det ≈ +1
        np.testing.assert_allclose(
            np.linalg.det(R_hat), 1.0, atol=1e-6,
            err_msg=f"Transported R has det≠+1 at trial {i}",
        )


# ---------------------------------------------------------------------------
# 4. Stiffness transport preserves symmetry and PSD
# ---------------------------------------------------------------------------

def test_3d_stiffness_symmetry_psd():
    """transform_stiffness on a 3×3 PSD matrix yields a symmetric PSD result."""
    set_global_seed(0)
    rng = np.random.default_rng(11)

    S = _make_3d_cloud(n=10, seed=5)
    T = S + np.array([-0.1, 0.3, 0.2])

    transport = PolicyTransport(n_iter_default=30)
    transport.fit(S, T)

    # Random symmetric PSD 3×3 matrix: K = A^T A + ε I
    A = rng.standard_normal((3, 3))
    K = A.T @ A + 0.1 * np.eye(3)  # PSD

    x = rng.standard_normal(3)
    K_hat = transport.transform_stiffness(x[None], K[None])[0]  # (3, 3)

    # Symmetry
    np.testing.assert_allclose(
        K_hat, K_hat.T, atol=1e-10,
        err_msg="Transported stiffness matrix is not symmetric",
    )

    # PSD: all eigenvalues ≥ 0
    eigs = np.linalg.eigvalsh(K_hat)
    assert np.all(eigs >= -1e-9), f"Transported stiffness has negative eigenvalues: {eigs}"


# ---------------------------------------------------------------------------
# 5. Analytical Jacobian matches central finite differences in 3-D
# ---------------------------------------------------------------------------

def test_3d_jacobian_finite_diff():
    """PolicyTransport Jacobian in 3-D matches central FD (atol=1e-3)."""
    set_global_seed(0)
    rng = np.random.default_rng(13)

    S = _make_3d_cloud(n=10, seed=6)
    # Random rotation + translation
    axis = np.array([1.0, 0.0, 0.0])
    R = _rodrigues(axis, np.pi / 6)
    T = (R @ S.T).T + np.array([0.1, -0.1, 0.05])

    transport = PolicyTransport(n_iter_default=50)
    transport.fit(S, T)

    x0 = rng.standard_normal(3)
    J_ana = transport.jacobian(x0[None])[0]  # (3, 3)

    # Central finite differences
    h = 1e-4
    J_fd = np.zeros((3, 3))
    for d in range(3):
        e = np.zeros(3)
        e[d] = h
        phi_plus = transport.transform((x0 + e)[None])[0]
        phi_minus = transport.transform((x0 - e)[None])[0]
        J_fd[:, d] = (phi_plus - phi_minus) / (2 * h)

    np.testing.assert_allclose(
        J_ana, J_fd, atol=1e-3,
        err_msg="3D Jacobian does not match central finite differences",
    )
