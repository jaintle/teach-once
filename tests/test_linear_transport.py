"""Tests for the linear (rigid) policy transportation γ (Sec. IV-A).

Covers:

* Identity recovery when S == T.
* Pure translation: A ≈ I and γ(X) ≈ X + t.
* Pure 90° rotation: recovered A matches the analytic rotation matrix;
  det(A) = +1.
* Reflection fix triggers correctly when the source / target differ
  only by an improper isometry; final A is a proper rotation.
* Jacobian equals A (constant); batch broadcasting matches.
* 3D rotation recovery (needed in later phases for orientation
  transport, even though Sec. VI is out of scope).
"""

from __future__ import annotations

import numpy as np
import pytest

from gpt_repro.transport import LinearTransport, kabsch_svd_rotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rot_2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _random_proper_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly random rotation in SO(d) via QR."""
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))  # canonicalize signs from QR
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


# ---------------------------------------------------------------------------
# Identity / translation
# ---------------------------------------------------------------------------
def test_identity_when_S_equals_T():
    """If S == T, the fitted A is exactly the identity and γ is the identity."""
    rng = np.random.default_rng(0)
    S = rng.standard_normal((20, 2))
    lt = LinearTransport().fit(S, S)
    np.testing.assert_allclose(lt.A, np.eye(2), atol=1e-6)
    np.testing.assert_allclose(lt.S_bar, lt.T_bar, atol=1e-12)
    np.testing.assert_allclose(lt.transform(S), S, atol=1e-10)
    # Arbitrary test point passes through unchanged.
    X = rng.standard_normal((10, 2))
    np.testing.assert_allclose(lt.transform(X), X, atol=1e-10)
    assert not lt.reflection_fixed


def test_pure_translation():
    """If T = S + t, A ≈ I and γ(X) ≈ X + t."""
    rng = np.random.default_rng(1)
    t = np.array([3.0, -2.0])
    S = rng.standard_normal((25, 2))
    T = S + t
    lt = LinearTransport().fit(S, T)
    np.testing.assert_allclose(lt.A, np.eye(2), atol=1e-6)
    np.testing.assert_allclose(lt.T_bar - lt.S_bar, t, atol=1e-10)
    X = rng.standard_normal((10, 2))
    np.testing.assert_allclose(lt.transform(X), X + t, atol=1e-10)


# ---------------------------------------------------------------------------
# Pure rotation, 2D
# ---------------------------------------------------------------------------
def test_pure_rotation_90deg():
    """If T = R_90 @ S, the recovered A is the same R_90 (det = +1)."""
    rng = np.random.default_rng(2)
    R = _rot_2d(np.pi / 2)
    S = rng.standard_normal((40, 2))
    T = S @ R.T  # row-wise rotation
    lt = LinearTransport().fit(S, T)
    np.testing.assert_allclose(lt.A, R, atol=1e-6)
    assert abs(np.linalg.det(lt.A) - 1.0) < 1e-9
    np.testing.assert_allclose(lt.transform(S), T, atol=1e-9)
    assert not lt.reflection_fixed


# ---------------------------------------------------------------------------
# Reflection fix
# ---------------------------------------------------------------------------
def test_reflection_fix_triggers():
    """If T is a reflection of S, the Kabsch reflection fix must trigger
    and the final A must remain a proper rotation (det = +1)."""
    rng = np.random.default_rng(3)
    S = rng.standard_normal((30, 2))
    # Flip y to introduce an improper isometry.
    T = S * np.array([1.0, -1.0])
    S_c = S - S.mean(axis=0)
    T_c = T - T.mean(axis=0)
    A, fixed = kabsch_svd_rotation(S_c, T_c)
    assert fixed, "reflection fix should have triggered for a reflected target"
    assert abs(np.linalg.det(A) - 1.0) < 1e-9, (
        f"final A is not a proper rotation: det={np.linalg.det(A)}"
    )
    # Class-level: same input through LinearTransport must also flag it.
    lt = LinearTransport().fit(S, T)
    assert lt.reflection_fixed
    assert abs(np.linalg.det(lt.A) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------
def test_jacobian_is_A():
    """jacobian() returns exactly A; the batched form broadcasts."""
    rng = np.random.default_rng(4)
    R = _rot_2d(0.7)
    S = rng.standard_normal((20, 2))
    T = S @ R.T + np.array([1.0, 2.0])
    lt = LinearTransport().fit(S, T)
    J = lt.jacobian()
    np.testing.assert_allclose(J, lt.A, atol=0)
    X = rng.standard_normal((5, 2))
    J_batched = lt.jacobian(X)
    assert J_batched.shape == (5, 2, 2)
    for i in range(5):
        np.testing.assert_allclose(J_batched[i], lt.A, atol=0)


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------
def test_3d_case():
    """LinearTransport works in 3D — needed for Phase 4 orientation transport."""
    rng = np.random.default_rng(5)
    R3 = _random_proper_rotation(3, rng)
    t3 = np.array([0.5, -0.3, 0.2])
    S = rng.standard_normal((50, 3))
    T = S @ R3.T + t3
    lt = LinearTransport().fit(S, T)
    np.testing.assert_allclose(lt.A, R3, atol=1e-6)
    assert abs(np.linalg.det(lt.A) - 1.0) < 1e-9
    np.testing.assert_allclose(lt.T_bar - lt.A @ lt.S_bar, t3, atol=1e-9)
    np.testing.assert_allclose(lt.transform(S), T, atol=1e-9)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_input_validation():
    rng = np.random.default_rng(6)
    S = rng.standard_normal((10, 2))
    T = rng.standard_normal((10, 2))
    # Mismatched shapes.
    with pytest.raises(ValueError):
        LinearTransport().fit(S, T[:5])
    # Too few points.
    with pytest.raises(ValueError):
        LinearTransport().fit(S[:1], T[:1])
    # Unsupported dimensionality.
    with pytest.raises(ValueError):
        LinearTransport().fit(rng.standard_normal((10, 4)),
                              rng.standard_normal((10, 4)))
    # transform before fit.
    with pytest.raises(RuntimeError):
        LinearTransport().transform(S)
