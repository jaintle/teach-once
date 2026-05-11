"""Laplacian trajectory editing (LE) baseline — paper ref. [13].

A trajectory ``X = (x_0, ..., x_{M-1})`` is represented as a chain
graph whose discrete Laplacian ``L`` encodes local shape. Given a set
of anchor displacements ``(S_i → T_i)``, LE solves

    .. math::
        \\min_{\\delta}\\;\\|L\\,\\delta\\|^2 + w \\sum_i \\|\\delta_{a_i} - (T_i - S_i)\\|^2

where ``a_i`` is the nearest chain node to ``S_i`` and ``w`` is a large
soft-constraint weight. The resulting ``δ`` is the per-node residual
returned by :meth:`transform`. No uncertainty / no velocity generalization.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from gpt_repro.baselines.base import ArrayLike, BaseTransportBaseline


def _chain_laplacian(M: int, periodic: bool = False) -> np.ndarray:
    """Standard 1D discrete Laplacian for a chain (or ring) of M nodes."""
    L = np.zeros((M, M))
    for i in range(M):
        L[i, i] = 2.0
        if i - 1 >= 0:
            L[i, i - 1] = -1.0
        elif periodic:
            L[i, M - 1] = -1.0
        if i + 1 < M:
            L[i, i + 1] = -1.0
        elif periodic:
            L[i, 0] = -1.0
    # Endpoints of an open chain have degree 1, not 2.
    if not periodic:
        L[0, 0] = 1.0
        L[-1, -1] = 1.0
    return L


class LaplacianEditingBaseline(BaseTransportBaseline):
    """Chain-Laplacian trajectory deformation (paper ref. [13])."""

    has_velocity_generalization = False
    has_uncertainty = False
    uncertainty_type = "none"

    def __init__(self, periodic: bool = False, anchor_weight: float = 1e6) -> None:
        self.periodic = bool(periodic)
        self.anchor_weight = float(anchor_weight)
        self._S: Optional[np.ndarray] = None
        self._T: Optional[np.ndarray] = None

    def fit(self, S_linear: ArrayLike, T: ArrayLike) -> "LaplacianEditingBaseline":
        S = np.asarray(S_linear, dtype=float)
        Tt = np.asarray(T, dtype=float)
        if S.shape != Tt.shape:
            raise ValueError(f"S_linear and T must match shape; got {S.shape} vs {Tt.shape}")
        self._S = S
        self._T = Tt
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Build a chain over ``X``, snap each (S, T) anchor to its nearest
        chain node, and solve for the per-node deformation."""
        if self._S is None or self._T is None:
            raise RuntimeError("LaplacianEditingBaseline.fit must be called before transform.")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
        M, d = X_arr.shape
        if M < 2:
            # Too few nodes for a chain — return the trivial nearest-anchor
            # offset.
            diffs = np.linalg.norm(X_arr[:, None, :] - self._S[None, :, :], axis=-1)
            nearest = diffs.argmin(axis=1)
            return self._T[nearest] - self._S[nearest]

        L = _chain_laplacian(M, periodic=self.periodic)
        LtL = L.T @ L
        # For each anchor (S_i, T_i), find the nearest chain node.
        diffs = np.linalg.norm(X_arr[:, None, :] - self._S[None, :, :], axis=-1)
        anchor_node = diffs.argmin(axis=0)        # (n_anchors,)
        anchor_target = self._T - self._S          # (n_anchors, d) residual

        delta = np.zeros_like(X_arr)
        w = self.anchor_weight
        for k in range(d):
            lhs = LtL.copy()
            rhs = np.zeros(M)
            # Soft constraints δ[node] ≈ residual[k].
            for i, node in enumerate(anchor_node):
                lhs[node, node] += w
                rhs[node] += w * anchor_target[i, k]
            delta[:, k] = np.linalg.solve(lhs, rhs)
        return delta

    def predict_with_std(
        self, X: ArrayLike
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.transform(X), None
