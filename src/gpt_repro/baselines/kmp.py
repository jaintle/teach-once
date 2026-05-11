"""Kernelized Movement Primitives (KMP) baseline — paper ref. [6].

Implementation: Nadaraya-Watson RBF kernel regression on the residual
``T - γ(S)``. At query x, the predicted residual is the kernel-weighted
average of the training residuals:

    .. math:: \\delta(x) = \\frac{\\sum_i k(x, S_i)\\,(T_i - S_i)}{\\sum_i k(x, S_i)}

This is the "via-point deformation" interpretation: each source / target
pair contributes additively, weighted by RBF proximity in input space.

KMP exposes no uncertainty estimate and no velocity generalization
(differentiating a normalized kernel average is awkward and not what
the paper does).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from gpt_repro.baselines.base import ArrayLike, BaseTransportBaseline


class KMPBaseline(BaseTransportBaseline):
    """RBF Nadaraya-Watson regressor (paper ref. [6])."""

    has_velocity_generalization = False
    has_uncertainty = False
    uncertainty_type = "none"

    def __init__(self, lengthscale: float = 0.3) -> None:
        self.lengthscale = float(lengthscale)
        self._S: Optional[np.ndarray] = None
        self._R: Optional[np.ndarray] = None  # residuals

    def fit(self, S_linear: ArrayLike, T: ArrayLike) -> "KMPBaseline":
        S = np.asarray(S_linear, dtype=float)
        Tt = np.asarray(T, dtype=float)
        if S.shape != Tt.shape:
            raise ValueError(f"S_linear and T must match shape; got {S.shape} vs {Tt.shape}")
        self._S = S
        self._R = Tt - S
        return self

    def _kernel(self, X: np.ndarray) -> np.ndarray:
        # (M, N) RBF kernel between query X and anchor S.
        diff = X[:, None, :] - self._S[None, :, :]
        sq = (diff ** 2).sum(-1)
        return np.exp(-0.5 * sq / (self.lengthscale ** 2))

    def transform(self, X: ArrayLike) -> np.ndarray:
        if self._S is None:
            raise RuntimeError("KMPBaseline.fit must be called before transform.")
        Xq = np.asarray(X, dtype=float)
        if Xq.ndim == 1:
            Xq = Xq[None, :]
        K = self._kernel(Xq)  # (M, N)
        weights = K / np.maximum(K.sum(axis=1, keepdims=True), 1e-12)
        return weights @ self._R  # (M, d)

    def predict_with_std(
        self, X: ArrayLike
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.transform(X), None
