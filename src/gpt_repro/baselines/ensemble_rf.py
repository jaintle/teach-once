"""Random-Forest ensemble (E-RF) baseline.

Bootstraps a small ensemble of :class:`sklearn.ensemble.RandomForestRegressor`
models on the residual ``T - γ(S)``. Mean prediction is the average over
trees; std is the spread of per-member predictions. Std plateaus rather
than grows out-of-distribution (the overconfidence noted in the paper's
Table I commentary).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from gpt_repro.baselines.base import ArrayLike, BaseTransportBaseline


class EnsembleRFBaseline(BaseTransportBaseline):
    """Bootstrap ensemble of RandomForestRegressors."""

    has_velocity_generalization = True
    has_uncertainty = True
    uncertainty_type = "estimated"

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 8,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.random_state = int(random_state)
        self._members: List[List[RandomForestRegressor]] = []  # per-output-dim list of forests
        self._d: int = 0

    def fit(self, S_linear: ArrayLike, T: ArrayLike) -> "EnsembleRFBaseline":
        S = np.asarray(S_linear, dtype=float)
        Tt = np.asarray(T, dtype=float)
        if S.shape != Tt.shape:
            raise ValueError(f"S_linear and T must match shape; got {S.shape} vs {Tt.shape}")
        residual = Tt - S
        N = S.shape[0]
        self._d = S.shape[1]
        rng = np.random.default_rng(self.random_state)

        # Ensemble of forests — each on a bootstrap subsample.
        self._members = []
        for k in range(self._d):
            forests = []
            for m in range(self.n_estimators):
                idx = rng.integers(0, N, size=N)
                rf = RandomForestRegressor(
                    n_estimators=20,
                    max_depth=self.max_depth,
                    random_state=self.random_state + m,
                )
                rf.fit(S[idx], residual[idx, k])
                forests.append(rf)
            self._members.append(forests)
        return self

    def _per_member(self, X: np.ndarray) -> np.ndarray:
        """Return (n_members, M, d) per-member predictions."""
        if not self._members:
            raise RuntimeError("EnsembleRFBaseline.fit must be called before transform.")
        M = X.shape[0]
        out = np.empty((self.n_estimators, M, self._d))
        for k in range(self._d):
            for m, rf in enumerate(self._members[k]):
                out[m, :, k] = rf.predict(X)
        return out

    def transform(self, X: ArrayLike) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
        return self._per_member(X_arr).mean(axis=0)

    def predict_with_std(self, X: ArrayLike) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
        per = self._per_member(X_arr)
        return per.mean(axis=0), per.std(axis=0)

    def per_member_predictions(self, X: ArrayLike) -> np.ndarray:
        """Return per-member predictions, shape (n_members, M, d).
        Used by the Fig. 7 plotting code to show ensemble spread."""
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
        return self._per_member(X_arr)
