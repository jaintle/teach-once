"""MLP-ensemble (E-NN) baseline.

``n_members`` small MLPs (2 hidden layers × 64 ReLU units), each trained
independently from a different random initialization on the full
residual data ``T - γ(S)``. Member diversity comes from init / SGD
randomness alone (no bootstrap). Mean prediction is the per-member
average; std is the per-member spread.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from gpt_repro.baselines.base import ArrayLike, BaseTransportBaseline


class _MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnsembleNNBaseline(BaseTransportBaseline):
    """Ensemble of small MLPs trained from different random initializations."""

    has_velocity_generalization = True
    has_uncertainty = True
    uncertainty_type = "estimated"

    def __init__(
        self,
        n_members: int = 5,
        hidden: int = 64,
        n_epochs: int = 500,
        lr: float = 1e-2,
        random_state: int = 0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.n_members = int(n_members)
        self.hidden = int(hidden)
        self.n_epochs = int(n_epochs)
        self.lr = float(lr)
        self.random_state = int(random_state)
        self._dtype = dtype
        self._members: List[_MLP] = []
        self._d: int = 0

    def fit(self, S_linear: ArrayLike, T: ArrayLike) -> "EnsembleNNBaseline":
        S = np.asarray(S_linear, dtype=float)
        Tt = np.asarray(T, dtype=float)
        if S.shape != Tt.shape:
            raise ValueError(f"S_linear and T must match shape; got {S.shape} vs {Tt.shape}")
        residual = Tt - S
        self._d = S.shape[1]
        S_t = torch.tensor(S, dtype=self._dtype)
        R_t = torch.tensor(residual, dtype=self._dtype)

        self._members = []
        for m in range(self.n_members):
            torch.manual_seed(self.random_state + m)
            mlp = _MLP(d_in=S.shape[1], d_out=self._d, hidden=self.hidden).to(self._dtype)
            opt = torch.optim.Adam(mlp.parameters(), lr=self.lr)
            for _ in range(self.n_epochs):
                opt.zero_grad()
                pred = mlp(S_t)
                loss = ((pred - R_t) ** 2).mean()
                loss.backward()
                opt.step()
            mlp.eval()
            self._members.append(mlp)
        return self

    def _per_member(self, X: np.ndarray) -> np.ndarray:
        if not self._members:
            raise RuntimeError("EnsembleNNBaseline.fit must be called before transform.")
        X_t = torch.tensor(X, dtype=self._dtype)
        outs = []
        with torch.no_grad():
            for mlp in self._members:
                outs.append(mlp(X_t).cpu().numpy())
        return np.stack(outs, axis=0)  # (n_members, M, d)

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
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
        return self._per_member(X_arr)
