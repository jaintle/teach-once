"""Normalizing-flow ensemble (E-NF) baseline — paper ref. [32].

Each member is a minimal Real NVP with two affine coupling layers. The
ensemble is trained as a paired regressor :math:`f_\\theta(\\gamma(S))\\approx T`
(MSE loss over the bijective forward map); member diversity from random
initialization gives an empirical std at test time.

The Real NVP is implemented in this file (no external flow library).
Each coupling layer applies, to the dimensions selected by ``mask``,
an affine map conditioned on the complementary dimensions:

    .. math::
        y = x \\odot m + (1 - m) \\odot \\bigl( x \\odot \\exp(s(x \\odot m))
                                              + t(x \\odot m) \\bigr).

``s`` is bounded with ``tanh`` for training stability and the layer is
exactly invertible.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from gpt_repro.baselines.base import ArrayLike, BaseTransportBaseline


class _AffineCoupling(nn.Module):
    def __init__(self, d: int, hidden: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * d),
        )

    def _params(self, x_masked: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x_masked)
        log_s, t = out.chunk(2, dim=-1)
        # Bound log_s with tanh to keep the affine scale near 1 (training
        # stability for small data sets).
        log_s = torch.tanh(log_s) * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        return log_s, t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_m = x * self.mask
        log_s, t = self._params(x_m)
        return x_m + (1.0 - self.mask) * (x * torch.exp(log_s) + t)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y_m = y * self.mask
        log_s, t = self._params(y_m)
        return y_m + (1.0 - self.mask) * ((y - t) * torch.exp(-log_s))


class _RealNVP(nn.Module):
    def __init__(self, d: int = 2, hidden: int = 32, n_couplings: int = 2) -> None:
        super().__init__()
        masks: List[torch.Tensor] = []
        for i in range(n_couplings):
            m = torch.zeros(d)
            m[i % d] = 1.0
            masks.append(m)
        self.couplings = nn.ModuleList(
            [_AffineCoupling(d=d, hidden=hidden, mask=m) for m in masks]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.couplings:
            x = layer(x)
        return x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        for layer in reversed(self.couplings):
            y = layer.inverse(y)
        return y


class EnsembleNFBaseline(BaseTransportBaseline):
    """Ensemble of small Real NVPs trained as paired regressors."""

    has_velocity_generalization = True
    has_uncertainty = True
    uncertainty_type = "estimated"

    def __init__(
        self,
        n_members: int = 5,
        hidden: int = 32,
        n_couplings: int = 2,
        n_epochs: int = 500,
        lr: float = 5e-3,
        random_state: int = 0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.n_members = int(n_members)
        self.hidden = int(hidden)
        self.n_couplings = int(n_couplings)
        self.n_epochs = int(n_epochs)
        self.lr = float(lr)
        self.random_state = int(random_state)
        self._dtype = dtype
        self._members: List[_RealNVP] = []
        self._d: int = 0

    def fit(self, S_linear: ArrayLike, T: ArrayLike) -> "EnsembleNFBaseline":
        S = np.asarray(S_linear, dtype=float)
        Tt = np.asarray(T, dtype=float)
        if S.shape != Tt.shape:
            raise ValueError(f"S_linear and T must match shape; got {S.shape} vs {Tt.shape}")
        self._d = S.shape[1]
        S_t = torch.tensor(S, dtype=self._dtype)
        T_t = torch.tensor(Tt, dtype=self._dtype)

        self._members = []
        for m in range(self.n_members):
            torch.manual_seed(self.random_state + m)
            flow = _RealNVP(
                d=self._d, hidden=self.hidden, n_couplings=self.n_couplings,
            ).to(self._dtype)
            opt = torch.optim.Adam(flow.parameters(), lr=self.lr)
            for _ in range(self.n_epochs):
                opt.zero_grad()
                pred = flow(S_t)
                loss = ((pred - T_t) ** 2).mean()
                loss.backward()
                opt.step()
            flow.eval()
            self._members.append(flow)
        return self

    def _per_member(self, X: np.ndarray) -> np.ndarray:
        if not self._members:
            raise RuntimeError("EnsembleNFBaseline.fit must be called before transform.")
        X_t = torch.tensor(X, dtype=self._dtype)
        outs = []
        with torch.no_grad():
            for flow in self._members:
                # Member predicts full target position; baseline interface
                # returns the residual, so subtract X.
                y = flow(X_t)
                outs.append((y - X_t).cpu().numpy())
        return np.stack(outs, axis=0)

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

    @property
    def members(self) -> List[_RealNVP]:
        """Internal flow members — exposed for the bijection test."""
        return self._members
