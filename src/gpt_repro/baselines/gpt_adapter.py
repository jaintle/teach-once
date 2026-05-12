"""GPT adapter (Sec. V-B): full ϕ = γ + ψ + transported GP DS rollout.

Wraps :class:`PolicyTransport` + :class:`GPDynamicalSystem` so the
multi-frame benchmark can call ``fit / rollout`` uniformly across all
four methods (GPT, DMP, TP-GMM, HMM).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from gpt_repro.policies.ds_policy import GPDynamicalSystem
from gpt_repro.transport import PolicyTransport


class GPTBaseline:
    """Phase-4 PolicyTransport + Phase-2 GP DS, packaged as a benchmark method."""

    def __init__(self, n_iter_transport: int = 200, n_iter_ds: int = 150,
                 lr: float = 0.1) -> None:
        self.n_iter_transport = int(n_iter_transport)
        self.n_iter_ds = int(n_iter_ds)
        self.lr = float(lr)
        self.transport: Optional[PolicyTransport] = None
        self.ds: Optional[GPDynamicalSystem] = None

    def fit(
        self,
        S: np.ndarray,
        T: np.ndarray,
        demo_x: np.ndarray,
        demo_xdot: np.ndarray,
    ) -> "GPTBaseline":
        self.transport = PolicyTransport(
            n_iter_default=self.n_iter_transport, lr=self.lr,
        ).fit(S, T)
        x_hat = self.transport.transform(demo_x)
        xdot_hat = self.transport.transform_velocity(demo_x, demo_xdot)
        self.ds = GPDynamicalSystem(n_iter_default=self.n_iter_ds, lr=self.lr)
        self.ds.fit(x_hat, xdot_hat)
        return self

    def rollout(
        self, x0: np.ndarray, dt: float = 0.05, n_steps: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.ds is None:
            raise RuntimeError("GPTBaseline.fit must be called before rollout.")
        return self.ds.rollout(x0, dt=dt, n_steps=n_steps)
