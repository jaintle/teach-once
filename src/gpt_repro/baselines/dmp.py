"""DMP baseline (Sec. V-B): same structure as GPT but linear-only.

The paper says DMP shares "the same mathematical structure as GPT but
only relies on a linear transformation". We implement that literally:

* Fit :class:`LinearTransport` γ on (S, T) frame-cross point pairs.
* Transport the canonical demonstration's positions through γ and
  its velocities through γ's constant Jacobian A.
* Fit a :class:`GPDynamicalSystem` on the transported labels and
  Euler-roll out from any start state.

No GP residual ψ → no non-linear deformation, no transportation
uncertainty.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from gpt_repro.policies.ds_policy import GPDynamicalSystem
from gpt_repro.transport.linear import LinearTransport


class DMPBaseline:
    """Linear-only transportation + GP DS."""

    def __init__(self, n_iter_gp: int = 150, lr: float = 0.1) -> None:
        self.n_iter_gp = int(n_iter_gp)
        self.lr = float(lr)
        self.gamma: Optional[LinearTransport] = None
        self.ds: Optional[GPDynamicalSystem] = None

    def fit(
        self,
        S: np.ndarray,
        T: np.ndarray,
        demo_x: np.ndarray,
        demo_xdot: np.ndarray,
    ) -> "DMPBaseline":
        self.gamma = LinearTransport().fit(S, T)
        x_hat = self.gamma.transform(demo_x)
        xdot_hat = demo_xdot @ self.gamma.A.T
        self.ds = GPDynamicalSystem(n_iter_default=self.n_iter_gp, lr=self.lr)
        self.ds.fit(x_hat, xdot_hat)
        return self

    def rollout(
        self, x0: np.ndarray, dt: float = 0.05, n_steps: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.ds is None:
            raise RuntimeError("DMPBaseline.fit must be called before rollout.")
        return self.ds.rollout(x0, dt=dt, n_steps=n_steps)
