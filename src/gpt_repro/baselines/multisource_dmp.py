"""Multi-source DMP ablation for Sec. V-C.

Identical structure to MultiSourceGPT but uses LinearTransport (γ only)
for each source. No GP residual ψ, no transportation uncertainty.
The paper calls this the "linear-only" ablation that shows what is lost
when the non-linear correction is omitted.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from gpt_repro.policies.ds_policy import GPDynamicalSystem
from gpt_repro.transport.linear import LinearTransport


class MultiSourceDMP:
    """Pool K linearly-transported demonstrations into one GP DS.

    Parameters
    ----------
    n_iter_gp : int
        GP training iterations for the pooled DS.
    lr : float
        Adam learning rate.
    """

    def __init__(self, n_iter_gp: int = 150, lr: float = 0.1) -> None:
        self.n_iter_gp = int(n_iter_gp)
        self.lr = float(lr)
        self.gammas: List[LinearTransport] = []
        self.ds: Optional[GPDynamicalSystem] = None

    def fit(
        self,
        S_list: List[np.ndarray],
        T: np.ndarray,
        demos: List[dict],
    ) -> "MultiSourceDMP":
        """Linearly transport each source demo and pool labels into one DS.

        Uses γ_k only (no ψ_k), so ϕ_k ≡ γ_k. This is the Sec. V-C
        "linear-only DMP" ablation.

        Parameters
        ----------
        S_list : list of (N_k, 2) source anchor arrays, one per source.
        T      : (N, 2) shared target anchor array.
        demos  : list of dicts with keys "x" (M_k, 2) and "xdot" (M_k, 2).
        """
        self.gammas = []
        T_arr = np.asarray(T, dtype=float)
        x_pool: List[np.ndarray] = []
        xdot_pool: List[np.ndarray] = []

        for S_k, demo in zip(S_list, demos):
            gamma = LinearTransport().fit(np.asarray(S_k, dtype=float), T_arr)
            self.gammas.append(gamma)
            x_k = np.asarray(demo["x"],    dtype=float)
            v_k = np.asarray(demo["xdot"], dtype=float)
            x_pool.append(gamma.transform(x_k))
            xdot_pool.append(v_k @ gamma.A.T)

        X_all    = np.vstack(x_pool)
        Xdot_all = np.vstack(xdot_pool)
        self.ds = GPDynamicalSystem(n_iter_default=self.n_iter_gp, lr=self.lr)
        self.ds.fit(X_all, Xdot_all)
        return self

    def rollout(
        self,
        x0: np.ndarray,
        dt: float = 0.05,
        n_steps: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Euler rollout of the pooled DS.

        Parameters
        ----------
        x0      : (2,) initial state in target frame.
        dt      : integration step.
        n_steps : number of steps.

        Returns
        -------
        traj : (n_steps + 1, 2) positions.
        vels : (n_steps + 1, 2) velocities.
        """
        if self.ds is None:
            raise RuntimeError("MultiSourceDMP.fit must be called before rollout.")
        return self.ds.rollout(x0, dt=dt, n_steps=n_steps)
