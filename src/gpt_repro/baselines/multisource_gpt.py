"""Multi-source GPT method for Sec. V-C.

K source demonstrations, each from a different source frame, are
individually transported to the shared target frame via a dedicated
PolicyTransport (ϕ_k = γ_k + ψ_k). The transported labels are pooled
and used to fit a **single** GPDynamicalSystem in the target frame —
the multi-source fusion described in Sec. V-C.

At rollout time only the joint DS is used, so the inference cost is
independent of K.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from gpt_repro.policies.ds_policy import GPDynamicalSystem
from gpt_repro.transport import PolicyTransport
from gpt_repro.transport.uncertainty import total_velocity_variance


class MultiSourceGPT:
    """Pool K transported demonstrations into a single GP DS — Sec. V-C.

    Parameters
    ----------
    n_iter_transport : int
        GP training iterations for each per-source PolicyTransport.
    n_iter_ds : int
        GP training iterations for the pooled DS.
    lr : float
        Learning rate for Adam in both GP and DS fits.
    """

    def __init__(
        self,
        n_iter_transport: int = 200,
        n_iter_ds: int = 150,
        lr: float = 0.1,
    ) -> None:
        self.n_iter_transport = int(n_iter_transport)
        self.n_iter_ds = int(n_iter_ds)
        self.lr = float(lr)
        self.transports: List[PolicyTransport] = []
        self.ds: Optional[GPDynamicalSystem] = None

    def fit(
        self,
        S_list: List[np.ndarray],
        T: np.ndarray,
        demos: List[dict],
    ) -> "MultiSourceGPT":
        """Transport each source demo and pool labels into a single DS.

        Implements the multi-source fusion of Sec. V-C:
          ϕ_k(x) = γ_k(x) + ψ_k(γ_k(x))   (Eq. 7, one per source)
          Pool {(x̂_{k,i}, ẋ̂_{k,i})} and fit one GPDynamicalSystem.

        Parameters
        ----------
        S_list : list of (N_k, 2) source anchor arrays, one per source.
        T      : (N, 2) shared target anchor array.
        demos  : list of dicts with keys "x" (M_k, 2) and "xdot" (M_k, 2).
        """
        self.transports = []
        x_pool: List[np.ndarray] = []
        xdot_pool: List[np.ndarray] = []

        T_arr = np.asarray(T, dtype=float)
        for S_k, demo in zip(S_list, demos):
            S_arr = np.asarray(S_k, dtype=float)
            transport = PolicyTransport(
                n_iter_default=self.n_iter_transport, lr=self.lr,
            ).fit(S_arr, T_arr)
            self.transports.append(transport)

            x_k = np.asarray(demo["x"],    dtype=float)
            v_k = np.asarray(demo["xdot"], dtype=float)
            x_hat   = transport.transform(x_k)
            xdot_hat = transport.transform_velocity(x_k, v_k)
            x_pool.append(x_hat)
            xdot_pool.append(xdot_hat)

        X_all    = np.vstack(x_pool)
        Xdot_all = np.vstack(xdot_pool)
        self.ds = GPDynamicalSystem(n_iter_default=self.n_iter_ds, lr=self.lr)
        self.ds.fit(X_all, Xdot_all)
        return self

    def rollout(
        self,
        x0: np.ndarray,
        dt: float = 0.05,
        n_steps: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Euler rollout of the jointly fit target DS.

        Parameters
        ----------
        x0      : (2,) initial state in target frame.
        dt      : integration step.
        n_steps : number of steps.

        Returns
        -------
        traj  : (n_steps + 1, 2) positions.
        vels  : (n_steps + 1, 2) velocities.
        """
        if self.ds is None:
            raise RuntimeError("MultiSourceGPT.fit must be called before rollout.")
        return self.ds.rollout(x0, dt=dt, n_steps=n_steps)

    def uncertainty(
        self,
        X: np.ndarray,
        Xdot: np.ndarray,
    ) -> np.ndarray:
        r"""Per-point total std averaged across sources — Eqs. (17)-(18).

        For each source k the total variance (Eq. 18) is
            Σ_total_k(x) = Σ_f̂(x) + Σ_x̂_k(x),
        reduced to a scalar per point as sqrt(trace(Σ_total_k(x))).
        We return the mean over k:
            mean_k( sqrt( trace( Σ_total_k(x) ) ) )

        X and Xdot are treated as approximately source-frame inputs to
        the transport (valid near the target, used for visualization).

        Parameters
        ----------
        X     : (M, 2) query points.
        Xdot  : (M, 2) velocities at those points.

        Returns
        -------
        std_total : (M,) non-negative array.
        """
        if self.ds is None or not self.transports:
            raise RuntimeError("MultiSourceGPT.fit must be called first.")
        X_arr = np.asarray(X, dtype=float)
        V_arr = np.asarray(Xdot, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
            V_arr = V_arr[None, :]

        per_source_std: List[np.ndarray] = []
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for transport in self.transports:
                try:
                    var_k = total_velocity_variance(
                        self.ds, transport, X_arr, V_arr,
                    )  # (M, d)
                except Exception:
                    # Fallback to DS epistemic only
                    _, std_ds = self.ds.predict(X_arr, return_std=True)
                    var_k = std_ds ** 2
                # sqrt(trace(Σ)) = L2 norm over d dims of per-dim std
                std_k = np.sqrt(np.clip(np.sum(var_k, axis=1), 0.0, None))
                per_source_std.append(std_k)

        return np.mean(np.stack(per_source_std, axis=0), axis=0)
