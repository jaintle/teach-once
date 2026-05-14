"""Dynamical-system policy learning from demonstrations (Sec. III-A).

Implements the autonomous DS

    ẋ = f(x)                                                    Eq. (1)

with ``f`` approximated by Gaussian Process regression on state →
velocity pairs collected from one or more demonstrations. We fit one
independent GP per output dimension (no multi-output kernel — that
would not change the predicted mean at all and only marginally affects
the variance estimate, which we do not need yet).

The GP uses a **zero-mean prior**, as required by Sec. III-A: "it is
safer to have a zero mean prior, such that the robot does not attempt
to do any movement if there is no significant evidence." With a zero
mean, the predicted velocity decays smoothly to zero outside the
training support, which is the key property that makes a naive Euler
rollout from an in-support initial state stay near the demonstration.

This module deliberately does not implement the stiffness / damping
update of ILoSA (Franzese et al. 2021, [25] in the paper); that
machinery is added in Phase 4 if needed.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Type, Union

import numpy as np

from gpt_repro.gp.exact_gp import ExactGPRegressor

ArrayLike = Union[np.ndarray, list, tuple]


class GPDynamicalSystem:
    """GP-parameterized autonomous dynamical system — Sec. III-A, Eq. (1).

    Fits an independent GP regressor per output dimension to learn
    :math:`f : \\mathbb{R}^d \\to \\mathbb{R}^d` such that
    :math:`\\dot x = f(x)`. The per-dimension GPs share the same class
    and hyper-parameters but have independent inferred lengthscales,
    outputscales, and noises.

    Parameters
    ----------
    gp_cls : type
        GP regressor class. Must implement the same fit / predict API as
        :class:`gpt_repro.gp.exact_gp.ExactGPRegressor` and accept a
        ``mean="zero"`` kwarg. Defaults to ``ExactGPRegressor``.
    **gp_kwargs
        Forwarded verbatim to ``gp_cls(**gp_kwargs)`` for each output
        dimension. The ``mean`` kwarg is forced to ``"zero"`` here per
        the Sec. III-A zero-mean-prior requirement; passing any other
        value raises ``ValueError``.
    """

    def __init__(
        self,
        gp_cls: Type = ExactGPRegressor,
        **gp_kwargs,
    ) -> None:
        if gp_kwargs.get("mean", "zero") != "zero":
            raise ValueError(
                "GPDynamicalSystem requires a zero-mean GP prior "
                "(Sec. III-A); got mean=" + repr(gp_kwargs["mean"])
            )
        gp_kwargs["mean"] = "zero"
        self.gp_cls = gp_cls
        self.gp_kwargs = dict(gp_kwargs)
        self._gps: List = []
        self._d: int = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, X: ArrayLike, Xdot: ArrayLike) -> "GPDynamicalSystem":
        """Fit one zero-mean GP per output dimension on (x, ẋ) pairs.

        Approximates ``f`` in **Eq. (1)** :math:`\\dot x = f(x)` by
        regressing ``Xdot`` on ``X`` independently for each output
        dimension. The independent-GP factorization assumes a diagonal
        observation covariance across output dimensions, which is the
        default in the paper.

        Parameters
        ----------
        X : (N, d) array
            Demonstration states.
        Xdot : (N, d) array
            Corresponding demonstration velocities, e.g. from
            finite-differencing.
        """
        X_arr = np.asarray(X, dtype=float)
        Xdot_arr = np.asarray(Xdot, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be (N, d), got {X_arr.shape}")
        if Xdot_arr.shape != X_arr.shape:
            raise ValueError(
                f"Xdot must match X shape, got {Xdot_arr.shape} vs {X_arr.shape}"
            )

        self._d = X_arr.shape[1]
        self._gps = []
        for k in range(self._d):
            gp = self.gp_cls(**self.gp_kwargs)
            gp.fit(X_arr, Xdot_arr[:, k])
            self._gps.append(gp)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    @property
    def is_fit(self) -> bool:
        return len(self._gps) > 0

    @property
    def output_dim(self) -> int:
        return self._d

    def _check_fit(self) -> None:
        if not self.is_fit:
            raise RuntimeError("GPDynamicalSystem.fit must be called before predict.")

    def predict(
        self, X_star: ArrayLike, return_std: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict the velocity field ``f(x)`` and its per-axis std.

        Implements the per-dimension marginal posterior of Sec. III-A:
        each component :math:`f_k(x_*)` follows Eqs. (2) and (3) of the
        underlying GP. The cross-dimensional posterior covariance is not
        modeled (independent GPs).

        Parameters
        ----------
        X_star : (M, d) or (d,) array
            Query state(s).
        return_std : bool
            If True, also return per-axis predictive std.

        Returns
        -------
        mean : (M, d) numpy array of predicted velocities.
        std  : (M, d) numpy array of per-axis predictive std,
               only returned when ``return_std`` is True.
        """
        self._check_fit()
        Xq = np.asarray(X_star, dtype=float)
        squeeze_back = False
        if Xq.ndim == 1:
            Xq = Xq.reshape(1, -1)
            squeeze_back = True
        if Xq.shape[1] != self._d:
            raise ValueError(
                f"X_star has dim {Xq.shape[1]} but model expects {self._d}"
            )

        means = np.empty((Xq.shape[0], self._d))
        stds = np.empty((Xq.shape[0], self._d))
        for k, gp in enumerate(self._gps):
            m, s = gp.predict(Xq, return_std=True)
            means[:, k] = m
            stds[:, k] = s

        if squeeze_back:
            means = means[0]
            stds = stds[0]
        if return_std:
            return means, stds
        return means

    def predict_with_std(self, X_star: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience alias for ``predict(X_star, return_std=True)``.

        Added in Phase 5 so Sec. IV-E's uncertainty-propagation code
        (:func:`gpt_repro.transport.uncertainty.total_velocity_variance`)
        can read the per-axis epistemic std of the refit policy ``f̂``
        without the boolean-flag dance.
        """
        return self.predict(X_star, return_std=True)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def rollout(
        self,
        x0: ArrayLike,
        dt: float = 0.05,
        n_steps: int = 200,
        x_goal: Optional[ArrayLike] = None,
        attractor_gain: float = 1.0,
        stop_eps: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward-Euler integration of the learned DS from ``x0``.

        Implements the standard explicit-Euler discretization of
        Eq. (1):

            :math:`x_{k+1} = x_k + dt \\cdot f(x_k)`

        An optional linear attractor term can be added after the GP
        prediction to prevent the zero-mean prior from stalling the
        rollout far from the goal (ILoSA framework, Sec. III-A [25]):

            :math:`v = f(x) + K(x_{goal} - x)`

        The attractor does NOT pass through the GP; it is added to the
        predicted velocity after prediction.

        Parameters
        ----------
        x0 : (d,) array
            Initial state.
        dt : float
            Time step. Smaller is more accurate at the cost of more GP
            forward evaluations.
        n_steps : int
            Number of Euler steps. The returned trajectory has
            ``n_steps + 1`` states.
        x_goal : (d,) array, optional
            Goal state for the linear attractor term. If ``None``, no
            attractor is applied.
        attractor_gain : float
            Gain K for the linear attractor ``K * (x_goal - x)``.
            Default 1.0. Ignored when ``x_goal`` is ``None``.
        stop_eps : float, optional
            Early-stopping threshold: if ``||f(x_k)|| < stop_eps`` the
            integration halts and the remaining states are padded with
            ``x_k`` (with zero predicted velocity std). Default ``None``
            disables early stopping.

        Returns
        -------
        traj : (n_steps + 1, d) array of states.
        std  : (n_steps + 1, d) array of per-axis predictive std along the trajectory.
        """
        self._check_fit()
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        x = np.asarray(x0, dtype=float).reshape(-1)
        if x.shape != (self._d,):
            raise ValueError(
                f"x0 has shape {x.shape}, expected ({self._d},)"
            )
        _x_goal = (
            np.asarray(x_goal, dtype=float).reshape(-1)
            if x_goal is not None
            else None
        )

        traj = np.empty((n_steps + 1, self._d))
        stds = np.empty((n_steps + 1, self._d))
        traj[0] = x
        # Evaluate at the initial state so the returned std array is
        # well-defined at index 0.
        v0, s0 = self.predict(x[None, :], return_std=True)
        stds[0] = s0[0]
        v_gp = v0[0]
        # Add attractor term (ILoSA, Sec. III-A [25]): v = f(x) + K*(x_goal - x)
        v = v_gp + attractor_gain * (_x_goal - x) if _x_goal is not None else v_gp

        for k in range(n_steps):
            if stop_eps is not None and np.linalg.norm(v) < stop_eps:
                # Pad the rest of the trajectory with the current state.
                traj[k + 1 :] = traj[k]
                stds[k + 1 :] = stds[k]
                break
            traj[k + 1] = traj[k] + dt * v
            v_gp_next, s_next = self.predict(traj[k + 1][None, :], return_std=True)
            stds[k + 1] = s_next[0]
            v_gp = v_gp_next[0]
            # Attractor added AFTER GP prediction (not through GP)
            v = v_gp + attractor_gain * (_x_goal - traj[k + 1]) if _x_goal is not None else v_gp

        return traj, stds
