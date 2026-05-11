"""Non-linear GP residual ψ for policy transportation (Sec. IV-B).

Given paired sources :math:`S` and targets :math:`T`, the linear
transport :math:`\\gamma` of Sec. IV-A (see :mod:`gpt_repro.transport.linear`)
maps each source to an approximation of its target up to a residual
:math:`r_i = T_i - \\gamma(S_i)`. Sec. IV-B models this residual with a
Gaussian Process:

* **Eq. (12)** – :math:`\\psi(\\gamma(S_i)) \\approx r_i`, fit with one
  zero-mean GP per output dimension.

A zero-mean prior is mandatory: it makes :math:`\\psi(x_*) \\to 0` for
test points :math:`x_*` far from :math:`\\{\\gamma(S_i)\\}`, so the full
transport :math:`\\phi(x) = \\gamma(x) + \\psi(\\gamma(x))` reverts to the
linear :math:`\\gamma` in regions where no source data is available. This
"fall-back-to-linear far away" property is stated immediately after
Eq. (12) in the paper.
"""

from __future__ import annotations

from typing import List, Tuple, Type, Union

import numpy as np

from gpt_repro.gp.exact_gp import ExactGPRegressor

ArrayLike = Union[np.ndarray, list, tuple]


class GPNonlinearResidual:
    """Non-linear GP residual model ψ — Sec. IV-B, Eq. (12).

    Fits one independent zero-mean GP per output dimension to predict the
    residual that the linear :math:`\\gamma` leaves uncorrected. Concretely,
    given linearly-aligned sources ``S_linear[i] = γ(S[i])`` and the
    corresponding targets ``T[i]``, the regressor learns

    .. math:: \\psi : \\mathbb{R}^d \\to \\mathbb{R}^d
              \\quad\\text{such that}\\quad
              \\psi(\\gamma(S_i)) \\approx T_i - \\gamma(S_i).

    Parameters
    ----------
    gp_cls : type
        Per-dim GP regressor class. Must accept ``mean="zero"``.
        Defaults to :class:`ExactGPRegressor`.
    **gp_kwargs
        Forwarded to ``gp_cls(**gp_kwargs)`` for every output dimension.
        The ``mean`` kwarg is forced to ``"zero"`` to satisfy the
        Sec. IV-B fall-back-to-linear requirement; passing any other
        value raises :class:`ValueError`.
    """

    def __init__(self, gp_cls: Type = ExactGPRegressor, **gp_kwargs) -> None:
        if gp_kwargs.get("mean", "zero") != "zero":
            raise ValueError(
                "GPNonlinearResidual requires a zero-mean GP prior so that "
                "ϕ falls back to γ outside the source distribution "
                "(Sec. IV-B); got mean=" + repr(gp_kwargs["mean"])
            )
        gp_kwargs["mean"] = "zero"
        # The training residuals come from a deterministic alignment step,
        # so they are effectively noiseless. Force the GP into
        # near-interpolation mode unless the caller has explicitly set
        # ``interp_mode=False``.
        gp_kwargs.setdefault("interp_mode", True)
        self.gp_cls = gp_cls
        self.gp_kwargs = dict(gp_kwargs)
        self._gps: List = []
        self._d: int = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self, S_linear: ArrayLike, T: ArrayLike
    ) -> "GPNonlinearResidual":
        """Fit ψ on the residual ``T - S_linear`` per output dimension.

        Implements **Eq. (12)** of Sec. IV-B.

        Parameters
        ----------
        S_linear : (N, d) array — linearly-aligned source points γ(S).
        T : (N, d) array — corresponding targets.
        """
        S_arr = np.asarray(S_linear, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        if S_arr.ndim != 2 or T_arr.ndim != 2:
            raise ValueError(
                f"S_linear and T must be 2D (N, d); got {S_arr.shape}, "
                f"{T_arr.shape}"
            )
        if S_arr.shape != T_arr.shape:
            raise ValueError(
                f"S_linear and T must have identical shape, got "
                f"{S_arr.shape} vs {T_arr.shape}"
            )

        residual = T_arr - S_arr
        self._d = S_arr.shape[1]
        self._gps = []
        for k in range(self._d):
            gp = self.gp_cls(**self.gp_kwargs)
            gp.fit(S_arr, residual[:, k])
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
            raise RuntimeError("GPNonlinearResidual.fit must be called before predict.")

    def transform(self, X_star: ArrayLike) -> np.ndarray:
        """Alias for ``predict(X, return_std=False)`` — returns just the
        posterior-mean residual ψ(X) at the query points. Added so the
        Phase-6 baselines (which share a common ``transform`` interface)
        can compare directly against the GP residual model."""
        return self.predict(X_star, return_std=False)

    def predict(
        self, X_star: ArrayLike, return_std: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Posterior mean (and optionally std) of ψ at the query points."""
        self._check_fit()
        X_arr = np.asarray(X_star, dtype=float)
        squeeze_back = False
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
            squeeze_back = True
        if X_arr.shape[1] != self._d:
            raise ValueError(
                f"X_star has dim {X_arr.shape[1]} but ψ expects {self._d}"
            )
        M = X_arr.shape[0]
        means = np.empty((M, self._d))
        stds = np.empty((M, self._d))
        for k, gp in enumerate(self._gps):
            m, s = gp.predict(X_arr, return_std=True)
            means[:, k] = m
            stds[:, k] = s
        if squeeze_back:
            means = means[0]
            stds = stds[0]
        if return_std:
            return means, stds
        return means

    # ------------------------------------------------------------------
    # Jacobian
    # ------------------------------------------------------------------
    def jacobian(self, X_star: ArrayLike) -> np.ndarray:
        """Jacobian of ψ at the query points — used by Eq. (13).

        Uses the autograd-based mean derivative already exposed by
        :meth:`ExactGPRegressor.predict_with_derivative` (which itself
        implements Eq. (16)). Because the residual model fits one
        independent GP per output dimension, the per-dimension gradients
        assemble row-by-row into the full output × input Jacobian.

        Parameters
        ----------
        X_star : (M, d) array of evaluation points.

        Returns
        -------
        J : (M, d, d) numpy array — :math:`J_{m, k, j} = \\partial \\psi_k / \\partial x_j` at ``X_star[m]``.
        """
        self._check_fit()
        X_arr = np.asarray(X_star, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
        if X_arr.ndim != 2 or X_arr.shape[1] != self._d:
            raise ValueError(
                f"X_star must be (M, {self._d}); got shape {X_arr.shape}"
            )
        M = X_arr.shape[0]
        J = np.empty((M, self._d, self._d))
        for k, gp in enumerate(self._gps):
            _, _, dmean, _ = gp.predict_with_derivative(X_arr)
            J[:, k, :] = dmean
        return J

    def predict_derivative(
        self, X_star: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Analytical mean and per-axis std of the Jacobian of ψ — Eq. (16).

        Stacks :meth:`ExactGPRegressor.predict_derivative` across output
        dimensions. For each test point ``X_star[m]``:

        * ``dmu[m, k, j]    = E[∂ψ_k/∂γ_j]`` at ``X_star[m]``.
        * ``dsigma[m, k, j] = std(∂ψ_k/∂γ_j)`` at ``X_star[m]``.

        Used by :func:`gpt_repro.transport.uncertainty.
        transportation_velocity_variance` (Eq. 17). Off-diagonal
        covariances between gradient components are ignored — Sec. IV-E
        only uses the per-element variance.

        Parameters
        ----------
        X_star : (M, d) array (linearly-aligned inputs γ(X)).

        Returns
        -------
        dmu    : (M, d, d) array.
        dsigma : (M, d, d) array — non-negative element-wise.
        """
        self._check_fit()
        X_arr = np.asarray(X_star, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
        if X_arr.ndim != 2 or X_arr.shape[1] != self._d:
            raise ValueError(
                f"X_star must be (M, {self._d}); got shape {X_arr.shape}"
            )
        M = X_arr.shape[0]
        dmu = np.empty((M, self._d, self._d))
        dsigma = np.empty((M, self._d, self._d))
        for k, gp in enumerate(self._gps):
            dm, ds = gp.predict_derivative(X_arr)
            dmu[:, k, :] = dm
            dsigma[:, k, :] = ds
        return dmu, dsigma
