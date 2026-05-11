"""Full policy transportation ϕ = γ + ψ ∘ γ, with derived transports.

Combines the linear :math:`\\gamma` (Sec. IV-A) with the GP residual
:math:`\\psi` (Sec. IV-B) into the full transportation map of Sec. IV:

* **Eq. (7)**  :math:`\\phi(x) = \\gamma(x) + \\psi(\\gamma(x))`.
* **Eq. (13)** :math:`\\dot{\\hat x} = J(x)\\,\\dot x` — velocity transport.
* **Eq. (14)** local Taylor expansion of ϕ — used implicitly by
  Eqs. (13) and (15).
* **Eq. (15)** :math:`\\hat R_{ee} = J\\, R_{ee}` with a determinant
  normalization + nearest-proper-rotation projection (paper's
  caveat right after Eq. (15) — J is not in general orthogonal).
* **Stiffness / damping** :math:`\\hat K = J K J^\\top`, :math:`\\hat D = J D J^\\top`
  (Sec. IV-D).

Note the chain rule used for the Jacobian:

    :math:`J(x) = \\partial\\phi/\\partial x
                = (I + \\partial\\psi/\\partial\\gamma|_{\\gamma(x)})\\, A`,

where :math:`A = \\partial\\gamma/\\partial x` is the constant linear
Jacobian (Sec. IV-A).
"""

from __future__ import annotations

from typing import Optional, Type, Union

import numpy as np

from gpt_repro.gp.exact_gp import ExactGPRegressor
from gpt_repro.transport.linear import LinearTransport
from gpt_repro.transport.nonlinear_gp import GPNonlinearResidual

ArrayLike = Union[np.ndarray, list, tuple]


def _nearest_proper_rotation(M: np.ndarray) -> np.ndarray:
    """Project a (d, d) matrix to the closest orthogonal matrix with det = +1.

    Implementation: QR decomposition with sign correction so that the
    triangular factor :math:`R` has a non-negative diagonal, plus a final
    column flip if the resulting :math:`Q` has determinant :math:`-1`.

    This is the "QR + sign correction" projection called out by the prose
    immediately following Eq. (15) in the paper. It is cheaper than the
    polar-decomposition projection and adequate for the cases the paper
    considers because the input matrix :math:`J R_{ee}` is already close
    to a rotation when ϕ is well-fit.

    Parameters
    ----------
    M : (d, d) numpy array.

    Returns
    -------
    Q : (d, d) numpy array — orthogonal with ``det(Q) = +1``.
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError(f"_nearest_proper_rotation expects square; got {M.shape}")
    Q, R = np.linalg.qr(M)
    # Canonicalize signs so R has non-negative diagonal.
    diag_signs = np.sign(np.diag(R))
    diag_signs[diag_signs == 0] = 1.0
    Q = Q * diag_signs  # broadcasts column-wise
    # Final correction: if det(Q) = -1, flip the last column to get +1.
    if np.linalg.det(Q) < 0.0:
        Q = Q.copy()
        Q[:, -1] *= -1.0
    return Q


class PolicyTransport:
    """Full policy transportation map ϕ = γ + ψ ∘ γ — Sec. IV.

    Composes the linear :math:`\\gamma` of :class:`LinearTransport`
    (Sec. IV-A) with the GP residual :math:`\\psi` of
    :class:`GPNonlinearResidual` (Sec. IV-B). Provides:

    * :meth:`transform`           — Eq. (7) point map.
    * :meth:`jacobian`            — analytical Jacobian (autograd through ψ).
    * :meth:`transform_velocity`  — Eq. (13) velocity transport.
    * :meth:`transform_orientation` — Eq. (15) with normalization+QR projection.
    * :meth:`transform_stiffness` — Sec. IV-D, :math:`J K J^\\top`.
    * :meth:`transform_damping`   — Sec. IV-D, :math:`J D J^\\top`.

    Parameters
    ----------
    gp_cls : type
        GP regressor used by the residual model. Defaults to
        :class:`ExactGPRegressor`.
    **gp_kwargs
        Forwarded to :class:`GPNonlinearResidual`, which forces
        ``mean="zero"`` to satisfy Sec. IV-B's fall-back-to-linear
        requirement.
    """

    def __init__(
        self,
        gp_cls: Type = ExactGPRegressor,
        **gp_kwargs,
    ) -> None:
        self.gamma = LinearTransport()
        self.psi = GPNonlinearResidual(gp_cls=gp_cls, **gp_kwargs)
        self._d: int = 0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, S: ArrayLike, T: ArrayLike) -> "PolicyTransport":
        """Fit γ on (S, T), then fit ψ on the residual (γ(S), T).

        Sec. IV procedure: linear alignment first, then GP residual on the
        linearly-aligned points to absorb whatever γ leaves uncorrected.
        """
        self.gamma.fit(S, T)
        S_lin = self.gamma.transform(np.asarray(S, dtype=float))
        self.psi.fit(S_lin, T)
        self._d = self.gamma.d
        return self

    # ------------------------------------------------------------------
    # Point map ϕ — Eq. (7)
    # ------------------------------------------------------------------
    def _check_fit(self) -> None:
        if self.gamma.A is None or not self.psi.is_fit:
            raise RuntimeError("PolicyTransport.fit must be called before transform.")

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Apply **Eq. (7)**: ``ϕ(x) = γ(x) + ψ(γ(x))``.

        Parameters
        ----------
        X : (M, d) or (d,) array in source frame.

        Returns
        -------
        Y : (M, d) array in target frame.
        """
        self._check_fit()
        X_arr = np.asarray(X, dtype=float)
        squeeze_back = False
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
            squeeze_back = True
        x_lin = self.gamma.transform(X_arr)
        psi_mean, _ = self.psi.predict(x_lin, return_std=True)
        out = x_lin + psi_mean
        if squeeze_back:
            return out[0]
        return out

    # ------------------------------------------------------------------
    # Jacobian — Sec. IV-C (Eq. 13), implemented as the chain rule
    # ------------------------------------------------------------------
    def jacobian(self, X: ArrayLike) -> np.ndarray:
        """Analytical Jacobian :math:`J(x) = (I + \\partial\\psi/\\partial\\gamma)\\,A`.

        Cited by Eq. (13) and used by :meth:`transform_velocity`,
        :meth:`transform_orientation`, :meth:`transform_stiffness`,
        :meth:`transform_damping`.

        Parameters
        ----------
        X : (M, d) array of points in source frame.

        Returns
        -------
        J : (M, d, d) — Jacobian of ϕ at each ``X[m]``.
        """
        self._check_fit()
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
        x_lin = self.gamma.transform(X_arr)
        J_psi = self.psi.jacobian(x_lin)  # (M, d, d)
        A = self.gamma.A  # (d, d)
        I = np.eye(self._d)
        # J(x) = (I + J_psi(γ(x))) @ A
        return np.einsum("mij,jk->mik", I + J_psi, A)

    # ------------------------------------------------------------------
    # Eq. (13) velocity transport
    # ------------------------------------------------------------------
    def transform_velocity(
        self, X: ArrayLike, Xdot: ArrayLike
    ) -> np.ndarray:
        """Apply **Eq. (13)**: :math:`\\dot{\\hat x} = J(x)\\,\\dot x`, batched."""
        self._check_fit()
        X_arr = np.asarray(X, dtype=float)
        V_arr = np.asarray(Xdot, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
            V_arr = V_arr[None, :]
            squeeze_back = True
        else:
            squeeze_back = False
        if V_arr.shape != X_arr.shape:
            raise ValueError(
                f"Xdot must match X shape; got {V_arr.shape} vs {X_arr.shape}"
            )
        J = self.jacobian(X_arr)
        out = np.einsum("mij,mj->mi", J, V_arr)
        if squeeze_back:
            return out[0]
        return out

    # ------------------------------------------------------------------
    # Eq. (15) orientation transport with normalization + projection
    # ------------------------------------------------------------------
    def transform_orientation(
        self, X: ArrayLike, R: ArrayLike
    ) -> np.ndarray:
        """Apply **Eq. (15)** with the post-fix from the paper's prose.

        Literally, Eq. (15) reads :math:`\\hat R_{ee} = J\\,R_{ee}`. Because
        :math:`J` is generally not orthogonal, the result is not a
        rotation. We apply the two corrections the paper describes
        immediately after Eq. (15):

        1. Normalize :math:`J` by :math:`\\det(J)^{1/d}` so it has
           unit determinant — this removes the volumetric scaling
           introduced by ϕ.
        2. Multiply by :math:`R_{ee}` and project the result to the
           nearest orthogonal matrix with :math:`\\det = +1` via QR
           (see :func:`_nearest_proper_rotation`).

        Parameters
        ----------
        X : (M, d) array of points in source frame.
        R : (M, d, d) array of source-frame end-effector rotation matrices.

        Returns
        -------
        R_hat : (M, d, d) — rotation matrices in target frame.
        """
        self._check_fit()
        X_arr = np.asarray(X, dtype=float)
        R_arr = np.asarray(R, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
        if R_arr.ndim == 2:
            R_arr = R_arr[None, :, :]
        if R_arr.shape != (X_arr.shape[0], self._d, self._d):
            raise ValueError(
                f"R must have shape ({X_arr.shape[0]}, {self._d}, {self._d}); "
                f"got {R_arr.shape}"
            )
        J = self.jacobian(X_arr)  # (M, d, d)
        # Step 1: normalize J by det(J)^(1/d) to remove the scale factor.
        det = np.linalg.det(J)
        # Preserve sign through the normalization so we don't accidentally
        # produce an improper rotation that the QR step would then have to
        # un-flip.
        scale = np.sign(det) * np.abs(det) ** (1.0 / self._d)
        # Avoid division by zero — clamp scale magnitude.
        eps = 1e-12
        safe_scale = np.where(np.abs(scale) < eps, eps, scale)
        J_norm = J / safe_scale[:, None, None]
        # Step 2: rotate, then project to the nearest proper rotation.
        M_jr = np.einsum("mij,mjk->mik", J_norm, R_arr)
        out = np.empty_like(M_jr)
        for i in range(M_jr.shape[0]):
            out[i] = _nearest_proper_rotation(M_jr[i])
        return out

    # ------------------------------------------------------------------
    # Sec. IV-D stiffness and damping transport
    # ------------------------------------------------------------------
    def _quadratic_form(self, X: ArrayLike, M: ArrayLike) -> np.ndarray:
        """Compute :math:`J(X) M J(X)^\\top` batched over the first axis."""
        self._check_fit()
        X_arr = np.asarray(X, dtype=float)
        M_arr = np.asarray(M, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
        if M_arr.ndim == 2:
            M_arr = M_arr[None, :, :]
        if M_arr.shape != (X_arr.shape[0], self._d, self._d):
            raise ValueError(
                f"matrix arg must have shape ({X_arr.shape[0]}, "
                f"{self._d}, {self._d}); got {M_arr.shape}"
            )
        J = self.jacobian(X_arr)
        # out[m, i, l] = Σ_{j,k} J[m, i, j] M[m, j, k] J[m, l, k]   (=> J M J^T)
        return np.einsum("mij,mjk,mlk->mil", J, M_arr, J)

    def transform_stiffness(self, X: ArrayLike, K: ArrayLike) -> np.ndarray:
        r"""Apply :math:`\hat K_s = J\,K_s\,J^\top` — Sec. IV-D."""
        return self._quadratic_form(X, K)

    def transform_damping(self, X: ArrayLike, D: ArrayLike) -> np.ndarray:
        r"""Apply :math:`\hat D = J\,D\,J^\top` — Sec. IV-D."""
        return self._quadratic_form(X, D)
