"""Linear policy transportation via SVD (Kabsch / Arun et al. 1987).

Implements the linear component :math:`\\gamma` of the transportation map
:math:`\\phi(x) = \\gamma(x) + \\psi(\\gamma(x))` from Sec. IV of Franzese
et al. (2024). The non-linear residual :math:`\\psi` is added in Phase 4.

Equations implemented here (Sec. IV-A):

* **Eq. (8)** – centered sources and targets are used as paired labels:
  :math:`(S - \\bar S)` and :math:`(T - \\bar T)`.
* **Eq. (9)** – cross-covariance SVD:
  :math:`(S - \\bar S)^\\top (T - \\bar T) = U \\Sigma V^\\top`.
* **Eq. (10)** – rotation
  :math:`A = V U^\\top`, with a reflection fix that flips the last
  column of :math:`V` when :math:`\\det(VU^\\top) < 0`, so that the result
  is a proper rotation (:math:`\\det A = +1`).
* **Eq. (11)** – linear transport:
  :math:`\\gamma(x) = A (x - \\bar S) + \\bar T`.

Reference: K. S. Arun, T. S. Huang, S. D. Blostein, "Least-Squares Fitting
of Two 3-D Point Sets," IEEE T-PAMI 9(5):698–700, 1987 (paper ref. [29]).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, list, tuple]


def kabsch_svd_rotation(
    S_centered: np.ndarray, T_centered: np.ndarray
) -> Tuple[np.ndarray, bool]:
    """Compute the closest proper rotation aligning ``S_centered`` to ``T_centered``.

    Implements Sec. IV-A, Eqs. (9)–(10). The two inputs are assumed
    centered at the origin (i.e. their centroids have been subtracted);
    this lets us focus on the rotation alone, decoupled from the
    translation handled by Eq. (11).

    Algorithm (Kabsch / Arun et al. 1987, ref. [29]):

    1. ``H = S_centered.T @ T_centered``  (cross-covariance, d×d).
    2. ``U, Σ, V^T = svd(H)``  (Eq. 9).
    3. ``A0 = V @ U.T``  (Eq. 10 without reflection fix).
    4. If ``det(A0) < 0``, the data only differ by a reflection;
       flip the sign of the last column of ``V`` so that the returned
       ``A`` is a proper rotation (``det A = +1``). This is the standard
       Kabsch reflection fix and is what the paper means by Eq. (10).

    Parameters
    ----------
    S_centered : (N, d) array — centered source points.
    T_centered : (N, d) array — centered target points.

    Returns
    -------
    A : (d, d) rotation matrix with ``det(A) = +1``.
    reflection_fixed : bool — True iff the reflection fix was applied.
    """
    S_centered = np.asarray(S_centered, dtype=float)
    T_centered = np.asarray(T_centered, dtype=float)
    if S_centered.ndim != 2 or T_centered.ndim != 2:
        raise ValueError(
            f"S_centered and T_centered must be 2D arrays, got "
            f"{S_centered.shape} and {T_centered.shape}"
        )
    if S_centered.shape != T_centered.shape:
        raise ValueError(
            "S_centered and T_centered must have the same shape, got "
            f"{S_centered.shape} vs {T_centered.shape}"
        )

    H = S_centered.T @ T_centered  # Eq. (9), (d, d)
    U, _Sigma, Vt = np.linalg.svd(H)
    V = Vt.T

    A = V @ U.T
    reflection_fixed = False
    if np.linalg.det(A) < 0.0:
        # Reflection fix: flip sign of the column of V corresponding to
        # the smallest singular value (the last one in numpy.svd's
        # descending order). This yields the closest proper rotation
        # rather than the (improper) reflection that minimizes the raw
        # Frobenius error.
        V = V.copy()
        V[:, -1] *= -1.0
        A = V @ U.T
        reflection_fixed = True
    return A, reflection_fixed


class LinearTransport:
    """Linear (rigid) policy transportation — Sec. IV-A, Eqs. (8)–(11).

    Computes the rotation :math:`A` and translation
    :math:`\\bar T - A \\bar S` that best align a source point cloud
    :math:`S` with a target point cloud :math:`T` in the least-squares
    sense, via the Kabsch / Arun et al. (1987) SVD procedure. Once
    fitted, :meth:`transform` applies the map
    :math:`\\gamma(x) = A(x - \\bar S) + \\bar T` (Eq. 11) to arbitrary
    test points; :meth:`jacobian` returns the (constant) Jacobian
    :math:`\\partial\\gamma/\\partial x = A`, which Phase 4 needs for
    velocity transport (Eq. 13) and stiffness transport (Sec. IV-D).
    """

    def __init__(self) -> None:
        self.A: Optional[np.ndarray] = None
        self.S_bar: Optional[np.ndarray] = None
        self.T_bar: Optional[np.ndarray] = None
        self.d: int = 0
        self.reflection_fixed: bool = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, S: ArrayLike, T: ArrayLike) -> "LinearTransport":
        """Fit the linear transport from paired source / target points.

        Implements Sec. IV-A, Eqs. (8)–(10):

        * Eq. (8) — centers the inputs: :math:`\\tilde S = S - \\bar S`,
          :math:`\\tilde T = T - \\bar T`.
        * Eq. (9) — SVD of the cross-covariance
          :math:`\\tilde S^\\top \\tilde T = U\\Sigma V^\\top`.
        * Eq. (10) — rotation
          :math:`A = V U^\\top` with the Kabsch reflection fix.

        Parameters
        ----------
        S : (N, d) array of source points.
        T : (N, d) array of target points, paired with ``S`` row-wise.

        Returns
        -------
        self : LinearTransport.
        """
        S_arr = np.asarray(S, dtype=float)
        T_arr = np.asarray(T, dtype=float)

        if S_arr.ndim != 2 or T_arr.ndim != 2:
            raise ValueError(
                f"S and T must be 2D (N, d) arrays, got shapes "
                f"{S_arr.shape} and {T_arr.shape}"
            )
        if S_arr.shape != T_arr.shape:
            raise ValueError(
                "S and T must have identical shape, got "
                f"{S_arr.shape} vs {T_arr.shape}"
            )
        N, d = S_arr.shape
        if d not in (2, 3):
            raise ValueError(
                f"LinearTransport supports d ∈ {{2, 3}}; got d={d}"
            )
        if N < d:
            raise ValueError(
                f"need at least d={d} paired points to determine the SVD-based "
                f"rotation; got N={N}"
            )

        S_bar = S_arr.mean(axis=0)
        T_bar = T_arr.mean(axis=0)
        S_centered = S_arr - S_bar
        T_centered = T_arr - T_bar
        A, reflection_fixed = kabsch_svd_rotation(S_centered, T_centered)

        self.A = A
        self.S_bar = S_bar
        self.T_bar = T_bar
        self.d = d
        self.reflection_fixed = reflection_fixed
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------
    def _check_fit(self) -> None:
        if self.A is None or self.S_bar is None or self.T_bar is None:
            raise RuntimeError("LinearTransport.fit must be called before transform.")

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Apply Eq. (11): :math:`\\gamma(x) = A(x - \\bar S) + \\bar T`.

        Parameters
        ----------
        X : (M, d) array of source-frame points.

        Returns
        -------
        Y : (M, d) array of points in target frame.
        """
        self._check_fit()
        X_arr = np.asarray(X, dtype=float)
        squeeze_back = False
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
            squeeze_back = True
        if X_arr.ndim != 2 or X_arr.shape[1] != self.d:
            raise ValueError(
                f"X must be (M, {self.d}); got shape {X_arr.shape}"
            )
        Y = (X_arr - self.S_bar) @ self.A.T + self.T_bar
        if squeeze_back:
            return Y[0]
        return Y

    # ------------------------------------------------------------------
    # Jacobian
    # ------------------------------------------------------------------
    def jacobian(self, X: Optional[ArrayLike] = None) -> np.ndarray:
        """Return the Jacobian of γ.

        :math:`\\partial \\gamma / \\partial x = A` is constant for the
        linear map of Eq. (11); used downstream in Phase 4 by the
        velocity-transport rule of Eq. (13) and the stiffness-transport
        rules of Sec. IV-D.

        Parameters
        ----------
        X : optional (M, d) array
            If provided, the (constant) Jacobian is broadcast along the
            batch dimension; the result has shape ``(M, d, d)`` with every
            slice equal to ``A``. This keeps downstream code uniform when
            Phase 4's non-linear residual contributes a state-dependent
            Jacobian.

        Returns
        -------
        J : (d, d) if ``X`` is None, else (M, d, d).
        """
        self._check_fit()
        if X is None:
            return self.A.copy()
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.d:
            raise ValueError(
                f"X must be (M, {self.d}); got shape {X_arr.shape}"
            )
        M = X_arr.shape[0]
        return np.broadcast_to(self.A, (M, self.d, self.d)).copy()
