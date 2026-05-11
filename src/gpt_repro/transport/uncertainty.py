"""Transportation + epistemic uncertainty propagation (Sec. IV-E).

Two pure helpers:

* :func:`transportation_velocity_variance` implements **Eq. (17)**:

    .. math::
        \\Sigma_{\\hat x}[m, k]
        = \\sum_j \\dot x_{m,j}^2 \\;\\mathrm{Var}\\bigl( J_{k,j}(X_m) \\bigr)

  where :math:`J = \\partial \\phi / \\partial x = (I + \\partial\\psi/\\partial\\gamma)\\,A`
  and :math:`\\mathrm{Var}(J_{k,j}) = \\sum_l A_{l,j}^2\\,\\mathrm{Var}(\\partial \\psi_k / \\partial \\gamma_l)`
  by the weighted-sum-of-Gaussians rule (paper ref. [30]). The linear
  factor :math:`A` is deterministic, so all of :math:`J`'s variance comes
  from the GP gradient of :math:`\\psi`.

* :func:`total_velocity_variance` implements **Eq. (18)**:

    .. math::
        \\Sigma_{\\dot{\\hat x}}[m, k]
        = \\Sigma_{\\hat f}[m, k] + \\Sigma_{\\hat x}[m, k]

  — the heteroscedastic-GP construction of paper ref. [31]: the
  epistemic variance of the refit policy :math:`\\hat f` in target
  space plus the (transportation) aleatoric variance from Eq. (17).
"""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[np.ndarray, list, tuple]


def transportation_velocity_variance(
    transport,
    X: ArrayLike,
    Xdot: ArrayLike,
) -> np.ndarray:
    r"""Transportation variance of the transported velocity — **Eq. (17)**.

    For each query state ``X[m]`` with source-frame velocity ``Xdot[m]``,
    returns the per-output-dimension variance of the transported velocity
    label :math:`\hat{\dot x} = J(X)\,\dot x`. Only the GP residual
    :math:`\psi` contributes uncertainty (γ has constant Jacobian).

    Parameters
    ----------
    transport : fitted :class:`PolicyTransport` instance.
    X : (M, d) source-frame query points.
    Xdot : (M, d) source-frame velocities at those points.

    Returns
    -------
    Sigma_xhat : (M, d) numpy array — non-negative element-wise.
    """
    if not transport.psi.is_fit:
        raise RuntimeError("PolicyTransport must be fitted before computing variance.")

    X_arr = np.asarray(X, dtype=float)
    V_arr = np.asarray(Xdot, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr[None, :]
        V_arr = V_arr[None, :]
    if V_arr.shape != X_arr.shape:
        raise ValueError(
            f"Xdot must match X shape; got {V_arr.shape} vs {X_arr.shape}"
        )

    d = transport.gamma.d
    x_lin = transport.gamma.transform(X_arr)              # (M, d)
    _, dsigma_psi = transport.psi.predict_derivative(x_lin)  # (M, d, d) std
    var_psi = dsigma_psi ** 2                              # (M, d, d) variance

    # Var(J_{k,j}) = Σ_l A_{l,j}² · Var(∂ψ_k/∂γ_l)        (deterministic A; ref [30])
    A_sq = (transport.gamma.A) ** 2                        # (d, d), A[l, j]²
    var_J = np.einsum("lj,mkl->mkj", A_sq, var_psi)        # (M, d, d)

    # Σ_x̂[m, k] = Σ_j ẋ_{m,j}² · Var(J_{k,j})
    Sigma = np.einsum("mj,mkj->mk", V_arr ** 2, var_J)
    # Clip tiny negative values from floating-point cancellation.
    if np.any(Sigma < -1e-12):
        import warnings
        warnings.warn(
            "transportation_velocity_variance produced large negative values; "
            "clipping at zero. This usually means the K11-K10 M K01 "
            "cancellation lost precision."
        )
    return np.clip(Sigma, 0.0, None)


def total_velocity_variance(
    f_hat_policy,
    transport,
    X: ArrayLike,
    Xdot: ArrayLike,
) -> np.ndarray:
    r"""Total variance of the transported velocity — **Eq. (18)**.

    Sums the epistemic variance of the target-frame policy ``f_hat`` (the
    Sec. III-A GP DS refit on transported labels) with the transportation
    variance returned by :func:`transportation_velocity_variance`. The
    epistemic component is evaluated at the transported query points
    :math:`\hat X = \phi(X)`, since that is where ``f_hat`` lives.

    Parameters
    ----------
    f_hat_policy : :class:`GPDynamicalSystem` already fit on (x̂, ẋ̂).
    transport : fitted :class:`PolicyTransport`.
    X : (M, d) source-frame query points.
    Xdot : (M, d) source-frame velocities.

    Returns
    -------
    Sigma_total : (M, d) numpy array — non-negative element-wise.
    """
    Sigma_xhat = transportation_velocity_variance(transport, X, Xdot)
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr[None, :]
    x_hat = transport.transform(X_arr)                              # target frame
    _, std_fhat = f_hat_policy.predict_with_std(x_hat)
    Sigma_fhat = std_fhat ** 2
    return Sigma_xhat + Sigma_fhat
