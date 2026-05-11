"""Exact Gaussian Process regression.

Implements Sec. III-B of Franzese et al. (2024).

Equations implemented in this module:

* **Eq. (2)** – posterior predictive mean
  :math:`\\mu(x_*) = k_*^\\top (K + \\sigma_n^2 I)^{-1} y`
* **Eq. (3)** – posterior predictive variance
  :math:`\\sigma^2(x_*) = k(x_*, x_*) - k_*^\\top (K + \\sigma_n^2 I)^{-1} k_*`
* **Eq. (16)** – posterior derivatives w.r.t. the test input
  :math:`\\partial_{x_*} \\mu(x_*) = (\\partial_{x_*} k_*)^\\top (K + \\sigma_n^2 I)^{-1} y`
  :math:`\\partial_{x_*} \\sigma^2(x_*) = -2 \\, k_*^\\top (K + \\sigma_n^2 I)^{-1} \\partial_{x_*} k_*`

Notation matches the paper: :math:`k_* = k(x_*, X)` is the train/test cross
covariance and ``X`` are the training inputs.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import gpytorch
import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_2d_tensor(x: ArrayLike, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Coerce a 1D/2D input to a (N, D) float tensor on the chosen device."""
    t = torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    if t.ndim != 2:
        raise ValueError(f"expected 1D or 2D input, got shape {tuple(t.shape)}")
    return t


def _to_1d_tensor(y: ArrayLike, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Coerce a 1D target vector to a (N,) float tensor."""
    t = torch.as_tensor(np.asarray(y), dtype=dtype, device=device)
    if t.ndim == 2 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    if t.ndim != 1:
        raise ValueError(f"expected 1D target vector, got shape {tuple(t.shape)}")
    return t


class _ExactGPModel(gpytorch.models.ExactGP):
    """Internal gpytorch ExactGP with selectable mean and an ARD-RBF kernel.

    Implements the prior used in Sec. III-B. We support a ConstantMean
    (trainable) — the standard GP default — or a ZeroMean, which Sec. III-A
    explicitly requires for the dynamical-system policy ("it is safer to have
    a zero mean prior, such that the robot does not attempt to do any
    movement if there is no significant evidence"). The covariance is a
    ScaleKernel wrapping an RBFKernel with one length-scale per input
    dimension (ARD), which is the standard squared-exponential kernel of the
    paper.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        mean_type: str = "constant",
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        if mean_type == "zero":
            self.mean_module = gpytorch.means.ZeroMean()
        elif mean_type == "constant":
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            raise ValueError(
                f"mean_type must be 'zero' or 'constant', got {mean_type!r}"
            )
        d = train_x.shape[-1]
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=d)
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:  # noqa: D401
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class ExactGPRegressor:
    """Exact GP regressor — implements Sec. III-B, Eqs. (2), (3), (16).

    Thin wrapper around gpytorch's :class:`~gpytorch.models.ExactGP` with a
    constant mean and ARD-RBF (squared-exponential) covariance. Hyper-parameters
    (output scale, length-scales, noise variance, mean constant) are fit by
    maximizing the exact marginal log likelihood via Adam.

    Parameters
    ----------
    n_iter_default : int, optional
        Default number of Adam steps used by :meth:`fit`. The actual count can
        be overridden per-call. Defaults to 200.
    lr : float, optional
        Adam learning rate. Defaults to 0.1.
    dtype : torch.dtype, optional
        Floating-point precision used internally. Defaults to ``torch.float64``
        for numerical stability of the variance-derivative computation.
    device : str or torch.device, optional
        Defaults to CPU.
    mean : {"constant", "zero"}, optional
        Mean function of the GP prior. ``"constant"`` (default) uses a
        trainable :class:`gpytorch.means.ConstantMean`, which is appropriate
        for generic regression (Sec. III-B). ``"zero"`` uses
        :class:`gpytorch.means.ZeroMean` and is required by the dynamical
        system policy in Sec. III-A.
    interp_mode : bool, optional
        If True, clamp the Gaussian observation noise to a near-zero
        interval so the GP effectively interpolates the training data.
        Used by the policy-transportation residual model in Phase 4,
        where the training residuals are essentially noiseless because
        they come from a deterministic alignment step. Default False.
    """

    def __init__(
        self,
        n_iter_default: int = 200,
        lr: float = 0.1,
        dtype: torch.dtype = torch.float64,
        device: Optional[Union[str, torch.device]] = None,
        mean: str = "constant",
        interp_mode: bool = False,
    ) -> None:
        self.n_iter_default = int(n_iter_default)
        self.lr = float(lr)
        self._dtype = dtype
        self._device = torch.device(device) if device is not None else torch.device("cpu")
        if mean not in {"constant", "zero"}:
            raise ValueError(f"mean must be 'constant' or 'zero', got {mean!r}")
        self._mean_type = mean
        self._interp_mode = bool(interp_mode)
        self.model: Optional[_ExactGPModel] = None
        self.likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None
        self._X_train: Optional[torch.Tensor] = None
        self._y_train: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, X: ArrayLike, y: ArrayLike, n_iter: Optional[int] = None) -> "ExactGPRegressor":
        """Fit GP hyper-parameters by maximizing the exact marginal log-likelihood.

        Implements the training procedure underpinning Eqs. (2)-(3) of
        Sec. III-B.

        Parameters
        ----------
        X : array-like of shape (N, D) or (N,)
        y : array-like of shape (N,)
        n_iter : int, optional
            Number of Adam optimization steps. Defaults to ``self.n_iter_default``.
        """
        X_t = _to_2d_tensor(X, self._dtype, self._device)
        y_t = _to_1d_tensor(y, self._dtype, self._device)
        if X_t.shape[0] != y_t.shape[0]:
            raise ValueError(
                f"X has {X_t.shape[0]} samples but y has {y_t.shape[0]}"
            )

        if self._interp_mode:
            # Constrain the observation noise to a tiny interval so the
            # posterior mean essentially interpolates the training data.
            noise_constraint = gpytorch.constraints.Interval(1e-10, 1e-6)
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=noise_constraint
            ).to(device=self._device, dtype=self._dtype)
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
                device=self._device, dtype=self._dtype
            )
        model = _ExactGPModel(X_t, y_t, likelihood, mean_type=self._mean_type).to(
            device=self._device, dtype=self._dtype
        )

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        steps = int(n_iter) if n_iter is not None else self.n_iter_default
        for _ in range(steps):
            optimizer.zero_grad()
            output = model(X_t)
            loss = -mll(output, y_t)
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()
        self.model = model
        self.likelihood = likelihood
        self._X_train = X_t.detach()
        self._y_train = y_t.detach()
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _ensure_fit(self) -> None:
        if self.model is None or self.likelihood is None:
            raise RuntimeError("ExactGPRegressor.fit must be called before predict.")

    def predict(
        self, X_star: ArrayLike, return_std: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Posterior predictive mean (Eq. 2) and optionally std (Eq. 3).

        Parameters
        ----------
        X_star : array-like of shape (N_*, D) or (N_*,)
        return_std : bool
            If True, also return the posterior std (square-root of Eq. 3).

        Returns
        -------
        mean : numpy.ndarray of shape (N_*,)
        std  : numpy.ndarray of shape (N_*,), only if ``return_std``.
        """
        self._ensure_fit()
        X_t = _to_2d_tensor(X_star, self._dtype, self._device)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # We deliberately do NOT pipe through the likelihood here; we want
            # the latent f distribution (Eq. 2/3 are about f(x_*)), not the
            # observation y(x_*) which adds observation noise σ_n².
            f_pred = self.model(X_t)
            mean = f_pred.mean.detach().cpu().numpy()
            if not return_std:
                return mean
            std = f_pred.variance.detach().clamp_min(0.0).sqrt().cpu().numpy()
        return mean, std

    # ------------------------------------------------------------------
    # Eq. (16) — derivatives w.r.t. the test input
    # ------------------------------------------------------------------
    def predict_with_derivative(
        self, X_star: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Posterior predictive mean/std and their gradients w.r.t. ``x_*``.

        Implements **Eq. (16)** of the paper:

        * The mean derivative
          :math:`\\partial_{x_*} \\mu = (\\partial_{x_*} k_*)^\\top (K+\\sigma_n^2 I)^{-1} y`
          is obtained with :mod:`torch.autograd` differentiating Eq. (2)
          through the gpytorch forward pass.
        * The variance derivative
          :math:`\\partial_{x_*} \\sigma^2 = -2\\, k_*^\\top (K+\\sigma_n^2 I)^{-1} \\partial_{x_*} k_*`
          is computed analytically. Since :math:`k(x_*, x_*)` is constant in
          :math:`x_*` for the RBF kernel, only the data-dependent term
          contributes. The std gradient follows from the chain rule
          :math:`\\partial_{x_*}\\sigma = (2\\sigma)^{-1}\\partial_{x_*}\\sigma^2`.

        Parameters
        ----------
        X_star : array-like of shape (N_*, D) or (N_*,)

        Returns
        -------
        mean     : numpy.ndarray of shape (N_*,)
        std      : numpy.ndarray of shape (N_*,)
        dmean_dx : numpy.ndarray of shape (N_*, D) — gradient of the mean
        dstd_dx  : numpy.ndarray of shape (N_*, D) — gradient of the std
        """
        self._ensure_fit()

        X_t = _to_2d_tensor(X_star, self._dtype, self._device)
        N_star, D = X_t.shape

        self.model.eval()
        self.likelihood.eval()

        # ---- Mean and its gradient via autograd (Eq. 2 + Eq. 16 first line) ----
        x_star = X_t.clone().detach().requires_grad_(True)
        with gpytorch.settings.fast_pred_var():
            f_pred = self.model(x_star)
            mean_t = f_pred.mean
        # Each predicted mean μ(x*_i) only depends on x*_i, so the Jacobian of
        # the (mean.sum) w.r.t. the stacked test inputs equals the row-wise
        # gradients we want, in a single grad call.
        dmean_dx_t = torch.autograd.grad(mean_t.sum(), x_star)[0].detach()

        # ---- Variance & its gradient — analytical (Eq. 3 + Eq. 16 second line) ----
        with torch.no_grad():
            X_tr = self._X_train  # (N, D)
            y_tr = self._y_train  # (N,)

            base_kernel = self.model.covar_module.base_kernel  # RBFKernel
            outputscale = self.model.covar_module.outputscale  # σ_f²
            lengthscale = base_kernel.lengthscale  # (1, D) ARD lengthscales
            noise = self.likelihood.noise  # σ_n² (1,) tensor

            # Train kernel + noise: K + σ_n² I
            K_xx = self.model.covar_module(X_tr).to_dense()
            A = K_xx + torch.eye(
                K_xx.shape[0], dtype=K_xx.dtype, device=K_xx.device
            ) * noise
            # Tiny jitter for Cholesky robustness, doesn't affect math materially
            jitter = 1e-8 * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
            L = torch.linalg.cholesky(A + jitter)

            # k_*: (N_*, N)
            diff = X_t.unsqueeze(1) - X_tr.unsqueeze(0)  # (N_*, N, D)
            sqdist = ((diff / lengthscale) ** 2).sum(-1)  # (N_*, N)
            k_star = outputscale * torch.exp(-0.5 * sqdist)  # (N_*, N)

            # Predictive variance σ²(x_*) = k(x_*, x_*) - k_*^T M k_*
            # k(x_*, x_*) = outputscale for RBF.
            Mk = torch.cholesky_solve(k_star.transpose(0, 1), L)  # (N, N_*)
            quad = (k_star * Mk.transpose(0, 1)).sum(-1)  # (N_*,)
            var = (outputscale - quad).clamp_min(0.0)
            std = var.sqrt()

            # ∂k_*[i, j] / ∂x_*_i_d = -k_*[i, j] · (x_*_i_d - X_j_d) / ℓ_d²
            # Shape: (N_*, N, D)
            inv_ls_sq = 1.0 / (lengthscale ** 2)  # (1, D)
            dk_dx = -k_star.unsqueeze(-1) * diff * inv_ls_sq  # broadcasts

            # ∂σ²/∂x_*_i_d = -2 Σ_j (Mk[j, i]) · dk_dx[i, j, d]
            dvar_dx = -2.0 * torch.einsum("ji,ijd->id", Mk, dk_dx)

            eps = 1e-12
            dstd_dx = dvar_dx / (2.0 * std.unsqueeze(-1).clamp_min(eps))

        return (
            mean_t.detach().cpu().numpy(),
            std.detach().cpu().numpy(),
            dmean_dx_t.cpu().numpy(),
            dstd_dx.detach().cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # Eq. (16) — proper "mean / std of the GP gradient" (the random
    # variable ∂f/∂x, not the gradient of the predictive scalars).
    # Used by Phase 5 for transportation uncertainty (Sec. IV-E).
    # ------------------------------------------------------------------
    def predict_derivative(
        self, X_star: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Mean and per-axis std of the GP gradient :math:`\partial f / \partial x`.

        Implements **Eq. (16)** of Franzese et al. (2024) analytically.
        Given a GP fit with the squared-exponential (ARD-RBF) kernel,
        the gradient ``∂f/∂x_*`` is also Gaussian distributed; using
        the standard derivation (see e.g. Rasmussen & Williams 2006 §9.4
        and paper ref. [26]):

        * mean :math:`\mu'(x_*)_d = \sum_n V_{n,d}\,\alpha_n`,
          where :math:`V_{n,d} = \partial k(x_*, X_n)/\partial x_{*,d}`
          and :math:`\alpha = (K + \sigma_n^2 I)^{-1} y`.
        * variance :math:`\Sigma'(x_*)_{d,d} = K_{11}(x_*, x_*)_{d,d}
          - V_{:,d}^\top (K + \sigma_n^2 I)^{-1} V_{:,d}` where
          :math:`K_{11}(x_*, x_*)_{d,d'} = \sigma_f^2\,\delta_{d,d'}/\ell_d^2`
          is the RBF prior covariance of the gradient at a single point.

        Only the per-axis diagonal of the gradient covariance is returned;
        the cross-axis covariances are not needed by Sec. IV-E.

        Parameters
        ----------
        X_star : array-like of shape (M, D) or (D,) — query points.

        Returns
        -------
        dmu    : numpy.ndarray (M, D) — mean of the GP gradient at each query.
        dsigma : numpy.ndarray (M, D) — per-axis std of the GP gradient.
        """
        self._ensure_fit()
        X_t = _to_2d_tensor(X_star, self._dtype, self._device)
        N_star, D = X_t.shape

        with torch.no_grad():
            X_tr = self._X_train  # (N, D)
            y_tr = self._y_train  # (N,)

            base_kernel = self.model.covar_module.base_kernel
            outputscale = self.model.covar_module.outputscale  # σ_f²
            lengthscale = base_kernel.lengthscale  # (1, D)
            noise = self.likelihood.noise  # σ_n²

            # K + σ_n² I and its Cholesky factor.
            K_xx = self.model.covar_module(X_tr).to_dense()
            A = K_xx + torch.eye(
                K_xx.shape[0], dtype=K_xx.dtype, device=K_xx.device
            ) * noise
            jitter = 1e-8 * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
            L = torch.linalg.cholesky(A + jitter)

            # k_* and ∂k_*/∂x_*  (V tensor)
            diff = X_t.unsqueeze(1) - X_tr.unsqueeze(0)  # (M, N, D)
            sqdist = ((diff / lengthscale) ** 2).sum(-1)  # (M, N)
            k_star = outputscale * torch.exp(-0.5 * sqdist)  # (M, N)
            inv_ls_sq = 1.0 / (lengthscale ** 2)  # (1, D)
            # V[m, n, d] = ∂k(x*_m, X_n)/∂x*_m_d = -k(...) * (x*_m_d - X_n_d)/ℓ_d²
            V = -k_star.unsqueeze(-1) * diff * inv_ls_sq  # (M, N, D)

            # Gradient mean: dmu[m, d] = V[m, :, d] · α  with α = M y.
            alpha = torch.cholesky_solve(y_tr.unsqueeze(-1), L).squeeze(-1)  # (N,)
            dmu = torch.einsum("mnd,n->md", V, alpha)

            # Gradient variance, per axis (diagonal of Σ').
            # K11_dd = σ_f² / ℓ_d²  (off-diagonal not needed).
            prior_var = outputscale * inv_ls_sq.squeeze(0)  # (D,)
            dvar = torch.empty((N_star, D), dtype=X_t.dtype, device=X_t.device)
            for d in range(D):
                Vd = V[:, :, d]  # (M, N)
                MVd = torch.cholesky_solve(Vd.transpose(0, 1), L)  # (N, M)
                quad = (Vd * MVd.transpose(0, 1)).sum(-1)  # (M,)
                dvar[:, d] = (prior_var[d] - quad).clamp_min(0.0)
            dsigma = dvar.sqrt()

        return dmu.detach().cpu().numpy(), dsigma.detach().cpu().numpy()

    def predict_derivative_autograd(self, X_star: ArrayLike) -> np.ndarray:
        """Autograd-based mean of the GP gradient at ``X_star``.

        Equivalent to the ``dmu`` returned by :meth:`predict_derivative`,
        but computed by differentiating the gpytorch predictive mean with
        :mod:`torch.autograd`. Kept for cross-validation in tests — the
        two paths must agree to ~1e-4 on smooth problems.
        """
        self._ensure_fit()
        X_t = _to_2d_tensor(X_star, self._dtype, self._device)
        self.model.eval()
        self.likelihood.eval()
        x_star = X_t.clone().detach().requires_grad_(True)
        with gpytorch.settings.fast_pred_var():
            f_pred = self.model(x_star)
            mean_t = f_pred.mean
        dmu = torch.autograd.grad(mean_t.sum(), x_star)[0]
        return dmu.detach().cpu().numpy()
