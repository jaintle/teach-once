"""Sparse Variational GP (SVGP) regression.

Implements the SVGP approximation of Sec. III-B
(Titsias 2009; Hensman et al. 2013), used in the paper for scalable
GP regression. The variational distribution over inducing values is
trained by maximizing the ELBO via mini-batched gradient descent.

Equations from Sec. III-B implemented in this module:

* **Eq. (2)** – mean prediction at a test point (uses the variational
  posterior over inducing values rather than the full-data conditional).
* **Eq. (3)** – variance prediction at a test point.

The derivative form of Eq. (16) is **not** implemented for SVGP in Phase 1
(see :meth:`SVGPRegressor.predict_with_derivative`). It can be added in a
later phase if the derivative is actually needed for the cleaning-surface
SVGP experiments.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import gpytorch
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_2d_tensor(x: ArrayLike, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    if t.ndim != 2:
        raise ValueError(f"expected 1D or 2D input, got shape {tuple(t.shape)}")
    return t


def _to_1d_tensor(y: ArrayLike, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(y), dtype=dtype, device=device)
    if t.ndim == 2 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    if t.ndim != 1:
        raise ValueError(f"expected 1D target vector, got shape {tuple(t.shape)}")
    return t


class _SVGPModel(gpytorch.models.ApproximateGP):
    """Internal SVGP model with ARD-RBF kernel and a Cholesky variational posterior.

    Implements the variational GP architecture used by the SVGP approximation
    discussed at the end of Sec. III-B.
    """

    def __init__(self, inducing_points: torch.Tensor) -> None:
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        d = inducing_points.shape[-1]
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=d)
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:  # noqa: D401
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class SVGPRegressor:
    """Sparse Variational GP regressor — implements Sec. III-B (SVGP variant).

    Trains a variational sparse GP with :math:`M` inducing points using the
    ELBO objective (:class:`gpytorch.mlls.VariationalELBO`). The public API
    mirrors :class:`ExactGPRegressor` so downstream code can swap them
    interchangeably for the mean / variance prediction surface.

    Parameters
    ----------
    n_inducing : int
        Number of inducing points :math:`M`. Initialized via uniform
        subsampling of the training data in :meth:`fit`.
    n_iter_default : int, optional
        Default number of Adam steps used by :meth:`fit`. Defaults to 200.
    lr : float, optional
        Adam learning rate. Defaults to 0.01.
    batch_size : int, optional
        Mini-batch size for the ELBO optimization. Defaults to 256. If the
        training set is smaller than this, full-batch is used.
    dtype : torch.dtype, optional
        Floating-point precision. Defaults to ``torch.float64``.
    device : str or torch.device, optional
        Defaults to CPU.
    """

    def __init__(
        self,
        n_inducing: int = 64,
        n_iter_default: int = 200,
        lr: float = 0.01,
        batch_size: int = 256,
        dtype: torch.dtype = torch.float64,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.n_inducing = int(n_inducing)
        self.n_iter_default = int(n_iter_default)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self._dtype = dtype
        self._device = torch.device(device) if device is not None else torch.device("cpu")
        self.model: Optional[_SVGPModel] = None
        self.likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None
        self._N_train: int = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, X: ArrayLike, y: ArrayLike, n_iter: Optional[int] = None) -> "SVGPRegressor":
        """Fit the variational posterior by maximizing the ELBO.

        Implements the SVGP training of Sec. III-B (Titsias 2009 /
        Hensman et al. 2013).
        """
        X_t = _to_2d_tensor(X, self._dtype, self._device)
        y_t = _to_1d_tensor(y, self._dtype, self._device)
        if X_t.shape[0] != y_t.shape[0]:
            raise ValueError(
                f"X has {X_t.shape[0]} samples but y has {y_t.shape[0]}"
            )

        N = X_t.shape[0]
        self._N_train = N
        # Initialize inducing inputs by uniformly subsampling the training set.
        M = min(self.n_inducing, N)
        if M < self.n_inducing:
            # Not enough data — fall back to all training points as inducing.
            inducing_points = X_t.clone()
        else:
            # Deterministic subsample (the global seed has been set by the caller).
            idx = torch.randperm(N, device=self._device)[:M]
            inducing_points = X_t[idx].clone()

        model = _SVGPModel(inducing_points).to(device=self._device, dtype=self._dtype)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
            device=self._device, dtype=self._dtype
        )

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters()},
                {"params": likelihood.parameters()},
            ],
            lr=self.lr,
        )
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N)

        batch_size = min(self.batch_size, N)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        steps = int(n_iter) if n_iter is not None else self.n_iter_default
        # We treat "n_iter" as number of epochs through the data (or, when the
        # data fits in one batch, simply optimizer steps).
        for _ in range(steps):
            for xb, yb in loader:
                optimizer.zero_grad()
                output = model(xb)
                loss = -mll(output, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        likelihood.eval()
        self.model = model
        self.likelihood = likelihood
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _ensure_fit(self) -> None:
        if self.model is None or self.likelihood is None:
            raise RuntimeError("SVGPRegressor.fit must be called before predict.")

    def predict(
        self, X_star: ArrayLike, return_std: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Posterior predictive mean (Eq. 2) and optionally std (Eq. 3)."""
        self._ensure_fit()
        X_t = _to_2d_tensor(X_star, self._dtype, self._device)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_pred = self.model(X_t)
            mean = f_pred.mean.detach().cpu().numpy()
            if not return_std:
                return mean
            std = f_pred.variance.detach().clamp_min(0.0).sqrt().cpu().numpy()
        return mean, std

    def predict_with_derivative(self, X_star: ArrayLike):
        """Not implemented in Phase 1.

        The derivative form of Eq. (16) under the SVGP variational posterior
        is deferred to Phase 4 — it is only needed for the cleaning-surface
        SVGP variant in Sec. V-A, which is added later. Calling this in
        Phase 1 is a programming error.
        """
        raise NotImplementedError(
            "SVGPRegressor.predict_with_derivative is not implemented in Phase 1. "
            "It will be added in Phase 4 if required for Sec. V-A's SVGP variant."
        )
