"""GP-residual baseline (paper's proposed method in Sec. V-A).

Thin wrapper around :class:`gpt_repro.transport.nonlinear_gp.GPNonlinearResidual`
so all six baselines share the same ``fit / transform / predict_with_std``
interface. Mean is the analytical GP posterior mean (Eq. 2); std is the
posterior std (Eq. 3) — the unique baseline in Table I whose
uncertainty is *analytical* rather than ensemble-estimated.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from gpt_repro.baselines.base import ArrayLike, BaseTransportBaseline
from gpt_repro.transport.nonlinear_gp import GPNonlinearResidual


class GPTransportBaseline(BaseTransportBaseline):
    """The paper's proposed GP residual model, packaged as a baseline."""

    has_velocity_generalization = True
    has_uncertainty = True
    uncertainty_type = "analytical"

    def __init__(self, n_iter_default: int = 200, lr: float = 0.1) -> None:
        self._residual = GPNonlinearResidual(
            n_iter_default=n_iter_default, lr=lr,
        )

    def fit(self, S_linear: ArrayLike, T: ArrayLike) -> "GPTransportBaseline":
        self._residual.fit(S_linear, T)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        return self._residual.transform(X)

    def predict_with_std(
        self, X: ArrayLike
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        mean, std = self._residual.predict(X, return_std=True)
        return mean, std

    @property
    def residual(self) -> GPNonlinearResidual:
        """Underlying :class:`GPNonlinearResidual` (exposed for the test
        that asserts ``transform`` agreement)."""
        return self._residual
