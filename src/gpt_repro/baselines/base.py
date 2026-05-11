"""Common interface for all 2D surface-cleaning baselines (Sec. V-A, Table I).

All baselines learn the **residual** map
:math:`\\delta : \\gamma(S) \\to T - \\gamma(S)` from paired
linearly-aligned sources :math:`\\gamma(S)` and targets :math:`T`. The
linear component :math:`\\gamma` (Sec. IV-A) is applied externally, before
any baseline is fit — matching the paper's exact experimental setup.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, list, tuple]


class BaseTransportBaseline(ABC):
    """Abstract base class for the six Sec. V-A transportation baselines.

    Class attributes correspond to the columns of Table I in the paper:

    * ``has_velocity_generalization`` — whether the method can also
      transport velocities (Sec. IV-C, Eq. 13). True for methods whose
      mean prediction is differentiable; false for KMP / LE which are
      pointwise interpolation operators.
    * ``has_uncertainty``             — whether the method emits a
      per-prediction std.
    * ``uncertainty_type``            — ``"none"`` / ``"estimated"`` /
      ``"analytical"``.
    """

    has_velocity_generalization: bool = False
    has_uncertainty: bool = False
    uncertainty_type: str = "none"

    @abstractmethod
    def fit(self, S_linear: ArrayLike, T: ArrayLike) -> "BaseTransportBaseline":
        """Fit the residual map on paired ``(γ(S), T)``."""

    @abstractmethod
    def transform(self, X: ArrayLike) -> np.ndarray:
        """Predict the residual mean at query points ``X``. Shape (M, d)."""

    @abstractmethod
    def predict_with_std(
        self, X: ArrayLike
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return ``(mean, std)``; ``std`` is ``None`` when the method has
        no uncertainty (KMP, LE)."""
