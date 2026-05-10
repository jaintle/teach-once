"""Policies sub-package — dynamical-system learning from demonstrations.

Implements Sec. III-A (Eq. 1) of Franzese et al. (2024) along with the
canonical 2D demonstration generators used by the paper's figures.
"""

from gpt_repro.policies.demonstrations import (
    make_cleaning_demo,
    make_letter_C_demo,
    make_surface_2d,
)
from gpt_repro.policies.ds_policy import GPDynamicalSystem

__all__ = [
    "GPDynamicalSystem",
    "make_letter_C_demo",
    "make_cleaning_demo",
    "make_surface_2d",
]
