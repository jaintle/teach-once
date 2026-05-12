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
from gpt_repro.policies.multiframe_demos import (
    FrameConfig,
    get_frame_points,
    make_9_frame_configs,
    make_canonical_demo,
    make_multiframe_demo,
)

__all__ = [
    "GPDynamicalSystem",
    "make_letter_C_demo",
    "make_cleaning_demo",
    "make_surface_2d",
    "FrameConfig",
    "make_multiframe_demo",
    "make_9_frame_configs",
    "get_frame_points",
    "make_canonical_demo",
]
