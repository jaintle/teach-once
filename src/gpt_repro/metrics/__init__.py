"""Numerical metrics used by the paper's tables and figures."""

from gpt_repro.metrics.table1 import build_table1, format_ascii, save_csv
from gpt_repro.metrics.trajectory_metrics import (
    METRIC_FNS,
    METRIC_LABELS,
    area_between_curves,
    dtw_distance,
    final_orientation_error,
    final_position_error,
    frechet_distance,
)
from gpt_repro.metrics.utest import (
    build_ranking_table,
    format_ranking_ascii,
    mann_whitney_ranking,
)

__all__ = [
    "build_table1",
    "format_ascii",
    "save_csv",
    "METRIC_FNS",
    "METRIC_LABELS",
    "frechet_distance",
    "area_between_curves",
    "dtw_distance",
    "final_position_error",
    "final_orientation_error",
    "mann_whitney_ranking",
    "build_ranking_table",
    "format_ranking_ascii",
]
