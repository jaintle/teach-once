"""Mann-Whitney U-test ranking (Sec. V-B, Fig. 9 / Fig. 10).

For each pair of methods (A, B) we run a one-sided Mann-Whitney U test
under the alternative that **A is significantly less than B**. If
``p < alpha``, A scores one point. Methods are ranked by total points
won (more points → better rank, rank 1 = best).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats import mannwhitneyu


def mann_whitney_ranking(
    results: Dict[str, Iterable[float]],
    alpha: float = 0.05,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Run pairwise one-sided U-tests and return (points, rank) per method.

    Parameters
    ----------
    results : mapping method name → list of *lower-is-better* scalar metric values.
    alpha : significance threshold for a "win".

    Returns
    -------
    points : dict[str, int] — total wins per method.
    rank   : dict[str, int] — 1-indexed rank (1 = best). Ties share rank.
    """
    methods = list(results)
    arrs = {m: np.asarray(list(results[m]), dtype=float) for m in methods}
    points: Dict[str, int] = {m: 0 for m in methods}
    for a in methods:
        for b in methods:
            if a == b:
                continue
            if len(arrs[a]) == 0 or len(arrs[b]) == 0:
                continue
            # H1: a is stochastically less than b → a wins.
            try:
                _, p = mannwhitneyu(arrs[a], arrs[b], alternative="less")
            except ValueError:
                # All values equal → no significant difference.
                continue
            if p < alpha:
                points[a] += 1

    # Convert points to ranks (descending by points; ties share a rank).
    sorted_methods = sorted(methods, key=lambda m: (-points[m], m))
    rank: Dict[str, int] = {}
    current_rank = 0
    prev_points: int = None  # type: ignore[assignment]
    for i, m in enumerate(sorted_methods):
        if points[m] != prev_points:
            current_rank = i + 1
            prev_points = points[m]
        rank[m] = current_rank
    return points, rank


def build_ranking_table(
    all_results: Dict[str, Dict[str, List[float]]],
    metric_names: List[str],
    alpha: float = 0.05,
) -> Dict[str, Dict]:
    """Run :func:`mann_whitney_ranking` per metric.

    ``all_results`` is keyed by method name; each value maps metric name
    → list of metric values across reps. Returns a dict::

        {
          "per_metric": {metric: {"points": {...}, "rank": {...}}},
          "method_order": [...],
        }
    """
    methods = list(all_results)
    per_metric: Dict[str, Dict] = {}
    for metric in metric_names:
        results_by_method = {m: all_results[m][metric] for m in methods}
        pts, rnk = mann_whitney_ranking(results_by_method, alpha=alpha)
        per_metric[metric] = {"points": pts, "rank": rnk}
    return {"per_metric": per_metric, "method_order": methods}


def format_ranking_ascii(
    ranking: Dict[str, Dict],
    metric_names: List[str],
    metric_labels: Dict[str, str] = None,
) -> str:
    """ASCII rank table — methods as rows, metrics as columns."""
    methods = ranking["method_order"]
    labels = {m: m for m in metric_names}
    if metric_labels:
        labels.update(metric_labels)
    headers = ["Method"] + [labels[m] for m in metric_names]
    widths = [max(len(h), max((len(m) for m in methods), default=0)) for h in headers]
    widths[0] = max(widths[0], max((len(m) for m in methods), default=0))
    rows: List[List[str]] = []
    for method in methods:
        row = [method]
        for metric in metric_names:
            row.append(str(ranking["per_metric"][metric]["rank"][method]))
        rows.append(row)
    # Re-compute widths from final cells.
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    line_sep = "  ".join("-" * w for w in widths)
    header_line = "  ".join(f"{headers[i]:<{widths[i]}}" for i in range(len(headers)))
    body_lines = [
        "  ".join(f"{r[i]:<{widths[i]}}" for i in range(len(headers))) for r in rows
    ]
    return "\n".join([header_line, line_sep] + body_lines)
