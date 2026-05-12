"""Phase 7 — Fig. 9: 5-metric training-set boxplots with U-test ranks.

Reads `reports/results/multiframe_benchmark_results.csv` produced by
`run_multiframe_benchmark.py` and renders a 1×5 figure (one panel per
metric). Each panel shows box plots for {GPT, DMP, TPGMM_5/6/7,
HMM_5/6/7} with the U-test rank for that metric annotated above each
box.

CLI flags: ``--seed``, ``--out_dir``, ``--results_csv`` (override the
benchmark CSV location).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gpt_repro.metrics import (  # noqa: E402
    METRIC_LABELS, build_ranking_table,
)

METRIC_NAMES = ["frechet", "area", "dtw", "final_pos", "final_orient"]
METHOD_ORDER = [
    "GPT", "DMP",
    "TPGMM_5", "TPGMM_6", "TPGMM_7",
    "HMM_5", "HMM_6", "HMM_7",
]


def _load_csv(path: Path) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            out[row["method"]][row["metric"]].append(float(row["value"]))
    return {m: dict(out[m]) for m in out}


def make_figure(
    seed: int = 0,
    out_dir: Path = _REPO_ROOT / "reports" / "figures",
    results_csv: Path = _REPO_ROOT / "reports" / "results" /
                        "multiframe_benchmark_results.csv",
    save: bool = True,
) -> dict:
    if not results_csv.exists():
        raise FileNotFoundError(
            f"benchmark CSV not found at {results_csv}. "
            "Run scripts/run_multiframe_benchmark.py first."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _load_csv(results_csv)
    # Restrict and order methods.
    methods_present = [m for m in METHOD_ORDER if m in data]
    ranking = build_ranking_table(
        {m: data[m] for m in methods_present}, METRIC_NAMES,
    )

    fig, axes = plt.subplots(1, 5, figsize=(18.0, 4.5), sharey=False)
    for i, metric in enumerate(METRIC_NAMES):
        ax = axes[i]
        values = [data[m][metric] for m in methods_present]
        bp = ax.boxplot(values, tick_labels=methods_present, showfliers=True,
                        patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("0.85")
        # Rank annotations.
        y_top = max(max(v) for v in values) * 1.05 if any(values) else 1.0
        for j, m in enumerate(methods_present, start=1):
            r = ranking["per_metric"][metric]["rank"][m]
            ax.text(j, y_top, str(r), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_title(METRIC_LABELS[metric])
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Phase 7 — Sec. V-B: Training set performance (Fig. 9)",
                 y=0.99)
    fig.text(
        0.5, 0.005,
        f"section=V-B  seed={seed}  "
        f"timestamp={_dt.datetime.now().isoformat(timespec='seconds')}",
        ha="center", fontsize=7, color="grey",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig_path = out_dir / "phase7_fig9_boxplots.png"
    if save:
        fig.savefig(fig_path, dpi=300)
        fig.savefig(fig_path.with_suffix(".pdf"))
    plt.close(fig)
    return {"fig_path": fig_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_dir", type=Path,
        default=_REPO_ROOT / "reports" / "figures",
    )
    parser.add_argument(
        "--results_csv", type=Path,
        default=_REPO_ROOT / "reports" / "results" /
                "multiframe_benchmark_results.csv",
    )
    args = parser.parse_args()
    res = make_figure(seed=args.seed, out_dir=args.out_dir,
                      results_csv=args.results_csv, save=True)
    print(f"Saved figure: {res['fig_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
