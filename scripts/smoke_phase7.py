"""Phase 7 smoke test — tiny run of benchmark + 3 figures.

Verifies all three figure files and both results CSVs exist, then
prints PASS / FAIL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from figure10_test_boxplots import make_figure as fig10_make  # noqa: E402
from figure8_qualitative import make_figure as fig8_make  # noqa: E402
from figure9_boxplots import make_figure as fig9_make  # noqa: E402
from run_multiframe_benchmark import _save_csv, _save_rank_csv, run_benchmark  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_dir", type=Path,
        default=_REPO_ROOT / "reports" / "figures",
    )
    args = parser.parse_args()
    results_dir = _REPO_ROOT / "reports" / "results"

    bench = run_benchmark(
        seed=args.seed, n_reps=1, m_values=(5,),
        n_steps=30, n_points=30,
        n_iter_transport=60, n_iter_ds=40,
        n_iter_dmp_gp=40, hmm_n_iter=10,
    )
    _save_csv(bench["results"], results_dir / "multiframe_benchmark_results.csv")
    _save_rank_csv(bench["ranking"], results_dir / "multiframe_ranking_table.csv")

    fig8 = fig8_make(
        seed=args.seed, n_steps=30, out_dir=args.out_dir, save=True,
        n_iter_transport=60, n_iter_ds=40, n_iter_dmp_gp=40, hmm_n_iter=10,
    )
    fig9 = fig9_make(seed=args.seed, out_dir=args.out_dir,
                     results_csv=results_dir / "multiframe_benchmark_results.csv")
    fig10 = fig10_make(seed=args.seed, n_reps=2, out_dir=args.out_dir)

    ok = (
        fig8["fig_path"].exists()
        and fig9["fig_path"].exists()
        and fig10["fig_path"].exists()
        and (results_dir / "multiframe_benchmark_results.csv").exists()
        and (results_dir / "multiframe_ranking_table.csv").exists()
    )
    print(f"fig8 : {fig8['fig_path']}  ok={fig8['fig_path'].exists()}")
    print(f"fig9 : {fig9['fig_path']}  ok={fig9['fig_path'].exists()}")
    print(f"fig10: {fig10['fig_path']}  ok={fig10['fig_path'].exists()}")
    print(f"PHASE7 SMOKE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
