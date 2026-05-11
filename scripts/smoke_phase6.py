"""Phase 6 smoke test — Sec. V-A baseline comparison.

Tiny version of figure7_cleaning_comparison; verifies that:

* the figure file is produced,
* Table I CSV is produced,
* every baseline's transported demo is finite.

Prints one-line PASS/FAIL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from figure7_cleaning_comparison import make_figure  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=Path,
                        default=_REPO_ROOT / "reports" / "figures")
    args = parser.parse_args()

    res = make_figure(
        seed=args.seed, n_demos=20, n_source_points=8,
        out_dir=args.out_dir, save=True,
    )
    fig_path = res["fig_path"]
    csv_path = _REPO_ROOT / "reports" / "results" / "table1.csv"
    fig_ok = fig_path.exists() and fig_path.stat().st_size > 0
    csv_ok = csv_path.exists() and csv_path.stat().st_size > 0

    finite_ok = True
    for name, r in res["method_results"].items():
        if not np.all(np.isfinite(r["traj_mean"])):
            finite_ok = False
            print(f"  non-finite output from {name}")
            break

    print(f"fig path : {fig_path}  ok={fig_ok}")
    print(f"csv path : {csv_path}  ok={csv_ok}")
    print(f"all baseline outputs finite : {finite_ok}")
    passed = fig_ok and csv_ok and finite_ok
    print(f"PHASE6 SMOKE: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
