"""Phase 5 smoke test — uncertainty propagation + Fig. 6 triptych.

Runs a tiny version of ``figure6_uncertainty`` and asserts:

* the figure file is produced (non-empty),
* every entry of every std field is finite and non-negative,
* Σ_total ≥ Σ_x̂ elementwise (i.e. Σ_f̂ ≥ 0 everywhere).

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

from figure6_uncertainty import make_figure  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=Path,
                        default=_REPO_ROOT / "reports" / "figures")
    args = parser.parse_args()

    res = make_figure(
        seed=args.seed, n_grid=8, n_demos=20, n_source_points=10,
        out_dir=args.out_dir, save=True, n_iter_gp=100,
    )
    fig_ok = res["fig_path"].exists() and res["fig_path"].stat().st_size > 0
    finite_ok = bool(
        np.all(np.isfinite(res["std_xhat"]))
        and np.all(np.isfinite(res["std_fhat"]))
        and np.all(np.isfinite(res["std_total"]))
    )
    nonneg_ok = bool(
        np.all(res["std_xhat"] >= 0.0)
        and np.all(res["std_fhat"] >= 0.0)
        and np.all(res["std_total"] >= 0.0)
    )
    # Σ_total ≥ Σ_x̂ elementwise — equivalent to Σ_f̂ ≥ 0.
    var_x   = res["std_xhat"] ** 2
    var_tot = res["std_total"] ** 2
    monotone_ok = bool(np.all(var_tot + 1e-10 >= var_x))

    print(f"fig_exists        : {fig_ok}  ({res['fig_path']})")
    print(f"all finite        : {finite_ok}")
    print(f"all non-negative  : {nonneg_ok}")
    print(f"Σ_total ≥ Σ_x̂   : {monotone_ok}")
    passed = fig_ok and finite_ok and nonneg_ok and monotone_ok
    print(f"PHASE5 SMOKE: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
