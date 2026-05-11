"""Phase 3 smoke test — Sec. IV-A linear policy transportation.

Runs the same figure-generation logic as :mod:`figure3_linear` with a
small ``n_source_points`` and verifies:

* The figure file is produced.
* The residual is finite.
* The fitted rotation has determinant within [+1 − 1e-6, +1 + 1e-6].

Prints one-line PASS/FAIL.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from figure3_linear import make_figure  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_source_points", type=int, default=12)
    parser.add_argument(
        "--out_dir", type=Path, default=_REPO_ROOT / "reports" / "figures"
    )
    args = parser.parse_args()

    res = make_figure(
        seed=args.seed,
        n_source_points=args.n_source_points,
        out_dir=args.out_dir,
        save=True,
    )

    fig_path = res["fig_path"]
    fig_ok = fig_path is not None and fig_path.exists() and fig_path.stat().st_size > 0
    residual_finite = math.isfinite(res["mean_residual"])
    det_ok = abs(res["det_A"] - 1.0) < 1e-6
    passed = fig_ok and residual_finite and det_ok

    print(f"fig_exists           : {fig_ok}  ({fig_path})")
    print(f"mean residual finite : {residual_finite}  (value={res['mean_residual']:.6f})")
    print(f"det(A) ≈ +1          : {det_ok}  (det={res['det_A']:.9f})")
    print(f"PHASE3 SMOKE: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
