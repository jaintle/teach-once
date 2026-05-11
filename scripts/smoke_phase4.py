"""Phase 4 smoke test — non-linear policy transportation + figures.

Runs tiny versions of ``figure3_full`` and ``figure5_scheme`` and
verifies:

* both figure files were produced (non-empty),
* the residual ``max ‖ϕ(S) − T‖`` at the training source points is
  below ``1e-2``.

Prints one-line PASS/FAIL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from figure3_full import make_figure as make_figure3_full  # noqa: E402
from figure5_scheme import make_figure as make_figure5_scheme  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=Path,
                        default=_REPO_ROOT / "reports" / "figures")
    args = parser.parse_args()

    res3 = make_figure3_full(
        seed=args.seed, n_source_points=12, out_dir=args.out_dir, save=True,
    )
    res5 = make_figure5_scheme(
        seed=args.seed, n_demos=30, n_source_points=12,
        out_dir=args.out_dir, save=True, n_iter_gp=120,
    )

    fig3_ok = res3["fig_path"].exists() and res3["fig_path"].stat().st_size > 0
    fig5_ok = res5["fig_path"].exists() and res5["fig_path"].stat().st_size > 0
    fit3_ok = res3["fit_max_err"] < 1e-2
    fit5_ok = res5["fit_max_err"] < 1e-2

    print(f"figure3_full path     : {res3['fig_path']}  ok={fig3_ok}")
    print(f"figure5_scheme path   : {res5['fig_path']}  ok={fig5_ok}")
    print(f"fig3 max ‖ϕ(S)-T‖     : {res3['fit_max_err']:.3e}  ok={fit3_ok}")
    print(f"fig5 max ‖ϕ(S)-T‖     : {res5['fit_max_err']:.3e}  ok={fit5_ok}")
    passed = fig3_ok and fig5_ok and fit3_ok and fit5_ok
    print(f"PHASE4 SMOKE: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
