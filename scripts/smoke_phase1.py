"""Phase 1 smoke test — Sec. III-B Gaussian Process regression demo.

Fits an :class:`ExactGPRegressor` and an :class:`SVGPRegressor` to a noisy
1D toy function and produces a single comparison figure at
``reports/figures/phase1_gp_demo.png``. Prints train/test RMSE and a
one-line ``PHASE1 SMOKE: PASS`` / ``FAIL`` summary.

CLI flags (defaults in parentheses):

* ``--seed``      RNG seed (default 0)
* ``--out_dir``   figure directory (default ``reports/figures/``)
* ``--n_demos``   number of training points (default 60)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive, safe in CI / headless
import matplotlib.pyplot as plt
import numpy as np

# Make `import gpt_repro` work when the package is not pip-installed.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gpt_repro.gp import ExactGPRegressor, SVGPRegressor  # noqa: E402
from gpt_repro.utils import set_global_seed  # noqa: E402


def _toy_function(x: np.ndarray) -> np.ndarray:
    """Ground-truth 1D toy function used by the smoke demo."""
    return np.sin(1.2 * x) + 0.3 * np.cos(2.5 * x)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_dir", type=Path, default=_REPO_ROOT / "reports" / "figures"
    )
    parser.add_argument("--n_demos", type=int, default=60)
    parser.add_argument("--n_source_points", type=int, default=0,
                        help="Unused in Phase 1; kept for CLI consistency.")
    args = parser.parse_args()

    set_global_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = _REPO_ROOT / "reports" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data -----------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    X = rng.uniform(-3.0, 3.0, size=(args.n_demos, 1))
    X = np.sort(X, axis=0)
    noise_std = 0.05
    y = _toy_function(X[:, 0]) + noise_std * rng.standard_normal(args.n_demos)

    X_test = np.linspace(-3.5, 3.5, 300).reshape(-1, 1)
    y_true = _toy_function(X_test[:, 0])

    # ---- Exact GP -------------------------------------------------------
    exact_gp = ExactGPRegressor(n_iter_default=200, lr=0.1).fit(X, y)
    m_exact_tr, _ = exact_gp.predict(X)
    m_exact, s_exact = exact_gp.predict(X_test)
    rmse_exact_train = _rmse(m_exact_tr, y)
    rmse_exact_test = _rmse(m_exact, y_true)

    # ---- SVGP -----------------------------------------------------------
    svgp = SVGPRegressor(
        n_inducing=min(20, args.n_demos), n_iter_default=400, lr=0.05,
        batch_size=128,
    ).fit(X, y)
    m_svgp_tr, _ = svgp.predict(X)
    m_svgp, s_svgp = svgp.predict(X_test)
    rmse_svgp_train = _rmse(m_svgp_tr, y)
    rmse_svgp_test = _rmse(m_svgp, y_true)

    # ---- Pass/fail criteria --------------------------------------------
    passed = (rmse_exact_test < 0.1) and (rmse_svgp_test < 0.15)

    print(f"Exact GP : train RMSE = {rmse_exact_train:.4f}, test RMSE = {rmse_exact_test:.4f}")
    print(f"SVGP     : train RMSE = {rmse_svgp_train:.4f}, test RMSE = {rmse_svgp_test:.4f}")
    print(f"PHASE1 SMOKE: {'PASS' if passed else 'FAIL'}")

    # ---- Figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, mean, std, title in (
        (axes[0], m_exact, s_exact, f"Exact GP — RMSE={rmse_exact_test:.3f}"),
        (axes[1], m_svgp, s_svgp, f"SVGP — RMSE={rmse_svgp_test:.3f}"),
    ):
        ax.plot(X_test[:, 0], y_true, "k--", lw=1.2, label="ground truth")
        ax.plot(X[:, 0], y, "k.", ms=4, label="train")
        ax.plot(X_test[:, 0], mean, "C0-", lw=1.5, label="GP mean")
        ax.fill_between(
            X_test[:, 0], mean - 2 * std, mean + 2 * std,
            color="C0", alpha=0.2, label="±2σ",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("f(x)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Phase 1 — Sec. III-B: Exact GP vs SVGP on a noisy 1D function")

    footer = (
        f"section=III-B  seed={args.seed}  n_demos={args.n_demos}  "
        f"timestamp={_dt.datetime.now().isoformat(timespec='seconds')}"
    )
    fig.text(0.5, 0.005, footer, ha="center", fontsize=7, color="grey")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    fig_path = args.out_dir / "phase1_gp_demo.png"
    fig.savefig(fig_path, dpi=300)
    fig.savefig(fig_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

    # ---- Numerical results ---------------------------------------------
    summary = {
        "phase": 1,
        "section": "III-B",
        "seed": args.seed,
        "n_demos": args.n_demos,
        "rmse_exact_train": rmse_exact_train,
        "rmse_exact_test": rmse_exact_test,
        "rmse_svgp_train": rmse_svgp_train,
        "rmse_svgp_test": rmse_svgp_test,
        "passed": bool(passed),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    summary_path = results_dir / "phase1_gp_demo.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    csv_path = results_dir / "phase1_gp_demo.csv"
    with csv_path.open("w") as f:
        f.write("model,train_rmse,test_rmse\n")
        f.write(f"exact_gp,{rmse_exact_train:.6f},{rmse_exact_test:.6f}\n")
        f.write(f"svgp,{rmse_svgp_train:.6f},{rmse_svgp_test:.6f}\n")
    print(f"Saved results: {summary_path}")
    print(f"Saved results: {csv_path}")

    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
