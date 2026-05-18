"""Phase 5 — Fig. 6: Transportation / Epistemic / Total uncertainty fields.

Reproduces the three 3-D surface panels of Fig. 6 in Franzese et al.
(2024) for the 2-D letter-C-on-curved-surface scenario:

1. **Transportation Uncertainty**  — Σ_x̂ (Eq. 17): variance of the
   transported velocity label given the source velocity and the
   uncertain GP Jacobian of ψ.
2. **Epistemic Uncertainty**       — Σ_f̂   (Eq. 3 applied to the target
   GP DS): how confidently ``f̂`` predicts the transported velocity.
3. **Total Uncertainty**           — Σ_total (Eq. 18) = Σ_x̂ + Σ_f̂.

Each (M, d) per-axis variance field is reduced to a scalar **standard
deviation** field via the L2 norm of the std vector at each grid point:

    .. math:: \\sigma(x) = \\sqrt{\\sum_k \\Sigma_{kk}(x)}
                        = \\| \\sigma_{\\mathrm{per-axis}}(x) \\|_2.

This matches the Fig. 6 caption ("norm of the velocity") and makes
the three panels directly comparable on a common color / z scale.

CLI flags (with defaults):

* ``--seed``     RNG seed (default 0)
* ``--n_grid``   resolution per axis of the (G × G) field (default 30)
* ``--n_demos``  number of demo samples (default 60)
* ``--out_dir``  figure directory (default ``reports/figures/``)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gpt_repro.policies import (  # noqa: E402
    GPDynamicalSystem,
    make_letter_C_demo,
    make_surface_2d,
)
from gpt_repro.transport import (  # noqa: E402
    PolicyTransport,
    total_velocity_variance,
    transportation_velocity_variance,
)
from gpt_repro.utils import set_global_seed  # noqa: E402
from gpt_repro.viz import plot_uncertainty_triptych  # noqa: E402


def _l2_std_from_variance(var_field: np.ndarray) -> np.ndarray:
    """Reduce a (..., d) variance field to a scalar std via
    :math:`\\sigma = \\|\\,\\sqrt{\\Sigma}\\,\\|_2`."""
    return np.sqrt(np.maximum(var_field, 0.0).sum(axis=-1))


def make_figure(
    seed: int = 0,
    n_grid: int = 30,
    n_demos: int = 60,
    n_source_points: int = 18,
    out_dir: Path = _REPO_ROOT / "reports" / "figures",
    save: bool = True,
    n_iter_gp: int = 200,
) -> dict:
    set_global_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Scenario --------------------------------------------------
    demo = make_letter_C_demo(n_points=n_demos, duration=1.0, radius=0.8)
    S_surface = make_surface_2d("flat", n_points=n_source_points,
                                x_range=(-1.4, 1.4))
    T_surface = make_surface_2d("curved", n_points=n_source_points,
                                x_range=(-1.4, 1.4), amplitude=0.5)
    pt = PolicyTransport(n_iter_default=n_iter_gp + 100, lr=0.1).fit(
        S_surface, T_surface
    )
    ds_src = GPDynamicalSystem(n_iter_default=n_iter_gp, lr=0.1).fit(
        demo["x"], demo["xdot"]
    )
    x_hat = pt.transform(demo["x"])
    xdot_hat = pt.transform_velocity(demo["x"], demo["xdot"])
    ds_tgt = GPDynamicalSystem(n_iter_default=n_iter_gp, lr=0.1).fit(
        x_hat, xdot_hat
    )

    # ----- Source-frame grid -----------------------------------------
    bounds = np.vstack([demo["x"], S_surface])
    margin = 0.3
    xs = np.linspace(bounds[:, 0].min() - margin,
                     bounds[:, 0].max() + margin, n_grid)
    ys = np.linspace(bounds[:, 1].min() - margin,
                     bounds[:, 1].max() + margin, n_grid)
    XX, YY = np.meshgrid(xs, ys)
    X_grid_src = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (G², 2)

    # ----- Velocities and the three variance fields ------------------
    Xdot_grid, _ = ds_src.predict_with_std(X_grid_src)
    Sigma_xhat = transportation_velocity_variance(pt, X_grid_src, Xdot_grid)
    Sigma_total = total_velocity_variance(ds_tgt, pt, X_grid_src, Xdot_grid)
    Sigma_fhat = Sigma_total - Sigma_xhat
    # Numerical floor — total_velocity_variance ≥ transportation by construction,
    # but clamp to be defensive.
    Sigma_fhat = np.maximum(Sigma_fhat, 0.0)

    # ----- L2-norm reduction to scalar std fields --------------------
    std_xhat = _l2_std_from_variance(Sigma_xhat).reshape(n_grid, n_grid)
    std_fhat = _l2_std_from_variance(Sigma_fhat).reshape(n_grid, n_grid)
    std_total = _l2_std_from_variance(Sigma_total).reshape(n_grid, n_grid)
    Xg = np.stack([XX, YY], axis=-1)  # (G, G, 2)

    # ----- Numerical diagnostics -------------------------------------
    summary = {
        "mean_transport_std": float(std_xhat.mean()),
        "max_transport_std":  float(std_xhat.max()),
        "mean_epistemic_std": float(std_fhat.mean()),
        "max_epistemic_std":  float(std_fhat.max()),
        "mean_total_std":     float(std_total.mean()),
        "max_total_std":      float(std_total.max()),
    }
    # "OOD" detector: fraction of grid cells where total std exceeds 2× the
    # median transportation std (a deliberately crude rule).
    median_trans = float(np.median(std_xhat))
    ood_threshold = 2.0 * median_trans
    ood_fraction = float((std_total > ood_threshold).mean())
    summary.update({
        "median_transport_std": median_trans,
        "ood_threshold":        ood_threshold,
        "ood_fraction":         ood_fraction,
    })

    fig_path = None
    if save:
        suptitle = (
            "Phase 5 — Sec. IV-E: Uncertainty fields (Eqs. 16–18)"
        )
        fig = plot_uncertainty_triptych(
            Xg=Xg,
            std_transport=std_xhat,
            std_epistemic=std_fhat,
            std_total=std_total,
            suptitle=suptitle,
        )
        # Add metadata footer.
        footer = (
            f"section=IV-E  seed={seed}  n_grid={n_grid}  n_demos={n_demos}  "
            f"timestamp={_dt.datetime.now().isoformat(timespec='seconds')}"
        )
        fig.text(0.5, 0.005, footer, ha="center", fontsize=7, color="grey")
        fig_path = out_dir / "phase5_fig6_uncertainty.png"
        fig.savefig(fig_path, dpi=300)
        fig.savefig(fig_path.with_suffix(".pdf"))
        plt.close(fig)

    return {
        "fig_path": fig_path,
        "summary": summary,
        "std_xhat": std_xhat,
        "std_fhat": std_fhat,
        "std_total": std_total,
        "Xg": Xg,
        "n_grid": n_grid,
        "n_demos": n_demos,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_grid", type=int, default=30)
    parser.add_argument("--n_demos", type=int, default=60)
    parser.add_argument("--n_source_points", type=int, default=18,
                        help="Surface (S, T) pair count.")
    parser.add_argument(
        "--out_dir", type=Path, default=_REPO_ROOT / "reports" / "figures"
    )
    args = parser.parse_args()

    res = make_figure(
        seed=args.seed,
        n_grid=args.n_grid,
        n_demos=args.n_demos,
        n_source_points=args.n_source_points,
        out_dir=args.out_dir,
        save=True,
    )
    s = res["summary"]
    print("Std-field summaries (L2-norm reduced across output dims):")
    print(f"  transport : mean = {s['mean_transport_std']:.4f}  max = {s['max_transport_std']:.4f}")
    print(f"  epistemic : mean = {s['mean_epistemic_std']:.4f}  max = {s['max_epistemic_std']:.4f}")
    print(f"  total     : mean = {s['mean_total_std']:.4f}  max = {s['max_total_std']:.4f}")
    print(f"  median(transport) = {s['median_transport_std']:.4f}, "
          f"OOD threshold (2× median) = {s['ood_threshold']:.4f}")
    print(f"  fraction of cells with total > OOD threshold : {s['ood_fraction']:.3f}")
    print(f"Saved figure: {res['fig_path']}")

    # Persist numbers for the report.
    results_dir = _REPO_ROOT / "reports" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "phase5_uncertainty.json").write_text(
        json.dumps({**s, "n_grid": res["n_grid"], "n_demos": res["n_demos"],
                    "seed": res["seed"]}, indent=2)
    )
    print(f"Saved results: {results_dir / 'phase5_uncertainty.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
