"""Phase 4 — Sec. IV-B: full Fig. 3 with the GP-transportation panel.

Extends Phase 3's ``figure3_linear.py`` with the fourth panel —
``GP Transportation`` — by reusing :func:`plot_grid_under_transform`
with ``transform_fn = PolicyTransport.transform``.

Saves ``reports/figures/phase4_fig3_full.png``. Phase 3's
``phase3_fig3_partial.png`` is left in place.

CLI flags (per CLAUDE.md):

* ``--seed``              RNG seed (default 0)
* ``--n_source_points``   number of paired (S, T) points (default 18)
* ``--out_dir``           figure directory (default ``reports/figures/``)
"""

from __future__ import annotations

import argparse
import datetime as _dt
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
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from figure3_linear import _build_source_target  # noqa: E402

from gpt_repro.transport import PolicyTransport  # noqa: E402
from gpt_repro.utils import set_global_seed  # noqa: E402
from gpt_repro.viz import (  # noqa: E402
    plot_distribution_match,
    plot_grid_under_transform,
)


def _save_metadata(fig: plt.Figure, *, section: str, seed: int, n_source: int) -> None:
    footer = (
        f"section={section}  seed={seed}  n_source_points={n_source}  "
        f"timestamp={_dt.datetime.now().isoformat(timespec='seconds')}"
    )
    fig.text(0.5, 0.005, footer, ha="center", fontsize=7, color="grey")


def make_figure(
    seed: int = 0,
    n_source_points: int = 18,
    out_dir: Path = _REPO_ROOT / "reports" / "figures",
    save: bool = True,
) -> dict:
    set_global_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    S, T, R_true, t_true, nl_amp = _build_source_target(n_source_points, seed)
    pt = PolicyTransport(n_iter_default=300, lr=0.1).fit(S, T)

    margin = 0.4
    x_range = (float(S[:, 0].min() - margin), float(S[:, 0].max() + margin))
    y_range = (float(S[:, 1].min() - margin), float(S[:, 1].max() + margin))

    phi_at_S = pt.transform(S)
    fit_max_err = float(np.max(np.linalg.norm(phi_at_S - T, axis=1)))

    fig_path = None
    if save:
        fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.6))

        plot_distribution_match(S, T, ax=axes[0], title="1. Distribution Match")
        plot_grid_under_transform(
            transform_fn=lambda X: X,
            x_range=x_range, y_range=y_range, n_grid=12, ax=axes[1],
            title="2. Source Distribution",
            overlay_points={"source": S},
        )
        plot_grid_under_transform(
            transform_fn=pt.gamma.transform,
            x_range=x_range, y_range=y_range, n_grid=12, ax=axes[2],
            title="3. Linear Transformation (γ)",
            overlay_points={"source": S, "target": T},
        )
        plot_grid_under_transform(
            transform_fn=pt.transform,
            x_range=x_range, y_range=y_range, n_grid=12, ax=axes[3],
            title="4. GP Transportation (ϕ = γ + ψ)",
            overlay_points={"source": S, "target": T},
        )

        fig.suptitle("Phase 4 — Sec. IV-B: Full ϕ = γ + ψ", y=0.98)
        _save_metadata(fig, section="IV-B", seed=seed, n_source=n_source_points)
        fig.tight_layout(rect=(0, 0.04, 1, 0.94))
        fig_path = out_dir / "phase4_fig3_full.png"
        fig.savefig(fig_path, dpi=300)
        fig.savefig(fig_path.with_suffix(".pdf"))
        plt.close(fig)

    return {
        "fig_path": fig_path,
        "fit_max_err": fit_max_err,
        "A": pt.gamma.A,
        "det_A": float(np.linalg.det(pt.gamma.A)),
        "centroid_shift": pt.gamma.T_bar - pt.gamma.S_bar,
        "R_true": R_true,
        "t_true": t_true,
        "perturbation_amplitude": nl_amp,
        "n_source_points": n_source_points,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_source_points", type=int, default=18)
    parser.add_argument(
        "--out_dir", type=Path, default=_REPO_ROOT / "reports" / "figures"
    )
    parser.add_argument("--n_demos", type=int, default=0,
                        help="Unused in figure3_full; kept for CLI consistency.")
    args = parser.parse_args()

    res = make_figure(
        seed=args.seed,
        n_source_points=args.n_source_points,
        out_dir=args.out_dir,
        save=True,
    )

    np.set_printoptions(precision=4, suppress=True)
    print("Recovered γ rotation A:")
    print(res["A"])
    print(f"det(A)                  : {res['det_A']:.6f}")
    print(f"centroid shift T̄-S̄    : {res['centroid_shift']}")
    print(f"max ‖ϕ(S) - T‖ at S     : {res['fit_max_err']:.6e}")
    print(f"non-linear perturbation : {res['perturbation_amplitude']:.4f}")
    print(f"Saved figure: {res['fig_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
