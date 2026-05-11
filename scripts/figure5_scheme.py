"""Phase 4 — Sec. IV-C: full Fig. 5 transportation scheme.

Reproduces the 2×2 schematic of Fig. 5:

* (a) top-left  — source demonstration ``x`` and its source-frame DS ``f``.
* (b) top-right — source distribution ``S`` (the flat surface).
* (c) bottom-left — transported demonstration ``x̂`` and the refit
  target-frame DS ``f̂``.
* (d) bottom-right — target distribution ``T`` (the curved surface).

Steps:

1. Build a small letter-C demonstration (Phase 2).
2. Build paired flat / curved 1-D surfaces in 2-D (Phase 2).
3. Fit :class:`PolicyTransport` on ``(S=flat, T=curved)``.
4. Fit a source-frame :class:`GPDynamicalSystem` on the demo's
   ``(x, ẋ)``.
5. Transport the demo: ``x̂ = ϕ(x)``, ``ẋ̂ = J(x) ẋ``.
6. Fit a target-frame :class:`GPDynamicalSystem` on ``(x̂, ẋ̂)`` to obtain
   the transported policy ``f̂`` (Sec. IV-C, the f̂ box in Fig. 4).
7. Draw the four-panel figure via :func:`plot_phi_scheme`.

CLI flags (per CLAUDE.md):

* ``--seed``       RNG seed (default 0)
* ``--n_demos``    number of demo samples (default 60)
* ``--n_source_points`` number of surface (S, T) pairs (default 18)
* ``--out_dir``    figure directory (default ``reports/figures/``)
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

from gpt_repro.policies import (  # noqa: E402
    GPDynamicalSystem,
    make_letter_C_demo,
    make_surface_2d,
)
from gpt_repro.transport import PolicyTransport  # noqa: E402
from gpt_repro.utils import set_global_seed  # noqa: E402
from gpt_repro.viz import plot_phi_scheme  # noqa: E402


def _save_metadata(
    fig: plt.Figure, *, section: str, seed: int, n_demos: int, n_source: int
) -> None:
    footer = (
        f"section={section}  seed={seed}  n_demos={n_demos}  "
        f"n_source_points={n_source}  "
        f"timestamp={_dt.datetime.now().isoformat(timespec='seconds')}"
    )
    fig.text(0.5, 0.005, footer, ha="center", fontsize=7, color="grey")


def make_figure(
    seed: int = 0,
    n_demos: int = 60,
    n_source_points: int = 18,
    out_dir: Path = _REPO_ROOT / "reports" / "figures",
    save: bool = True,
    n_iter_gp: int = 200,
) -> dict:
    set_global_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Demo + 2. Surfaces.
    demo = make_letter_C_demo(n_points=n_demos, duration=1.0, radius=0.8)
    S_surface = make_surface_2d("flat", n_points=n_source_points,
                                x_range=(-1.4, 1.4))
    T_surface = make_surface_2d("curved", n_points=n_source_points,
                                x_range=(-1.4, 1.4), amplitude=0.5)

    # 3. Policy transport.
    pt = PolicyTransport(n_iter_default=n_iter_gp + 100, lr=0.1).fit(
        S_surface, T_surface
    )

    # 4. Source DS.
    ds_src = GPDynamicalSystem(n_iter_default=n_iter_gp, lr=0.1).fit(
        demo["x"], demo["xdot"]
    )

    # 5. Transport demo positions + velocities.
    x_hat = pt.transform(demo["x"])
    xdot_hat = pt.transform_velocity(demo["x"], demo["xdot"])

    # 6. Target DS refit on transported labels (the f̂ box of Fig. 4).
    ds_tgt = GPDynamicalSystem(n_iter_default=n_iter_gp, lr=0.1).fit(
        x_hat, xdot_hat
    )

    # Diagnostics.
    fit_err = float(np.max(np.linalg.norm(pt.transform(S_surface) - T_surface, axis=1)))

    fig_path = None
    if save:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0))
        ax_grid = list(axes.flatten())

        # Use the same plot extents as in the demo+surface bounding boxes,
        # so the (a)/(b) and (c)/(d) panels share scale.
        margin = 0.5
        src_pts = np.vstack([demo["x"], S_surface])
        x_range_src = (
            float(src_pts[:, 0].min() - margin),
            float(src_pts[:, 0].max() + margin),
        )
        y_range_src = (
            float(src_pts[:, 1].min() - margin),
            float(src_pts[:, 1].max() + margin),
        )
        tgt_pts = np.vstack([x_hat, T_surface])
        x_range_tgt = (
            float(tgt_pts[:, 0].min() - margin),
            float(tgt_pts[:, 0].max() + margin),
        )
        y_range_tgt = (
            float(tgt_pts[:, 1].min() - margin),
            float(tgt_pts[:, 1].max() + margin),
        )

        plot_phi_scheme(
            demo_x=demo["x"], demo_xdot=demo["xdot"],
            S=S_surface, T=T_surface,
            transport=pt,
            ax_grid=ax_grid,
            ds_source=ds_src,
            ds_target=ds_tgt,
            x_range_source=x_range_src, y_range_source=y_range_src,
            x_range_target=x_range_tgt, y_range_target=y_range_tgt,
        )

        # Overlay surface curves for visual context.
        ax_grid[1].plot(S_surface[:, 0], S_surface[:, 1], "-",
                        color="tab:blue", alpha=0.5, lw=1.0)
        ax_grid[3].plot(T_surface[:, 0], T_surface[:, 1], "-",
                        color="tab:red", alpha=0.5, lw=1.0)

        fig.suptitle(
            "Phase 4 — Sec. IV-C: Transportation of demo and dynamics",
            y=0.99,
        )
        _save_metadata(fig, section="IV-C", seed=seed,
                       n_demos=n_demos, n_source=n_source_points)
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        fig_path = out_dir / "phase4_fig5_scheme.png"
        fig.savefig(fig_path, dpi=300)
        fig.savefig(fig_path.with_suffix(".pdf"))
        plt.close(fig)

    return {
        "fig_path": fig_path,
        "fit_max_err": fit_err,
        "n_demos": n_demos,
        "n_source_points": n_source_points,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_demos", type=int, default=60)
    parser.add_argument("--n_source_points", type=int, default=18)
    parser.add_argument(
        "--out_dir", type=Path, default=_REPO_ROOT / "reports" / "figures"
    )
    args = parser.parse_args()

    res = make_figure(
        seed=args.seed,
        n_demos=args.n_demos,
        n_source_points=args.n_source_points,
        out_dir=args.out_dir,
        save=True,
    )
    print(f"max ‖ϕ(S) - T‖ at surface points : {res['fit_max_err']:.6e}")
    print(f"Saved figure: {res['fig_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
