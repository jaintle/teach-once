"""Phase 6 — Sec. V-A: 2D surface cleaning comparison (Fig. 7 + Table I).

Setup matches the paper's Sec. V-A:

* periodic cleaning demonstration from Phase 2,
* source surface ``S`` = flat 1-D curve in 2-D,
* target surface ``T`` = curved 1-D curve in 2-D,
* :class:`LinearTransport` γ fit on ``(S, T)`` and applied to **all**
  trajectories before any baseline runs.

Six baselines (KMP, LE, E-RF, E-NN, E-NF, GP) are then fit on the
residual ``(γ(S), T)`` and used to transport the cleaning demonstration.
Each panel of the resulting 2 × 3 figure shows: the target surface T
(black), the transported demo (colored line for the mean), per-member
spread (light colored lines for ensembles), and a ±2σ uncertainty band
(orange shading) where available.

Prints: per-method mean / max distance of the transported demo to the
target surface and a Table I summary.

CLI flags:

* ``--seed``    RNG seed (default 0)
* ``--n_demos`` cleaning-demo sample count (default 120)
* ``--n_source_points`` number of (S, T) anchor pairs (default 24)
* ``--out_dir`` figure directory (default ``reports/figures/``)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gpt_repro.baselines import BASELINE_NAMES, BASELINES  # noqa: E402
from gpt_repro.metrics.table1 import print_and_save  # noqa: E402
from gpt_repro.policies import (  # noqa: E402
    make_cleaning_demo,
    make_surface_2d,
)
from gpt_repro.transport import LinearTransport  # noqa: E402
from gpt_repro.utils import set_global_seed  # noqa: E402

# Paper's panel order (top-row, bottom-row).
PANEL_ORDER = ["kmp", "erf", "enn", "le", "enf", "gp"]


def _save_metadata(fig: plt.Figure, *, section: str, seed: int,
                   n_demos: int, n_source: int) -> None:
    footer = (
        f"section={section}  seed={seed}  n_demos={n_demos}  "
        f"n_source_points={n_source}  "
        f"timestamp={_dt.datetime.now().isoformat(timespec='seconds')}"
    )
    fig.text(0.5, 0.005, footer, ha="center", fontsize=7, color="grey")


def _build_baseline(name: str, seed: int):
    """Instantiate a baseline with comparison-friendly defaults."""
    cls = BASELINES[name]
    if name == "gp":
        return cls(n_iter_default=200, lr=0.1)
    if name in ("enn", "enf"):
        return cls(n_members=6, n_epochs=400, random_state=seed)
    if name == "erf":
        return cls(n_estimators=8, max_depth=8, random_state=seed)
    return cls()


def _surface_distance(traj: np.ndarray, surface: np.ndarray) -> float:
    """Mean distance from each trajectory point to its nearest surface point."""
    d = np.linalg.norm(traj[:, None, :] - surface[None, :, :], axis=-1)
    return float(d.min(axis=1).mean())


def make_figure(
    seed: int = 0,
    n_demos: int = 120,
    n_source_points: int = 24,
    out_dir: Path = _REPO_ROOT / "reports" / "figures",
    save: bool = True,
) -> Dict:
    set_global_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = _REPO_ROOT / "reports" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ----- Scenario --------------------------------------------------
    demo = make_cleaning_demo(n_points=n_demos, n_cycles=3,
                              x_range=(-1.0, 1.0), lift_height=0.4)
    S_surface = make_surface_2d("flat",  n_points=n_source_points,
                                x_range=(-1.2, 1.2))
    T_surface = make_surface_2d("curved", n_points=n_source_points,
                                x_range=(-1.2, 1.2), amplitude=0.4)

    # ----- Apply γ to everything *first* (paper's protocol) ----------
    gamma = LinearTransport().fit(S_surface, T_surface)
    S_lin     = gamma.transform(S_surface)
    demo_lin  = gamma.transform(demo["x"])

    # ----- Fit each baseline on the residual + transport demo --------
    method_results: Dict[str, Dict] = {}
    for name in PANEL_ORDER:
        b = _build_baseline(name, seed=seed).fit(S_lin, T_surface)
        delta_mean = b.transform(demo_lin)                       # (N, 2)
        traj_mean  = demo_lin + delta_mean                       # transported

        # Per-member trajectories (if applicable).
        per_member = None
        if hasattr(b, "per_member_predictions"):
            per_delta = b.per_member_predictions(demo_lin)       # (M, N, 2)
            per_member = demo_lin[None, :, :] + per_delta

        # Std along the transported trajectory.
        _, std_delta = b.predict_with_std(demo_lin)
        std_scalar = (
            np.sqrt((std_delta ** 2).sum(axis=1))
            if std_delta is not None else None
        )

        mean_dist = _surface_distance(traj_mean, T_surface)
        method_results[name] = {
            "traj_mean": traj_mean,
            "per_member": per_member,
            "std_scalar": std_scalar,
            "mean_dist_to_surface": mean_dist,
        }

    # ----- Figure ----------------------------------------------------
    fig_path = None
    if save:
        fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5))
        ax_grid = axes.flatten()
        x_lo, x_hi = -1.4, 1.4
        y_lo, y_hi = -0.4, 0.7
        member_colors = plt.get_cmap("tab10")

        for i, name in enumerate(PANEL_ORDER):
            ax = ax_grid[i]
            res = method_results[name]
            traj = res["traj_mean"]
            per_member = res["per_member"]
            std_scalar = res["std_scalar"]

            # Black target surface.
            ax.plot(T_surface[:, 0], T_surface[:, 1], "-",
                    color="black", lw=2.0, label="target surface T")

            # ±2σ band (perpendicular-to-trajectory).
            if std_scalar is not None:
                tang = np.diff(traj, axis=0)
                tang = np.concatenate([tang, tang[-1:]], axis=0)
                tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
                norm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
                w = 2.0 * std_scalar[:, None]
                upper = traj + norm * w
                lower = traj - norm * w
                poly_x = np.concatenate([upper[:, 0], lower[::-1, 0]])
                poly_y = np.concatenate([upper[:, 1], lower[::-1, 1]])
                ax.fill(poly_x, poly_y, color="tab:orange",
                        alpha=0.25, linewidth=0.0, label="±2 σ band")

            # Per-member trajectories.
            if per_member is not None:
                n_members = per_member.shape[0]
                for k in range(n_members):
                    ax.plot(per_member[k, :, 0], per_member[k, :, 1],
                            "-", color=member_colors(k % 10),
                            alpha=0.45, lw=0.8)

            # Mean transported trajectory.
            ax.plot(traj[:, 0], traj[:, 1], "-",
                    color="tab:red", lw=1.7, label="transported demo")

            ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.set_title(
                f"{BASELINE_NAMES[name]} — mean dist = "
                f"{res['mean_dist_to_surface']:.3f}"
            )
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc="upper left", fontsize=7)

        fig.suptitle(
            "Phase 6 — Sec. V-A: 2D surface cleaning comparison",
            y=0.99,
        )
        _save_metadata(fig, section="V-A", seed=seed,
                       n_demos=n_demos, n_source=n_source_points)
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        fig_path = out_dir / "phase6_fig7_comparison.png"
        fig.savefig(fig_path, dpi=300)
        fig.savefig(fig_path.with_suffix(".pdf"))
        plt.close(fig)

    # Table I.
    table_rows = print_and_save(results_dir / "table1.csv")

    return {
        "fig_path": fig_path,
        "method_results": method_results,
        "table_rows": table_rows,
        "T_surface": T_surface,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_demos", type=int, default=120)
    parser.add_argument("--n_source_points", type=int, default=24)
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

    print()
    print("Per-method transported-demo distance to target surface:")
    for name in PANEL_ORDER:
        r = res["method_results"][name]
        std_summary = (
            f"std mean = {r['std_scalar'].mean():.4f}"
            if r["std_scalar"] is not None else "std = N/A"
        )
        print(
            f"  {BASELINE_NAMES[name]:>5}: "
            f"mean dist = {r['mean_dist_to_surface']:.4f}, {std_summary}"
        )
    print()
    print(f"Saved figure: {res['fig_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
