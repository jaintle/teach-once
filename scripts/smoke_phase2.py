"""Phase 2 smoke test — Sec. III-A dynamical-system learning.

Generates the canonical letter-C and periodic cleaning demonstrations,
fits a GP-parameterized DS policy on the letter-C demo (Eq. 1), rolls
it out from the first demo state, and saves two figures:

* ``reports/figures/phase2_letter_C_field.png`` — learned velocity field
  colored by predictive std, with the demo (red) and rollout (black)
  overlaid.
* ``reports/figures/phase2_cleaning_demo.png`` — periodic cleaning
  demonstration on top of a flat surface (no DS yet — just shows the
  data that Phase 3+ will transport).

Prints train RMSE, the rollout's closest approach to the demo endpoint,
the on-demo / out-of-distribution std ratio, and a one-line PASS/FAIL.

CLI flags (per CLAUDE.md):

* ``--seed``      RNG seed (default 0)
* ``--out_dir``   figure directory (default ``reports/figures/``)
* ``--n_demos``   number of demonstration samples (default 100)
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
    make_cleaning_demo,
    make_letter_C_demo,
    make_surface_2d,
)
from gpt_repro.utils import set_global_seed  # noqa: E402
from gpt_repro.viz import plot_vector_field  # noqa: E402


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _save_metadata(fig: plt.Figure, *, section: str, seed: int, n_demos: int) -> None:
    footer = (
        f"section={section}  seed={seed}  n_demos={n_demos}  "
        f"timestamp={_dt.datetime.now().isoformat(timespec='seconds')}"
    )
    fig.text(0.5, 0.005, footer, ha="center", fontsize=7, color="grey")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_dir", type=Path, default=_REPO_ROOT / "reports" / "figures"
    )
    parser.add_argument("--n_demos", type=int, default=100)
    parser.add_argument("--n_source_points", type=int, default=0,
                        help="Unused in Phase 2; kept for CLI consistency.")
    args = parser.parse_args()

    set_global_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = _REPO_ROOT / "reports" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Demonstrations
    # ------------------------------------------------------------------
    letter_C = make_letter_C_demo(n_points=args.n_demos, duration=1.0)
    cleaning = make_cleaning_demo(n_points=max(60, args.n_demos), n_cycles=3)
    surface_flat = make_surface_2d("flat", n_points=40)

    # ------------------------------------------------------------------
    # Fit DS on letter-C
    # ------------------------------------------------------------------
    ds = GPDynamicalSystem(n_iter_default=200, lr=0.1).fit(
        letter_C["x"], letter_C["xdot"]
    )
    pred_train, std_train = ds.predict(letter_C["x"], return_std=True)
    train_rmse = _rmse(pred_train, letter_C["xdot"])
    on_demo_std = float(np.linalg.norm(std_train, axis=1).mean())

    # OOD std at far-field corners
    far_pts = np.array([[3.0, 3.0], [-3.0, -3.0], [4.0, 0.0]])
    _, std_far = ds.predict(far_pts, return_std=True)
    far_std = float(np.linalg.norm(std_far, axis=1).mean())

    # Rollout
    traj, _ = ds.rollout(letter_C["x"][0], dt=0.025, n_steps=80)
    dists = np.linalg.norm(traj - letter_C["x"][-1], axis=1)
    closest = float(dists.min())
    closest_idx = int(np.argmin(dists))

    # ------------------------------------------------------------------
    # Pass / fail criteria
    # ------------------------------------------------------------------
    passed = (
        train_rmse < 0.2
        and closest < 0.4
        and far_std > on_demo_std
    )

    print(f"DS train RMSE             : {train_rmse:.4f}")
    print(f"Rollout closest approach  : {closest:.4f} at step {closest_idx}")
    print(f"On-demo / OOD mean std    : {on_demo_std:.4f} / {far_std:.4f}")
    print(f"PHASE2 SMOKE: {'PASS' if passed else 'FAIL'}")

    # ------------------------------------------------------------------
    # Figure 1 — learned field for the letter-C demo
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(6.0, 5.4))
    plot_vector_field(
        ds,
        x_range=(-1.8, 1.8),
        y_range=(-1.8, 1.8),
        n_grid=22,
        ax=ax1,
        cmap_by_std=True,
        demo={"x": letter_C["x"]},
        rollout=traj[: closest_idx + 1],  # plot rollout up through closest approach
    )
    ax1.set_title("Phase 2 — Sec. III-A: Learned DS for letter C demonstration")
    _save_metadata(fig1, section="III-A", seed=args.seed, n_demos=args.n_demos)
    fig1.tight_layout(rect=(0, 0.03, 1, 1))
    fig1_path = args.out_dir / "phase2_letter_C_field.png"
    fig1.savefig(fig1_path, dpi=300)
    fig1.savefig(fig1_path.with_suffix(".pdf"))
    plt.close(fig1)
    print(f"Saved figure: {fig1_path}")

    # ------------------------------------------------------------------
    # Figure 2 — cleaning demo on a flat surface (no DS)
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7.5, 3.0))
    ax2.plot(surface_flat[:, 0], surface_flat[:, 1],
             color="0.4", lw=2.0, label="surface (flat)")
    ax2.plot(cleaning["x"][:, 0], cleaning["x"][:, 1],
             "o-", color="red", ms=3, lw=0.8, label="cleaning demo")
    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-0.2, 0.55)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_title("Phase 2 — Sec. III-A: Cleaning demonstration over a flat surface")
    _save_metadata(fig2, section="III-A", seed=args.seed,
                   n_demos=cleaning["x"].shape[0])
    fig2.tight_layout(rect=(0, 0.05, 1, 1))
    fig2_path = args.out_dir / "phase2_cleaning_demo.png"
    fig2.savefig(fig2_path, dpi=300)
    fig2.savefig(fig2_path.with_suffix(".pdf"))
    plt.close(fig2)
    print(f"Saved figure: {fig2_path}")

    # ------------------------------------------------------------------
    # Numerical results
    # ------------------------------------------------------------------
    summary = {
        "phase": 2,
        "section": "III-A",
        "seed": args.seed,
        "n_demos": args.n_demos,
        "train_rmse": train_rmse,
        "rollout_closest_approach": closest,
        "rollout_closest_step": closest_idx,
        "on_demo_mean_std": on_demo_std,
        "ood_mean_std": far_std,
        "passed": bool(passed),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    summary_path = results_dir / "phase2_ds_demo.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    csv_path = results_dir / "phase2_ds_demo.csv"
    with csv_path.open("w") as f:
        f.write("metric,value\n")
        f.write(f"train_rmse,{train_rmse:.6f}\n")
        f.write(f"rollout_closest_approach,{closest:.6f}\n")
        f.write(f"on_demo_mean_std,{on_demo_std:.6f}\n")
        f.write(f"ood_mean_std,{far_std:.6f}\n")
    print(f"Saved results: {summary_path}")
    print(f"Saved results: {csv_path}")

    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
