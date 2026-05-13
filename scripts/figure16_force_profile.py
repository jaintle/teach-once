#!/usr/bin/env python
"""Figure 16 analog — Phase 10: Force profile generalization (Sec. VI-C).

Plots force norm over time for 5 surface variants, overlaid with the
demo force profile (dashed black). Prints Pearson correlation per surface.

Saves to reports/figures/phase10_fig16_force.png.

CLI: --seed, --n_inducing, --out_dir, --fast
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from gpt_repro.policies.surfaces_3d import SurfaceConfig, make_surface_demo
from gpt_repro.transport.cleaning_pipeline_3d import run_cleaning_pipeline
from gpt_repro.utils.seeding import set_global_seed


COLORS = ["steelblue", "darkorange", "seagreen", "crimson", "purple"]
LABELS = ["flat (demo)", "tilted", "curved", "bumpy", "tilted+curved"]

SOURCE_CFG = SurfaceConfig("flat", np.array([0.5, 0.0, 0.5]))

SURFACE_VARIANTS = [
    SurfaceConfig("flat",    np.array([0.5, 0.0, 0.5])),
    SurfaceConfig("tilted",  np.array([0.5, 0.0, 0.5]),
                  normal=np.array([0.0, 0.3, 1.0])),
    SurfaceConfig("curved",  np.array([0.5, 0.0, 0.5]),
                  params={"amplitude": 0.05, "frequency": 2*np.pi}),
    SurfaceConfig("bumpy",   np.array([0.5, 0.0, 0.5]),
                  params={"n_bumps": 5, "bump_amplitude": 0.04}),
    SurfaceConfig("curved",  np.array([0.5, 0.0, 0.6]),
                  params={"amplitude": 0.04, "frequency": 3*np.pi}),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_inducing", type=int, default=100)
    parser.add_argument("--out_dir", type=str,
                        default=str(_REPO_ROOT / "reports" / "figures"))
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    set_global_seed(args.seed)
    n_pts = 100 if args.fast else 400
    n_ind = 20 if args.fast else args.n_inducing
    n_demo = 50 if args.fast else 100
    n_iter = 50 if args.fast else 300
    n_steps = 60 if args.fast else 150

    # Demo force profile (source surface)
    demo = make_surface_demo(SOURCE_CFG, n_points=n_demo, seed=args.seed)
    demo_forces = np.array([
        np.linalg.norm(demo["stiffness"][i] @ demo["xdot"][i])
        for i in range(len(demo["xdot"]))
    ])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(demo_forces, "k--", linewidth=1.5, label="Demo (source)", zorder=5)

    print(f"\n{'Surface':<20} {'Mean F':>10} {'Pearson r':>12}")
    print("-" * 45)

    for i, (tgt_cfg, color, label) in enumerate(
        zip(SURFACE_VARIANTS, COLORS, LABELS)
    ):
        print(f"  Pipeline: {label} ...", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = run_cleaning_pipeline(
                SOURCE_CFG, tgt_cfg,
                n_source_pts=n_pts, n_target_pts=n_pts,
                n_inducing=n_ind, n_demo_pts=n_demo,
                seed=args.seed, n_iter=n_iter, n_steps=n_steps,
                gp_n_iter=80,
            )

        fn = r["force_norms"]
        t_axis = np.arange(len(fn))
        ax.plot(t_axis, fn, color=color, linewidth=1.2, label=label, alpha=0.85)

        mean_f = float(np.mean(fn))
        min_len = min(len(demo_forces), len(fn))
        if (min_len > 2
                and np.std(demo_forces[:min_len]) > 1e-10
                and np.std(fn[:min_len]) > 1e-10):
            pearson_r, _ = pearsonr(demo_forces[:min_len], fn[:min_len])
        else:
            pearson_r = float("nan")

        print(f"  {label:<20} {mean_f:>10.2f} {pearson_r:>12.4f}")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Force norm ‖F‖ = ‖Ks · ẋ‖  [N]")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    ax.set_title(
        f"Phase 10 — Sec. VI-C analog: Force profile generalization\n"
        f"seed={args.seed}  n_inducing={args.n_inducing}  {ts}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase10_fig16_force.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
