#!/usr/bin/env python
"""Figure 15 analog — Phase 10: Surface cleaning generalization (Sec. VI-C).

Runs run_cleaning_pipeline for 5 target surface variants and plots:
- Top row: 3D rollout + transported demo on each surface.
- Bottom row: source (blue) and target (orange) point clouds.

Saves to reports/figures/phase10_fig15_cleaning.png.

CLI: --seed, --n_inducing, --out_dir, --fast (uses n_pts=100, n_ind=20)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from gpt_repro.policies.surfaces_3d import SurfaceConfig
from gpt_repro.transport.cleaning_pipeline_3d import run_cleaning_pipeline
from gpt_repro.utils.seeding import set_global_seed


SURFACE_VARIANTS = [
    ("flat (demo)",    SurfaceConfig("flat",    np.array([0.5, 0.0, 0.5]))),
    ("tilted",         SurfaceConfig("tilted",  np.array([0.5, 0.0, 0.5]),
                                    normal=np.array([0.0, 0.3, 1.0]))),
    ("curved",         SurfaceConfig("curved",  np.array([0.5, 0.0, 0.5]),
                                    params={"amplitude": 0.05, "frequency": 2*np.pi})),
    ("bumpy",          SurfaceConfig("bumpy",   np.array([0.5, 0.0, 0.5]),
                                    params={"n_bumps": 5, "bump_amplitude": 0.04})),
    ("tilted+curved",  SurfaceConfig("curved",  np.array([0.5, 0.0, 0.6]),
                                    params={"amplitude": 0.04, "frequency": 3*np.pi})),
]

SOURCE_CFG = SurfaceConfig("flat", np.array([0.5, 0.0, 0.5]))


def run_all(args):
    set_global_seed(args.seed)
    n_pts = 100 if args.fast else 400
    n_ind = 20 if args.fast else args.n_inducing
    n_demo = 50 if args.fast else 100
    n_iter = 50 if args.fast else 300
    n_steps = 60 if args.fast else 150

    results = []
    coverages = []
    for label, tgt_cfg in SURFACE_VARIANTS:
        print(f"  Running pipeline: {label} ...", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = run_cleaning_pipeline(
                SOURCE_CFG, tgt_cfg,
                n_source_pts=n_pts, n_target_pts=n_pts,
                n_inducing=n_ind, n_demo_pts=n_demo,
                seed=args.seed, n_iter=n_iter, n_steps=n_steps,
                gp_n_iter=80,
            )
        results.append((label, tgt_cfg, r))
        coverages.append(r["coverage"])
        print(f"    coverage={r['coverage']:.3f}  "
              f"mean_dist={r['mean_surface_dist']:.4f}m")

    return results, coverages


def make_figure(results, args, out_path):
    n = len(results)
    fig, axes = plt.subplots(
        2, n, figsize=(4 * n, 8),
        subplot_kw={"projection": "3d"},
    )

    for col, (label, tgt_cfg, r) in enumerate(results):
        # Top row: rollout + transported demo
        ax_top = axes[0, col]
        rx = r["rollout_x"]
        tx = r["transported_x"]
        T = r["T"]

        ax_top.scatter(T[:, 0], T[:, 1], T[:, 2],
                       c="orange", s=3, alpha=0.4, label="Target cloud")
        ax_top.plot(tx[:, 0], tx[:, 1], tx[:, 2],
                    "b--", linewidth=1.0, alpha=0.7, label="Transported demo")
        ax_top.plot(rx[:, 0], rx[:, 1], rx[:, 2],
                    "r-", linewidth=1.5, label="Rollout")
        ax_top.set_title(label, fontsize=9)
        if col == 0:
            ax_top.set_ylabel("Rollout", fontsize=8)
        ax_top.tick_params(labelsize=6)

        # Bottom row: point clouds only
        ax_bot = axes[1, col]
        S = r["S"]
        ax_bot.scatter(S[:, 0], S[:, 1], S[:, 2],
                       c="steelblue", s=3, alpha=0.5, label="Source")
        ax_bot.scatter(T[:, 0], T[:, 1], T[:, 2],
                       c="orange", s=3, alpha=0.5, label="Target")
        ax_bot.set_title(f"cov={r['coverage']:.2f}", fontsize=8)
        if col == 0:
            ax_bot.set_ylabel("Point clouds", fontsize=8)
        ax_bot.tick_params(labelsize=6)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.suptitle(
        f"Phase 10 — Sec. VI-C analog: Surface cleaning generalization\n"
        f"seed={args.seed}  n_inducing={args.n_inducing}  {ts}",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_inducing", type=int, default=100)
    parser.add_argument("--out_dir", type=str,
                        default=str(_REPO_ROOT / "reports" / "figures"))
    parser.add_argument("--fast", action="store_true",
                        help="Use n_pts=100, n_ind=20 for quick iteration")
    args = parser.parse_args()

    print("Running Phase 10 Figure 15 — cleaning surface generalization")
    results, coverages = run_all(args)

    print("\nCoverage fractions:")
    for (label, _, r), cov in zip(results, coverages):
        print(f"  {label:20s}: {cov:.3f}")

    out_path = Path(args.out_dir) / "phase10_fig15_cleaning.png"
    make_figure(results, args, out_path)


if __name__ == "__main__":
    main()
