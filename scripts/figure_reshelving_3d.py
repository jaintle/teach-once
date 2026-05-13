#!/usr/bin/env python
"""Figure: reshelving 3-D generalisation experiment — Phase 9.

Runs the reshelving analogue of the GPT generalisation experiment in 3-D:

  (a) Source and transported demo trajectory pair.
  (b) Success rate bar chart across randomised scenes.
  (c) Final error box plot across trials.

Saves to ``reports/figures/phase9_reshelving_3d.png`` (or ``--out_dir``).

Usage
-----
    python scripts/figure_reshelving_3d.py --seed 0 --n_trials 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from gpt_repro.policies.demos_3d import make_reshelving_demo, randomize_reshelving_scene
from gpt_repro.transport.rollout_3d import (
    transport_and_rollout_3d,
    evaluate_generalization_3d,
)
from gpt_repro.envs.reshelving_env import ReshelvingEnv
from gpt_repro.utils.seeding import set_global_seed
from gpt_repro.viz.viz_3d import plot_3d_trajectory_pair, plot_generalization_trials


def main(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Base demo and scene ------------------------------------------------
    demo, scene = make_reshelving_demo(seed=args.seed)

    # -- Transport base demo (identity: S == T for visualization) -----------
    env_base = ReshelvingEnv(scene=scene)
    base_result = transport_and_rollout_3d(
        demo=demo,
        S=scene["S"],
        T=scene["T"],
        env=env_base,
        gp_n_iter=args.gp_n_iter,
        n_steps=args.n_steps,
        seed=args.seed,
    )
    env_base.close()

    # -- Generalisation evaluation ------------------------------------------
    gen_result = evaluate_generalization_3d(
        base_demo=demo,
        base_scene=scene,
        randomize_fn=randomize_reshelving_scene,
        n_trials=args.n_trials,
        seed=args.seed,
        env_cls=ReshelvingEnv,
        gp_n_iter=args.gp_n_iter,
        n_steps=args.n_steps,
    )

    print(f"Reshelving success rate : {gen_result['success_rate']:.2f}")
    print(f"Mean final error        : {gen_result['mean_error']:.4f} m")

    # -- Figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel (a): trajectory pair
    ax_traj = axes[0]
    ax_traj.remove()
    fig_traj = plt.figure(figsize=(5, 4))
    ax_3d = fig_traj.add_subplot(1, 1, 1, projection="3d")
    demo_x = demo["x"]
    trans_x = base_result["transported_x"]
    ax_3d.plot(demo_x[:, 0], demo_x[:, 1], demo_x[:, 2], "b-", label="Demo")
    ax_3d.plot(trans_x[:, 0], trans_x[:, 1], trans_x[:, 2], "r--", label="Transported")
    ax_3d.set_title("(a) Trajectory")
    ax_3d.legend(fontsize=7)

    # Panel (b): success rate bar
    axes[1].bar(
        ["GPT 3D"],
        [gen_result["success_rate"]],
        color=["steelblue"],
        width=0.4,
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Success rate")
    axes[1].set_title(f"(b) Success rate (n={args.n_trials})")
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)

    # Panel (c): final error box plot
    axes[2].boxplot(
        gen_result["final_errors"],
        labels=["GPT 3D"],
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
    )
    axes[2].set_ylabel("Final error (m)")
    axes[2].set_title("(c) Final error distribution")

    fig.suptitle(f"Phase 9: Reshelving 3D generalisation (seed={args.seed})", fontsize=12)
    fig.tight_layout()

    out_path = out_dir / "phase9_reshelving_3d.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Also save the 3D trajectory panel
    traj_out = out_dir / "phase9_reshelving_traj.png"
    fig_traj.tight_layout()
    fig_traj.savefig(traj_out, dpi=150)
    plt.close(fig_traj)

    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 9 reshelving 3D figure")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--gp_n_iter", type=int, default=100)
    parser.add_argument("--n_steps", type=int, default=150)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "reports" / "figures"),
    )
    main(parser.parse_args())
