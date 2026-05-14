"""Validate the IK solver over a grid of target positions — Phase 13.

Produces:
  reports/figures/phase13_ik_validation.png  — 4×5 grid of IK reach attempts
  reports/figures/phase13_workspace.png      — 3D scatter of reachable workspace
  reports/results/phase13_ik_validation.csv  — per-target results

CLI args
--------
--seed        int  (default 0)
--out_dir     path (default reports/figures/)
--n_targets   int  (default 20, arranged in a 4×5 grid)
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- project imports --------------------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from gpt_repro.utils.seeding import set_global_seed as set_all_seeds
from gpt_repro.envs.franka_env import FrankaKinematicEnv

# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--out_dir",   type=str, default="reports/figures/")
    p.add_argument("--n_targets", type=int, default=20)
    return p.parse_args()


def make_grid_targets(n: int, seed: int) -> np.ndarray:
    """Return (n, 3) targets uniformly sampled in the arm workspace."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.28, 0.70, n)
    ys = rng.uniform(-0.35, 0.35, n)
    zs = rng.uniform(0.45, 0.90, n)
    return np.column_stack([xs, ys, zs])


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir = pathlib.Path("reports/results")
    res_dir.mkdir(parents=True, exist_ok=True)

    n = args.n_targets
    targets = make_grid_targets(n, args.seed)

    # Build env and run IK on each target
    env = FrankaKinematicEnv("reshelving", render_mode="rgb_array")
    env.reset(seed=args.seed)

    results = []
    print(f"Testing IK on {n} targets …")
    for i, tgt in enumerate(targets):
        t0 = time.perf_counter()
        env.reset(seed=args.seed)
        ok = env.set_ee_pos(tgt)
        elapsed = time.perf_counter() - t0
        ee = env.get_ee_pos()
        err = float(np.linalg.norm(ee - tgt))
        results.append({"target": tgt, "ee": ee, "success": ok, "err": err, "time_s": elapsed})
        status = "✓" if ok else "✗"
        print(f"  [{i+1:3d}/{n}] {status} target={np.round(tgt, 3)}  err={err:.4f}m")

    # Summary
    successes = [r["success"] for r in results]
    errors    = [r["err"]     for r in results]
    rate      = np.mean(successes) * 100
    mean_err  = np.mean(errors)
    print(f"\nIK success rate: {rate:.1f}%   mean pos error: {mean_err*1000:.2f} mm")

    # --- Figure 1: 4×5 grid render + IK status ---
    nrows, ncols = 4, 5
    assert nrows * ncols >= n, f"Grid too small for {n} targets"
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 14))
    fig.suptitle(
        f"Franka IK Validation — {n} targets\n"
        f"Success rate: {rate:.1f}%   Mean error: {mean_err*1000:.2f} mm",
        fontsize=14,
    )
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        ax.axis("off")
        if idx >= n:
            continue
        r = results[idx]
        env.reset(seed=args.seed)
        env.set_ee_pos(r["target"])
        frame = env.render()
        ax.imshow(frame)
        color = "#00cc44" if r["success"] else "#cc2200"
        ax.set_title(
            f"{'OK' if r['success'] else 'FAIL'}  err={r['err']*1000:.1f}mm\n"
            f"z={r['target'][2]:.2f}  y={r['target'][1]:.2f}",
            fontsize=7, color=color,
        )
    plt.tight_layout()
    fig_path = out_dir / "phase13_ik_validation.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # --- Figure 2: 3D workspace scatter ---
    fig3d = plt.figure(figsize=(10, 8))
    ax3d  = fig3d.add_subplot(111, projection="3d")
    for r in results:
        c = "#00cc44" if r["success"] else "#cc2200"
        m = "o"       if r["success"] else "x"
        ax3d.scatter(*r["target"], color=c, marker=m, s=50)
        if r["success"]:
            ax3d.scatter(*r["ee"], color="blue", marker=".", s=20, alpha=0.5)
    ax3d.set_xlabel("x (m)"); ax3d.set_ylabel("y (m)"); ax3d.set_zlabel("z (m)")
    ax3d.set_title(
        f"Franka workspace validation\n"
        f"green=IK success, red=fail, blue dots=actual EE positions"
    )
    ws_path = out_dir / "phase13_workspace.png"
    fig3d.savefig(ws_path, dpi=150, bbox_inches="tight")
    plt.close(fig3d)
    print(f"Saved: {ws_path}")

    # --- Save CSV ---
    import csv
    csv_path = res_dir / "phase13_ik_validation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tx", "ty", "tz", "ex", "ey", "ez", "success", "err_m", "time_s"])
        for r in results:
            w.writerow([
                *r["target"], *r["ee"],
                int(r["success"]), r["err"], r["time_s"],
            ])
    print(f"Saved: {csv_path}")

    env.close()

    # Check pass/fail threshold
    if rate < 85.0:
        print(f"WARNING: IK success rate {rate:.1f}% is below 85% threshold.")
    if mean_err > 0.005:
        print(f"WARNING: Mean IK error {mean_err:.4f}m exceeds 5mm threshold.")


if __name__ == "__main__":
    main()
