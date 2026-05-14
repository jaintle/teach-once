"""Tune impedance gains (K_s diagonal) per task — Phase 16.

Grid search over K_s diagonal values for each task.
Evaluates 100 steps of impedance rollout with a fixed target.
Prints best K_s per task based on final tracking error.

Saves: reports/results/phase16_tuned_gains.json
"""

import sys
import json
import argparse
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from gpt_repro.envs.franka_impedance_env import FrankaImpedanceEnv
from gpt_repro.utils.seeding import set_global_seed as set_all_seeds

RESULTS_DIR = pathlib.Path(__file__).parent.parent / "reports" / "results"

# K_s diagonal grid to search
K_GRID = [100.0, 150.0, 200.0, 300.0, 400.0]

# Per-task default target offsets from home EE
TASK_OFFSETS = {
    "reshelving": np.array([0.08, 0.05, 0.04]),
    "cleaning":   np.array([0.05, 0.08, -0.03]),
    "armpose":    np.array([0.06, 0.00, 0.06]),
}


def evaluate_k(task: str, k_val: float, seed: int = 0, n_steps: int = 100) -> float:
    """Run n_steps of impedance control with K_s = k_val * I.
    Returns final tracking error in metres.
    """
    env = FrankaImpedanceEnv(task=task, render_mode=None, dt=0.002, control_hz=500)
    obs, _ = env.reset(seed=seed)
    x_home = obs[:3].copy()
    x_des = x_home + TASK_OFFSETS[task]

    K_s = np.diag([k_val, k_val, k_val])
    diag_k = np.array([k_val, k_val, k_val])

    x_cur = x_home.copy()
    for _ in range(n_steps):
        action = np.concatenate([x_des, np.zeros(3), diag_k])
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            x_cur = x_home  # diverged
            break
        x_cur = obs[:3].copy()

    env.close()
    return float(np.linalg.norm(x_cur - x_des))


def tune_task(task: str, seed: int = 0) -> dict:
    """Grid search K_s for a single task.  Returns best K and error."""
    print(f"\n  Task: {task}")
    best_k = K_GRID[0]
    best_err = np.inf
    rows = []

    for k_val in K_GRID:
        err = evaluate_k(task, k_val, seed=seed, n_steps=100)
        rows.append({"k": k_val, "error": err})
        print(f"    K={k_val:5.0f}: final_err={err:.4f} m")
        if err < best_err:
            best_err = err
            best_k = k_val

    print(f"  Best K_s = {best_k}·I  (error={best_err:.4f} m)")
    return {"best_k": best_k, "best_error": best_err, "grid": rows}


def main(seed: int = 0) -> dict:
    set_all_seeds(seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 16 — Impedance Gain Tuning")
    print(f"  K grid: {K_GRID}")
    print(f"  100 steps per evaluation, seed={seed}")
    print("=" * 60)

    results = {}
    for task in ["reshelving", "cleaning", "armpose"]:
        results[task] = tune_task(task, seed=seed)

    print("\n" + "=" * 60)
    print("Summary:")
    for task, r in results.items():
        print(f"  {task:12s}: best K_s = {r['best_k']:.0f}·I  err={r['best_error']:.4f} m")
    print("=" * 60)

    out = {"seed": seed, "k_grid": K_GRID, "tasks": results}
    path = RESULTS_DIR / "phase16_tuned_gains.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(seed=args.seed)
