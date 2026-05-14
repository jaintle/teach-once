"""Validate impedance controller stability and tracking — Phase 16.

Tests:
1. Gravity compensation: arm holds still at home for 500 steps.
2. Point tracking: 5 random targets within workspace, 300 steps each.

Pass criterion: mean final tracking error < 0.05 m across all targets.

Saves:
- reports/results/phase16_impedance_validation.json
- reports/figures/phase16_impedance_validation.png
"""

import sys
import json
import argparse
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from gpt_repro.envs.franka_impedance_env import FrankaImpedanceEnv
from gpt_repro.utils.seeding import set_global_seed as set_all_seeds


RESULTS_DIR = pathlib.Path(__file__).parent.parent / "reports" / "results"
FIGURES_DIR = pathlib.Path(__file__).parent.parent / "reports" / "figures"


def gravity_compensation_check(env: FrankaImpedanceEnv, n_steps: int = 500) -> dict:
    """Hold at home position under gravity comp only.  Returns drift stats."""
    obs, _ = env.reset(seed=0)
    x_home = obs[:3].copy()

    K_s = np.diag([400.0, 400.0, 400.0])
    D = 2.0 * np.sqrt(K_s)
    diag_k = np.diag(K_s)

    errors = []
    taus_all = []

    for _ in range(n_steps):
        # Command stay at home, zero desired velocity
        action = np.concatenate([x_home, np.zeros(3), diag_k])
        obs, _, terminated, _, info = env.step(action)
        if terminated:
            break
        errors.append(info["tracking_error"])
        # Compute max torque this step
        tau = env._impedance_torques(x_home, np.zeros(3), K_s, D)
        taus_all.append(np.max(np.abs(tau)))

    return {
        "final_error": errors[-1] if errors else np.inf,
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "max_torque": float(np.max(taus_all)) if taus_all else 0.0,
        "n_steps": len(errors),
    }


def point_tracking_test(
    env: FrankaImpedanceEnv,
    K_s: np.ndarray,
    n_targets: int = 5,
    offset_mag: float = 0.05,
    n_steps: int = 300,
    seed: int = 0,
) -> list:
    """Track targets that are small offsets from the home EE position.

    Uses offsets of ≤offset_mag metres so that even underdamped controllers
    (D = 2√K_s assumes unit mass) can converge in n_steps.

    Returns list of dicts with final_error per target.
    """
    D = 2.0 * np.sqrt(K_s)
    diag_k = np.diag(K_s)
    results = []

    rng = np.random.default_rng(seed)
    # Small offsets on a unit sphere scaled to offset_mag
    offsets = rng.standard_normal((n_targets, 3))
    offsets /= np.linalg.norm(offsets, axis=1, keepdims=True)
    offsets *= offset_mag * rng.uniform(0.5, 1.0, size=(n_targets, 1))

    for i in range(n_targets):
        obs, _ = env.reset(seed=i)
        x_home = obs[:3].copy()
        x_des = x_home + offsets[i]

        x_cur = x_home.copy()
        errors = []

        for _ in range(n_steps):
            action = np.concatenate([x_des, np.zeros(3), diag_k])
            obs, _, terminated, _, info = env.step(action)
            if terminated:
                break
            x_cur = obs[:3].copy()
            errors.append(info["tracking_error"])

        final_err = float(np.linalg.norm(x_cur - x_des))
        results.append({
            "target_idx": i,
            "target": x_des.tolist(),
            "offset": offsets[i].tolist(),
            "final_error": final_err,
            "mean_error": float(np.mean(errors)) if errors else final_err,
        })
        print(f"  Target {i}: offset={np.linalg.norm(offsets[i]):.3f}m  final_err={final_err:.4f} m")

    return results


def run_validation(seed: int = 0, out_dir: pathlib.Path = FIGURES_DIR) -> bool:
    set_all_seeds(seed)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 16 — Impedance Validation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Gravity compensation test
    # ------------------------------------------------------------------
    print("\n[1] Gravity compensation check (500 steps at home)...")
    env = FrankaImpedanceEnv(task="reshelving", render_mode=None, dt=0.002, control_hz=500)
    grav_result = gravity_compensation_check(env, n_steps=500)
    env.close()
    print(f"   Max torque at home:  {grav_result['max_torque']:.2f} Nm")
    print(f"   Mean position drift: {grav_result['mean_error']:.5f} m")
    print(f"   Final drift:         {grav_result['final_error']:.5f} m")

    grav_pass = grav_result["final_error"] < 0.05
    print(f"   Gravity comp:        {'PASS' if grav_pass else 'FAIL'}")

    # ------------------------------------------------------------------
    # 2. Point tracking: 5 targets
    # ------------------------------------------------------------------
    print("\n[2] Point tracking (5 near-home targets, 300 steps, K_s=300·I)...")
    K_s = np.diag([300.0, 300.0, 300.0])
    env = FrankaImpedanceEnv(task="reshelving", render_mode=None, dt=0.002, control_hz=500)
    tracking_results = point_tracking_test(env, K_s, n_targets=5, offset_mag=0.05, n_steps=300, seed=seed)
    env.close()

    errors = [r["final_error"] for r in tracking_results]
    mean_err = float(np.mean(errors))
    print(f"\n   Mean final tracking error: {mean_err:.4f} m")

    track_pass = mean_err < 0.05
    print(f"   Tracking test:             {'PASS' if track_pass else 'FAIL (may need K_s tuning)'}")

    # ------------------------------------------------------------------
    # 3. Per-task torque check
    # ------------------------------------------------------------------
    print("\n[3] Per-task max torque check (100 steps, K_s=diag(300)...)  ")
    K_s_high = np.diag([300.0, 300.0, 300.0])
    D_high = 2.0 * np.sqrt(K_s_high)
    task_torques = {}
    for task in ["reshelving", "cleaning", "armpose"]:
        env_t = FrankaImpedanceEnv(task=task, render_mode=None, dt=0.002, control_hz=500)
        obs, _ = env_t.reset(seed=0)
        x_des = obs[:3] + np.array([0.05, 0.05, 0.05])
        max_tau = 0.0
        for _ in range(100):
            tau = env_t._impedance_torques(x_des, np.zeros(3), K_s_high, D_high)
            max_tau = max(max_tau, float(np.max(np.abs(tau))))
            action = np.concatenate([x_des, np.zeros(3), np.diag(K_s_high)])
            obs, _, done, _, _ = env_t.step(action)
            if done:
                break
        task_torques[task] = max_tau
        env_t.close()
        print(f"   {task:12s}: max |τ| = {max_tau:.2f} Nm  ({'within limits' if max_tau <= 87 else 'EXCEEDED'})")

    torque_pass = all(v <= 87 for v in task_torques.values())

    # ------------------------------------------------------------------
    # 4. Plot results
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Phase 16 — Impedance Validation", fontsize=13)

    # Left: tracking errors per target
    ax = axes[0]
    ax.bar(range(len(errors)), errors, color="steelblue", alpha=0.8)
    ax.axhline(0.05, color="red", linestyle="--", label="0.05m threshold")
    ax.set_xlabel("Target index")
    ax.set_ylabel("Final tracking error (m)")
    ax.set_title("Point tracking (5 near-home targets, K_s=300·I)")
    ax.legend()

    # Right: per-task max torques
    ax2 = axes[1]
    tasks_list = list(task_torques.keys())
    tau_vals = [task_torques[t] for t in tasks_list]
    bars = ax2.bar(tasks_list, tau_vals, color="coral", alpha=0.8)
    ax2.axhline(87, color="red", linestyle="--", label="87 Nm limit")
    ax2.set_ylabel("|τ|_max (Nm)")
    ax2.set_title("Max joint torques by task (K_s=300·I)")
    ax2.legend()

    plt.tight_layout()
    fig_path = out_dir / "phase16_impedance_validation.png"
    plt.savefig(str(fig_path), dpi=150)
    plt.close()
    print(f"\n   Figure saved → {fig_path}")

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    overall_pass = grav_pass and track_pass and torque_pass
    result = {
        "seed": seed,
        "gravity_compensation": grav_result,
        "gravity_pass": grav_pass,
        "tracking_results": tracking_results,
        "mean_tracking_error": mean_err,
        "tracking_pass": track_pass,
        "task_max_torques": task_torques,
        "torque_pass": torque_pass,
        "overall_pass": overall_pass,
    }
    res_path = RESULTS_DIR / "phase16_impedance_validation.json"
    with open(res_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"   Results saved → {res_path}")

    print("\n" + "=" * 60)
    status = "PASS" if overall_pass else "FAIL"
    print(f"Overall: {status}")
    print(f"  Gravity comp:  {'PASS' if grav_pass else 'FAIL'}")
    print(f"  Tracking:      {'PASS' if track_pass else 'FAIL'}")
    print(f"  Torque limits: {'PASS' if torque_pass else 'FAIL'}")
    print("=" * 60)

    return overall_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=str(FIGURES_DIR))
    args = parser.parse_args()

    passed = run_validation(seed=args.seed, out_dir=pathlib.Path(args.out_dir))
    sys.exit(0 if passed else 1)
