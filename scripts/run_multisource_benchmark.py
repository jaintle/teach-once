"""Phase 8 — Sec. V-C multi-source single-target benchmark runner.

Compares three methods:
  * MultiSourceGPT   — full ϕ_k = γ_k + ψ_k per source, pooled DS.
  * MultiSourceDMP   — linear-only γ_k per source, pooled DS.
  * SingleSourceGPT  — GPTBaseline (Phase 7) using only the first source.

Three metrics per rollout vs ground-truth target demo:
  Fréchet distance, final position error, final orientation error.

CLI flags:
  --seed      global RNG seed (default 0).
  --n_reps    number of random repetitions (default 10).
  --n_sources number of source frames per rep (default 4).
  --out_dir   results directory (default reports/results/).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gpt_repro.baselines.gpt_adapter import GPTBaseline          # noqa: E402
from gpt_repro.baselines.multisource_dmp import MultiSourceDMP   # noqa: E402
from gpt_repro.baselines.multisource_gpt import MultiSourceGPT   # noqa: E402
from gpt_repro.metrics.trajectory_metrics import (               # noqa: E402
    frechet_distance, final_position_error, final_orientation_error,
)
from gpt_repro.policies.multisource_demos import make_multisource_scenario  # noqa: E402
from gpt_repro.utils import set_global_seed                       # noqa: E402

METRIC_NAMES = ["frechet", "final_pos", "final_orient"]
METHOD_NAMES = ["MultiSourceGPT", "MultiSourceDMP", "SingleSourceGPT"]


def _compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    # Ensure both arrays have >= 2 rows for orientation
    if pred.shape[0] < 2:
        pred = np.vstack([pred, pred])
    if gt.shape[0] < 2:
        gt = np.vstack([gt, gt])
    return {
        "frechet":      frechet_distance(pred, gt),
        "final_pos":    final_position_error(pred, gt),
        "final_orient": final_orientation_error(pred, gt),
    }


def run_benchmark(
    seed: int = 0,
    n_reps: int = 10,
    n_sources: int = 4,
    n_steps: int = 100,
    n_points: int = 60,
    n_iter_transport: int = 150,
    n_iter_ds: int = 100,
) -> Dict:
    """Run the multi-source benchmark. Returns per-method per-metric value lists."""
    all_results: Dict[str, Dict[str, List[float]]] = {
        m: {metric: [] for metric in METRIC_NAMES} for m in METHOD_NAMES
    }
    rep_rows: List[dict] = []

    for rep in range(n_reps):
        rep_seed = seed + rep
        set_global_seed(rep_seed)

        scenario = make_multisource_scenario(
            n_sources=n_sources, seed=rep_seed, n_points=n_points,
        )
        S_list  = scenario["S_list"]
        T       = scenario["T"]
        demos   = scenario["source_demos"]
        gt_demo = scenario["target_demo"]
        gt_traj = gt_demo["x"]
        x0      = gt_traj[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # MultiSourceGPT
            msg = MultiSourceGPT(
                n_iter_transport=n_iter_transport, n_iter_ds=n_iter_ds,
            ).fit(S_list, T, demos)
            traj_msg, _ = msg.rollout(x0, dt=0.05, n_steps=n_steps)
            met_msg = _compute_metrics(traj_msg, gt_traj)

            # MultiSourceDMP
            mdmp = MultiSourceDMP(n_iter_gp=n_iter_ds).fit(S_list, T, demos)
            traj_mdmp, _ = mdmp.rollout(x0, dt=0.05, n_steps=n_steps)
            met_mdmp = _compute_metrics(traj_mdmp, gt_traj)

            # SingleSourceGPT (only first source)
            sgpt = GPTBaseline(
                n_iter_transport=n_iter_transport, n_iter_ds=n_iter_ds,
            ).fit(
                S_list[0], T,
                demos[0]["x"], demos[0]["xdot"],
            )
            traj_sgpt, _ = sgpt.rollout(x0, n_steps=n_steps)
            met_sgpt = _compute_metrics(traj_sgpt, gt_traj)

        for metric in METRIC_NAMES:
            all_results["MultiSourceGPT"][metric].append(met_msg[metric])
            all_results["MultiSourceDMP"][metric].append(met_mdmp[metric])
            all_results["SingleSourceGPT"][metric].append(met_sgpt[metric])
            for method, vals in [("MultiSourceGPT", met_msg),
                                  ("MultiSourceDMP", met_mdmp),
                                  ("SingleSourceGPT", met_sgpt)]:
                rep_rows.append({
                    "rep": rep, "method": method,
                    "metric": metric, "value": vals[metric],
                })

        print(f"  rep {rep:3d}/{n_reps} done")

    return all_results, rep_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-source benchmark (Sec. V-C)")
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--n_reps",    type=int, default=10)
    parser.add_argument("--n_sources", type=int, default=4)
    parser.add_argument("--out_dir",   type=str,
                        default=str(_REPO_ROOT / "reports" / "results"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running multi-source benchmark: seed={args.seed}, "
          f"n_reps={args.n_reps}, n_sources={args.n_sources}")
    all_results, rep_rows = run_benchmark(
        seed=args.seed, n_reps=args.n_reps, n_sources=args.n_sources,
    )

    # Save per-rep CSV
    csv_path = out_dir / "multisource_benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rep", "method", "metric", "value"])
        writer.writeheader()
        writer.writerows(rep_rows)
    print(f"Saved {csv_path}")

    # Print mean ± std per method per metric
    summary: Dict = {}
    print("\nResults (mean ± std):")
    header = f"{'Method':<20} " + "  ".join(f"{m:>18}" for m in METRIC_NAMES)
    print(header)
    for method in METHOD_NAMES:
        row = f"{method:<20}"
        summary[method] = {}
        for metric in METRIC_NAMES:
            vals = np.array(all_results[method][metric])
            mu, sd = float(vals.mean()), float(vals.std())
            summary[method][metric] = {"mean": round(mu, 4), "std": round(sd, 4)}
            row += f"  {mu:8.4f} ± {sd:6.4f}"
        print(row)

    # Report Sec. V-C main claim: MultiSourceGPT vs SingleSourceGPT on Fréchet
    msg_frechet = np.mean(all_results["MultiSourceGPT"]["frechet"])
    sgpt_frechet = np.mean(all_results["SingleSourceGPT"]["frechet"])
    beats = msg_frechet <= sgpt_frechet
    print(f"\nSec. V-C claim — MultiSourceGPT Fréchet ({msg_frechet:.4f}) "
          f"≤ SingleSourceGPT Fréchet ({sgpt_frechet:.4f}): {beats}")

    # Save JSON summary
    json_path = out_dir / "phase8_multisource.json"
    with open(json_path, "w") as f:
        json.dump({"summary": summary,
                   "MultiSourceGPT_beats_SingleGPT_frechet": bool(beats),
                   "msg_frechet_mean": round(float(msg_frechet), 4),
                   "sgpt_frechet_mean": round(float(sgpt_frechet), 4),
                   "seed": args.seed,
                   "n_reps": args.n_reps,
                   "n_sources": args.n_sources}, f, indent=2)
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
