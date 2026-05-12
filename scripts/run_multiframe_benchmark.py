"""Phase 7 — Sec. V-B multi-reference-frame benchmark runner.

Runs ``--n_reps`` randomized comparisons between GPT, DMP, TP-GMM (m∈{5,6,7}),
and HMM (m∈{5,6,7}) on the 9 synthetic multi-frame demonstrations from
``policies/multiframe_demos.py``. Computes the five Sec. V-B metrics
(Fréchet, Area, DTW, Final-pos err, Final-orient err), aggregates results,
and produces a Mann-Whitney U-test ranking table.

CLI flags:

* ``--seed``            global RNG seed (default 0).
* ``--n_reps``          number of randomized repetitions (default 20).
* ``--n_demos_gmm``     comma-sep list of "m" values for GMM/HMM
  (default ``"5,6,7"``).
* ``--out_dir``         results directory.
"""

from __future__ import annotations

import argparse
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

from gpt_repro.baselines import (  # noqa: E402
    DMPBaseline, GPTBaseline, HMMBaseline, TPGMMBaseline,
)
from gpt_repro.metrics import (  # noqa: E402
    METRIC_FNS, METRIC_LABELS, build_ranking_table, format_ranking_ascii,
)
from gpt_repro.policies import (  # noqa: E402
    get_frame_points, make_9_frame_configs, make_canonical_demo,
    make_multiframe_demo,
)
from gpt_repro.utils import set_global_seed  # noqa: E402

METRIC_NAMES = ["frechet", "area", "dtw", "final_pos", "final_orient"]


def _build_demos(seed: int, n_points: int = 60):
    cfgs = make_9_frame_configs(seed=seed)
    demos = [make_multiframe_demo(c, n_points=n_points, seed=seed + i)
             for i, c in enumerate(cfgs)]
    return cfgs, demos


def _rollout_one_eval(method, cfg, n_steps: int) -> np.ndarray:
    """Rollout dispatch — TP-GMM / HMM take FrameConfigs, DMP / GPT take x0."""
    if isinstance(method, (TPGMMBaseline, HMMBaseline)):
        traj = method.rollout(cfg, cfg, cfg.start_pos, n_steps=n_steps)
    else:
        traj, _ = method.rollout(cfg.start_pos, n_steps=n_steps)
    return traj


def _compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    return {name: METRIC_FNS[name](pred, gt) for name in METRIC_NAMES}


def run_benchmark(
    seed: int = 0, n_reps: int = 20, m_values: List[int] = (5, 6, 7),
    n_steps: int = 80, n_points: int = 60,
    n_iter_transport: int = 150, n_iter_ds: int = 100,
    n_iter_dmp_gp: int = 100, hmm_n_iter: int = 30,
) -> Dict:
    """Run the full benchmark — returns per-method per-metric value lists."""
    method_names = ["GPT", "DMP"] + [f"TPGMM_{m}" for m in m_values] + \
                   [f"HMM_{m}" for m in m_values]
    all_results: Dict[str, Dict[str, List[float]]] = {
        name: {metric: [] for metric in METRIC_NAMES} for name in method_names
    }

    for rep in range(n_reps):
        rep_seed = seed + rep
        set_global_seed(rep_seed)
        rng = np.random.default_rng(rep_seed)
        cfgs, demos = _build_demos(rep_seed, n_points=n_points)
        n_cfgs = len(cfgs)
        # Single-demo training index for DMP/GPT.
        idx_single = int(rng.integers(0, n_cfgs))
        # Eval indices: all the other configs.
        eval_idx = [i for i in range(n_cfgs) if i != idx_single]

        # GPT and DMP — single-demo train.
        canon = make_canonical_demo(cfgs[idx_single], n_points=n_points,
                                    seed=rep_seed)
        for cfg_idx in eval_idx:
            S, T = get_frame_points(cfgs[cfg_idx])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gpt = GPTBaseline(
                    n_iter_transport=n_iter_transport, n_iter_ds=n_iter_ds,
                ).fit(S, T, canon["x"], canon["xdot"])
                dmp = DMPBaseline(n_iter_gp=n_iter_dmp_gp).fit(
                    S, T, canon["x"], canon["xdot"],
                )
                gt = demos[cfg_idx]["x"]
                pred_gpt = _rollout_one_eval(gpt, cfgs[cfg_idx], n_steps)
                pred_dmp = _rollout_one_eval(dmp, cfgs[cfg_idx], n_steps)
            for name, met in _compute_metrics(pred_gpt, gt).items():
                all_results["GPT"][name].append(met)
            for name, met in _compute_metrics(pred_dmp, gt).items():
                all_results["DMP"][name].append(met)

        # TP-GMM / HMM — m-demo train for each m.
        for m in m_values:
            train_idx = list(rng.choice(n_cfgs, size=m, replace=False))
            test_idx = [i for i in range(n_cfgs) if i not in train_idx]
            train_demos = [demos[i] for i in train_idx]
            train_cfgs  = [cfgs[i]  for i in train_idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tp = TPGMMBaseline(
                    n_components=5, random_state=rep_seed,
                ).fit(train_demos, train_cfgs)
                hm = HMMBaseline(
                    n_states=5, random_state=rep_seed, n_iter=hmm_n_iter,
                ).fit(train_demos, train_cfgs)
                for ti in test_idx:
                    gt = demos[ti]["x"]
                    pred_tp = _rollout_one_eval(tp, cfgs[ti], n_steps)
                    pred_hm = _rollout_one_eval(hm, cfgs[ti], n_steps)
                    for name, met in _compute_metrics(pred_tp, gt).items():
                        all_results[f"TPGMM_{m}"][name].append(met)
                    for name, met in _compute_metrics(pred_hm, gt).items():
                        all_results[f"HMM_{m}"][name].append(met)

    ranking = build_ranking_table(all_results, METRIC_NAMES, alpha=0.05)
    return {
        "results": all_results,
        "ranking": ranking,
        "method_names": method_names,
        "m_values": list(m_values),
    }


def _save_csv(all_results: Dict[str, Dict[str, List[float]]],
              path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = list(all_results)
    with path.open("w") as f:
        f.write("method,metric,value\n")
        for m in methods:
            for metric in METRIC_NAMES:
                for v in all_results[m][metric]:
                    f.write(f"{m},{metric},{v:.6f}\n")


def _save_rank_csv(ranking: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = ranking["method_order"]
    with path.open("w") as f:
        f.write("method," + ",".join(METRIC_NAMES) + "\n")
        for m in methods:
            ranks = [str(ranking["per_metric"][met]["rank"][m]) for met in METRIC_NAMES]
            f.write(m + "," + ",".join(ranks) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_reps", type=int, default=20)
    parser.add_argument("--n_demos_gmm", type=str, default="5,6,7")
    parser.add_argument(
        "--out_dir", type=Path,
        default=_REPO_ROOT / "reports" / "results",
    )
    args = parser.parse_args()
    m_values = tuple(int(s) for s in args.n_demos_gmm.split(","))

    res = run_benchmark(seed=args.seed, n_reps=args.n_reps, m_values=m_values)
    _save_csv(res["results"], args.out_dir / "multiframe_benchmark_results.csv")
    _save_rank_csv(res["ranking"], args.out_dir / "multiframe_ranking_table.csv")

    print("Mann-Whitney U-test ranking (lower is better; rank 1 = best):")
    print(format_ranking_ascii(
        res["ranking"], METRIC_NAMES, metric_labels=METRIC_LABELS,
    ))
    print()
    summary_path = args.out_dir / "multiframe_summary.json"
    summary_path.write_text(json.dumps({
        "n_reps": args.n_reps,
        "seed": args.seed,
        "ranks": {
            m: res["ranking"]["per_metric"][m]["rank"] for m in METRIC_NAMES
        },
    }, indent=2))
    print(f"Saved: {args.out_dir / 'multiframe_benchmark_results.csv'}")
    print(f"Saved: {args.out_dir / 'multiframe_ranking_table.csv'}")
    print(f"Saved: {summary_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
