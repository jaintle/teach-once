"""Phase 7 — Fig. 10: test-set boxplots (final-pos + final-orient errors).

Tests on randomly generated frame configs not seen during training.
TP-GMM_9 and HMM_9 are trained on all 9 demos; GPT and DMP use a
single randomly-chosen demo as before. Reports two metrics:
final-position and final-orientation error.

CLI flags: ``--seed``, ``--n_reps``, ``--out_dir``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gpt_repro.baselines import (  # noqa: E402
    DMPBaseline, GPTBaseline, HMMBaseline, TPGMMBaseline,
)
from gpt_repro.metrics import (  # noqa: E402
    METRIC_LABELS, build_ranking_table,
    final_orientation_error, final_position_error,
)
from gpt_repro.policies import (  # noqa: E402
    FrameConfig, get_frame_points, make_9_frame_configs, make_canonical_demo,
    make_multiframe_demo,
)
from gpt_repro.utils import set_global_seed  # noqa: E402

METHOD_ORDER = ["GPT", "DMP", "HMM_9", "TPGMM_9"]
METRIC_NAMES = ["final_pos", "final_orient"]


def _random_test_config(rng) -> FrameConfig:
    start_pos = rng.uniform(low=[-2.0, -1.5], high=[-0.5, 1.5])
    goal_pos  = rng.uniform(low=[ 0.5, -1.5], high=[ 2.0, 1.5])
    return FrameConfig(
        start_pos=start_pos,
        start_angle=float(rng.uniform(-np.pi / 3, np.pi / 3)),
        goal_pos=goal_pos,
        goal_angle=float(rng.uniform(np.pi - np.pi / 3, np.pi + np.pi / 3)),
    )


def _rollout(method, cfg, n_steps: int):
    if isinstance(method, (TPGMMBaseline, HMMBaseline)):
        return method.rollout(cfg, cfg, cfg.start_pos, n_steps=n_steps)
    traj, _ = method.rollout(cfg.start_pos, n_steps=n_steps)
    return traj


def run(seed: int, n_reps: int, n_steps: int = 80) -> Dict[str, Dict[str, List[float]]]:
    results: Dict[str, Dict[str, List[float]]] = {
        m: {met: [] for met in METRIC_NAMES} for m in METHOD_ORDER
    }
    for rep in range(n_reps):
        rep_seed = seed + rep
        set_global_seed(rep_seed)
        rng = np.random.default_rng(rep_seed)
        train_cfgs = make_9_frame_configs(seed=rep_seed)
        train_demos = [make_multiframe_demo(c, n_points=60, seed=rep_seed + i)
                       for i, c in enumerate(train_cfgs)]
        # TP-GMM / HMM trained on all 9 train configs.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tp9 = TPGMMBaseline(n_components=5, random_state=rep_seed).fit(
                train_demos, train_cfgs,
            )
            hm9 = HMMBaseline(n_states=5, random_state=rep_seed, n_iter=30).fit(
                train_demos, train_cfgs,
            )
        # GPT/DMP single-demo: pick a random train demo.
        idx = int(rng.integers(0, 9))
        canon = make_canonical_demo(train_cfgs[idx], n_points=60, seed=rep_seed)
        # Evaluate on 3 random test configs per rep.
        for _ in range(3):
            test_cfg = _random_test_config(rng)
            gt = make_multiframe_demo(test_cfg, n_points=60,
                                      seed=rep_seed * 7 + 1)["x"]
            S, T = get_frame_points(test_cfg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gpt = GPTBaseline(n_iter_transport=120, n_iter_ds=80).fit(
                    S, T, canon["x"], canon["xdot"],
                )
                dmp = DMPBaseline(n_iter_gp=80).fit(
                    S, T, canon["x"], canon["xdot"],
                )
                preds = {
                    "GPT": _rollout(gpt, test_cfg, n_steps),
                    "DMP": _rollout(dmp, test_cfg, n_steps),
                    "HMM_9":   _rollout(hm9, test_cfg, n_steps),
                    "TPGMM_9": _rollout(tp9, test_cfg, n_steps),
                }
            for m, pred in preds.items():
                results[m]["final_pos"].append(final_position_error(pred, gt))
                results[m]["final_orient"].append(final_orientation_error(pred, gt))
    return results


def make_figure(seed: int = 0, n_reps: int = 20,
                out_dir: Path = _REPO_ROOT / "reports" / "figures",
                save: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = run(seed=seed, n_reps=n_reps)
    ranking = build_ranking_table(data, METRIC_NAMES)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    for i, metric in enumerate(METRIC_NAMES):
        ax = axes[i]
        values = [data[m][metric] for m in METHOD_ORDER]
        bp = ax.boxplot(values, tick_labels=METHOD_ORDER, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("0.85")
        y_top = max(max(v) for v in values) * 1.05 if any(values) else 1.0
        for j, m in enumerate(METHOD_ORDER, start=1):
            r = ranking["per_metric"][metric]["rank"][m]
            ax.text(j, y_top, str(r), ha="center", va="bottom",
                    fontweight="bold")
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Phase 7 — Sec. V-B: Test set performance (Fig. 10)",
                 y=0.99)
    fig.text(
        0.5, 0.005,
        f"section=V-B  seed={seed}  n_reps={n_reps}  "
        f"timestamp={_dt.datetime.now().isoformat(timespec='seconds')}",
        ha="center", fontsize=7, color="grey",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig_path = out_dir / "phase7_fig10_test_boxplots.png"
    if save:
        fig.savefig(fig_path, dpi=300)
        fig.savefig(fig_path.with_suffix(".pdf"))
    plt.close(fig)
    return {"fig_path": fig_path, "results": data, "ranking": ranking}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_reps", type=int, default=20)
    parser.add_argument(
        "--out_dir", type=Path,
        default=_REPO_ROOT / "reports" / "figures",
    )
    args = parser.parse_args()
    res = make_figure(seed=args.seed, n_reps=args.n_reps, out_dir=args.out_dir)
    print(f"Saved figure: {res['fig_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
