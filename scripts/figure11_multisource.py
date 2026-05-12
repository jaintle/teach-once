"""Phase 8 — Fig. 11: multi-source single-target (Sec. V-C).

Top row (1 × 3 qualitative panels):
  A — Sources: all K source demo trajectories + anchor crosses S_k.
  B — Transported (Multi-source GPT): target anchors T, individually
      transported demos, combined DS rollout, ±2σ uncertainty band.
  C — Transported (Single-source GPT): same but only 1 source.

Bottom row (1 × 3 quantitative boxplots):
  Three metrics (Fréchet, Final pos. err, Final orient. err) with
  U-test rank annotated above each box.

Metadata footer: section=V-C  seed=<seed>  n_sources=K  timestamp=<iso>.

CLI flags: --seed, --n_reps, --n_sources, --n_steps, --out_dir.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from gpt_repro.metrics.utest import mann_whitney_ranking          # noqa: E402
from gpt_repro.policies.multisource_demos import make_multisource_scenario  # noqa: E402
from gpt_repro.utils import set_global_seed                       # noqa: E402

METHOD_NAMES = ["MultiSourceGPT", "MultiSourceDMP", "SingleSourceGPT"]
METRIC_NAMES = ["frechet", "final_pos", "final_orient"]
METRIC_LABELS = {
    "frechet":      "Fréchet",
    "final_pos":    "Final pos. err",
    "final_orient": "Final orient. err (rad)",
}
SOURCE_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple",
                 "tab:brown", "tab:pink", "tab:gray", "tab:olive"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    if pred.shape[0] < 2:
        pred = np.vstack([pred, pred])
    if gt.shape[0] < 2:
        gt = np.vstack([gt, gt])
    return {
        "frechet":      frechet_distance(pred, gt),
        "final_pos":    final_position_error(pred, gt),
        "final_orient": final_orientation_error(pred, gt),
    }


def _uncertainty_band(ax, traj: np.ndarray, std_vals: np.ndarray,
                      color: str = "orange", alpha: float = 0.30,
                      n_sigma: float = 2.0) -> None:
    """Draw a ±n_sigma band perpendicular to a 2D polyline."""
    if traj.shape[0] < 2:
        return
    # Tangent vectors
    tangents = np.diff(traj, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    tangents = tangents / norms
    # Perpendicular (rotate 90°)
    perp = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    # Use interior points (exclude endpoints for diff length alignment)
    pts = traj[:-1]
    sigma = std_vals[:-1] if len(std_vals) > 1 else std_vals * np.ones(len(pts))
    sigma = sigma[:len(pts)]
    upper = pts + n_sigma * sigma[:, None] * perp
    lower = pts - n_sigma * sigma[:, None] * perp
    poly_x = np.concatenate([upper[:, 0], lower[::-1, 0]])
    poly_y = np.concatenate([upper[:, 1], lower[::-1, 1]])
    ax.fill(poly_x, poly_y, color=color, alpha=alpha, linewidth=0)


# ---------------------------------------------------------------------------
# Per-rep benchmark helper
# ---------------------------------------------------------------------------
def _run_one_rep(seed: int, n_sources: int, n_steps: int, n_points: int,
                 n_iter_transport: int, n_iter_ds: int):
    """Return (methods_trajs, metrics, scenario) for one seed."""
    set_global_seed(seed)
    scenario = make_multisource_scenario(
        n_sources=n_sources, seed=seed, n_points=n_points,
    )
    S_list  = scenario["S_list"]
    T       = scenario["T"]
    demos   = scenario["source_demos"]
    gt_demo = scenario["target_demo"]
    gt_traj = gt_demo["x"]
    x0      = gt_traj[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        msg = MultiSourceGPT(
            n_iter_transport=n_iter_transport, n_iter_ds=n_iter_ds,
        ).fit(S_list, T, demos)
        traj_msg, _ = msg.rollout(x0, dt=0.05, n_steps=n_steps)

        mdmp = MultiSourceDMP(n_iter_gp=n_iter_ds).fit(S_list, T, demos)
        traj_mdmp, _ = mdmp.rollout(x0, dt=0.05, n_steps=n_steps)

        sgpt = GPTBaseline(
            n_iter_transport=n_iter_transport, n_iter_ds=n_iter_ds,
        ).fit(S_list[0], T, demos[0]["x"], demos[0]["xdot"])
        traj_sgpt, _ = sgpt.rollout(x0, n_steps=n_steps)

    # Uncertainty field for MultiSourceGPT (along rollout)
    xdot_rollout = np.diff(traj_msg, axis=0, prepend=traj_msg[:1])
    unc_std = msg.uncertainty(traj_msg, xdot_rollout)

    # Individually transported demos (for panel B)
    transported_indiv = []
    for transport in msg.transports:
        x_hat = transport.transform(demos[0]["x"])
        transported_indiv.append(x_hat)

    metrics = {
        "MultiSourceGPT":  _compute_metrics(traj_msg,  gt_traj),
        "MultiSourceDMP":  _compute_metrics(traj_mdmp, gt_traj),
        "SingleSourceGPT": _compute_metrics(traj_sgpt, gt_traj),
    }
    return {
        "scenario": scenario,
        "traj_msg":   traj_msg,
        "traj_mdmp":  traj_mdmp,
        "traj_sgpt":  traj_sgpt,
        "gt_traj":    gt_traj,
        "unc_std":    unc_std,
        "transported_indiv": transported_indiv,
        "msg_obj":    msg,
        "sgpt_obj":   sgpt,
        "metrics":    metrics,
    }


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------
def make_figure(
    seed: int = 0,
    n_reps: int = 10,
    n_sources: int = 4,
    n_steps: int = 100,
    n_points: int = 60,
    n_iter_transport: int = 150,
    n_iter_ds: int = 100,
    out_dir: Optional[str] = None,
    save: bool = True,
) -> Dict:
    """Build Fig. 11 and run the quantitative benchmark."""
    if out_dir is None:
        out_dir = str(_REPO_ROOT / "reports" / "figures")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results_dir = _REPO_ROOT / "reports" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Qualitative panel from seed ---
    rep0 = _run_one_rep(seed, n_sources, n_steps, n_points,
                        n_iter_transport, n_iter_ds)
    scenario = rep0["scenario"]
    S_list    = scenario["S_list"]
    T_arr     = scenario["T"]
    demos     = scenario["source_demos"]
    gt_traj   = rep0["gt_traj"]
    traj_msg  = rep0["traj_msg"]
    traj_sgpt = rep0["traj_sgpt"]
    unc_std   = rep0["unc_std"]

    # Also transport demos individually for panel B
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        msg_qual = rep0["msg_obj"]
        transported = [
            transport.transform(demos[k]["x"])
            for k, transport in enumerate(msg_qual.transports)
        ]
        # SingleSourceGPT: transport demo_0 only
        sgpt_transported = rep0["sgpt_obj"].transport.transform(demos[0]["x"])

    # --- Quantitative: collect metrics over n_reps ---
    all_metrics: Dict[str, Dict[str, List[float]]] = {
        m: {met: [] for met in METRIC_NAMES} for m in METHOD_NAMES
    }
    for rep in range(n_reps):
        rep_data = _run_one_rep(seed + rep, n_sources, n_steps, n_points,
                                n_iter_transport, n_iter_ds)
        for method in METHOD_NAMES:
            for metric in METRIC_NAMES:
                all_metrics[method][metric].append(
                    rep_data["metrics"][method][metric]
                )
        print(f"  bench rep {rep:3d}/{n_reps} done")

    # U-test ranks per metric
    ranks_per_metric: Dict[str, Dict[str, int]] = {}
    for metric in METRIC_NAMES:
        _, rank = mann_whitney_ranking(
            {m: all_metrics[m][metric] for m in METHOD_NAMES}
        )
        ranks_per_metric[metric] = rank

    # --- Build figure ---
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    method_colors = {
        "MultiSourceGPT":  "tab:red",
        "MultiSourceDMP":  "tab:blue",
        "SingleSourceGPT": "tab:green",
    }
    method_labels = {
        "MultiSourceGPT":  "Multi-source GPT",
        "MultiSourceDMP":  "Multi-source DMP",
        "SingleSourceGPT": "Single-source GPT",
    }

    # ----- Panel A: Sources -----
    ax_a = fig.add_subplot(gs[0, 0])
    for k, (demo, S_k) in enumerate(zip(demos, S_list)):
        col = SOURCE_COLORS[k % len(SOURCE_COLORS)]
        ax_a.plot(demo["x"][:, 0], demo["x"][:, 1],
                  color=col, lw=1.2, alpha=0.85, label=f"Source {k+1}")
        ax_a.scatter(S_k[:, 0], S_k[:, 1], color=col, s=12, marker=".", zorder=5)
    ax_a.plot(gt_traj[:, 0], gt_traj[:, 1], "k--", lw=1.0, alpha=0.5,
              label="GT target")
    ax_a.scatter(T_arr[:, 0], T_arr[:, 1], color="black", s=12, marker=".", zorder=5)
    ax_a.set_title("(A) Sources")
    ax_a.legend(fontsize=6, loc="best")
    ax_a.set_aspect("equal", adjustable="datalim")
    ax_a.set_xlabel("x"); ax_a.set_ylabel("y")

    # ----- Panel B: Multi-source GPT -----
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.scatter(T_arr[:, 0], T_arr[:, 1], color="black", s=16, zorder=5,
                 label="Target anchors")
    ax_b.plot(gt_traj[:, 0], gt_traj[:, 1], "k--", lw=1.0, alpha=0.5,
              label="GT target")
    for k, x_hat in enumerate(transported):
        col = SOURCE_COLORS[k % len(SOURCE_COLORS)]
        ax_b.plot(x_hat[:, 0], x_hat[:, 1], color=col, lw=0.8, alpha=0.6)
    # Uncertainty band on GPT rollout
    _uncertainty_band(ax_b, traj_msg, unc_std, color="orange", alpha=0.35)
    ax_b.plot(traj_msg[:, 0], traj_msg[:, 1], color="tab:red", lw=2.0,
              label="Multi-source GPT rollout")
    ax_b.set_title("(B) Transported — Multi-source GPT")
    ax_b.legend(fontsize=6, loc="best")
    ax_b.set_aspect("equal", adjustable="datalim")
    ax_b.set_xlabel("x"); ax_b.set_ylabel("y")

    # ----- Panel C: Single-source GPT -----
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.scatter(T_arr[:, 0], T_arr[:, 1], color="black", s=16, zorder=5,
                 label="Target anchors")
    ax_c.plot(gt_traj[:, 0], gt_traj[:, 1], "k--", lw=1.0, alpha=0.5,
              label="GT target")
    ax_c.plot(sgpt_transported[:, 0], sgpt_transported[:, 1],
              color=SOURCE_COLORS[0], lw=0.8, alpha=0.6, label="Transported demo 0")
    ax_c.plot(traj_sgpt[:, 0], traj_sgpt[:, 1], color="tab:green", lw=2.0,
              label="Single-source GPT rollout")
    ax_c.set_title("(C) Transported — Single-source GPT")
    ax_c.legend(fontsize=6, loc="best")
    ax_c.set_aspect("equal", adjustable="datalim")
    ax_c.set_xlabel("x"); ax_c.set_ylabel("y")

    # ----- Bottom row: boxplots -----
    for col_idx, metric in enumerate(METRIC_NAMES):
        ax = fig.add_subplot(gs[1, col_idx])
        data = [np.array(all_metrics[m][metric]) for m in METHOD_NAMES]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        for patch, method in zip(bp["boxes"], METHOD_NAMES):
            patch.set_facecolor(method_colors[method])
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(METHOD_NAMES) + 1))
        ax.set_xticklabels(
            [method_labels[m] for m in METHOD_NAMES],
            fontsize=7, rotation=15, ha="right",
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylabel(METRIC_LABELS[metric])
        # Annotate U-test ranks
        rank_dict = ranks_per_metric[metric]
        for i, method in enumerate(METHOD_NAMES, start=1):
            rank = rank_dict.get(method, "?")
            ax.text(i, ax.get_ylim()[1] * 0.95, f"rank {rank}",
                    ha="center", va="top", fontsize=6)

    # Footer
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    footer = (f"section=V-C  seed={seed}  n_sources={n_sources}  "
              f"n_reps={n_reps}  timestamp={timestamp}")
    fig.text(0.5, 0.01, footer, ha="center", fontsize=7, color="gray")
    fig.suptitle("Fig. 11 — Multi-source single-target (Sec. V-C)", fontsize=12)

    if save:
        png_path = Path(out_dir) / "phase8_fig11_multisource.png"
        pdf_path = Path(out_dir) / "phase8_fig11_multisource.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved {png_path}")
        print(f"Saved {pdf_path}")

    # Save JSON summary
    summary: Dict = {}
    for method in METHOD_NAMES:
        summary[method] = {}
        for metric in METRIC_NAMES:
            vals = np.array(all_metrics[method][metric])
            summary[method][metric] = {
                "mean": round(float(vals.mean()), 4),
                "std":  round(float(vals.std()), 4),
            }
    msg_f = summary["MultiSourceGPT"]["frechet"]["mean"]
    sgpt_f = summary["SingleSourceGPT"]["frechet"]["mean"]
    json_data = {
        "summary": summary,
        "MultiSourceGPT_beats_SingleGPT_frechet": msg_f <= sgpt_f,
        "msg_frechet_mean":  round(msg_f, 4),
        "sgpt_frechet_mean": round(sgpt_f, 4),
        "seed": seed,
        "n_reps": n_reps,
        "n_sources": n_sources,
        "timestamp": timestamp,
    }
    json_path = results_dir / "phase8_multisource.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved {json_path}")

    plt.close(fig)
    return json_data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Fig. 11 — Sec. V-C")
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--n_reps",    type=int, default=10)
    parser.add_argument("--n_sources", type=int, default=4)
    parser.add_argument("--n_steps",   type=int, default=100)
    parser.add_argument("--out_dir",   type=str,
                        default=str(_REPO_ROOT / "reports" / "figures"))
    args = parser.parse_args()

    results = make_figure(
        seed=args.seed,
        n_reps=args.n_reps,
        n_sources=args.n_sources,
        n_steps=args.n_steps,
        out_dir=args.out_dir,
        save=True,
    )
    print("\nSummary per method (Fréchet):")
    for m in METHOD_NAMES:
        mean = results["summary"][m]["frechet"]["mean"]
        std  = results["summary"][m]["frechet"]["std"]
        print(f"  {m:<22}: {mean:.4f} ± {std:.4f}")
    claim = results["MultiSourceGPT_beats_SingleGPT_frechet"]
    print(f"\nMultiSourceGPT beats SingleSourceGPT on Fréchet: {claim}")


if __name__ == "__main__":
    main()
