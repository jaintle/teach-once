"""Phase 7 — Fig. 8: Multi-frame qualitative comparison (HMM, TP-GMM, DMP, GPT).

2 × 4 panel grid. Top row: training-set frame configs (known during fit).
Bottom row: random unseen test-set frame configs. For each: rollout of
each method + dashed black ground-truth demonstration + blue stars at
the tracked frame points (5 pts per frame).

CLI flags: ``--seed``, ``--out_dir``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import warnings
from pathlib import Path

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
from gpt_repro.policies import (  # noqa: E402
    FrameConfig, get_frame_points, make_9_frame_configs, make_canonical_demo,
    make_multiframe_demo,
)
from gpt_repro.transport import transportation_velocity_variance  # noqa: E402
from gpt_repro.utils import set_global_seed  # noqa: E402

METHODS_ORDER = ["HMM", "TP-GMM", "DMP", "GPT"]


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


def _plot_panel(ax, name: str, cfg: FrameConfig, traj: np.ndarray,
                gt: np.ndarray, S: np.ndarray, T: np.ndarray,
                gpt_band: np.ndarray = None):
    ax.plot(gt[:, 0], gt[:, 1], "k--", lw=1.2, label="ground truth")
    ax.plot(traj[:, 0], traj[:, 1], "-", color="tab:red", lw=1.6, label=name)
    if gpt_band is not None:
        # ±2σ band perpendicular to trajectory.
        tang = np.diff(traj, axis=0)
        tang = np.concatenate([tang, tang[-1:]], axis=0)
        tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
        normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
        w = 2.0 * gpt_band[:, None]
        upper = traj + normal * w
        lower = traj - normal * w
        poly_x = np.concatenate([upper[:, 0], lower[::-1, 0]])
        poly_y = np.concatenate([upper[:, 1], lower[::-1, 1]])
        ax.fill(poly_x, poly_y, color="tab:orange", alpha=0.25, lw=0.0)
    ax.scatter(T[:, 0], T[:, 1], marker="*", color="tab:blue", s=60,
               zorder=4, label="frame points")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_title(name, fontsize=9)


def make_figure(seed: int = 0, n_steps: int = 80,
                out_dir: Path = _REPO_ROOT / "reports" / "figures",
                save: bool = True,
                n_iter_transport: int = 200, n_iter_ds: int = 120,
                n_iter_dmp_gp: int = 120, hmm_n_iter: int = 30) -> dict:
    set_global_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    cfgs = make_9_frame_configs(seed=seed)
    demos = [make_multiframe_demo(c, n_points=60, seed=seed + i)
             for i, c in enumerate(cfgs)]

    # TP-GMM / HMM trained on all 9 configs (Fig. 8 uses all available).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tp = TPGMMBaseline(n_components=5, random_state=seed).fit(demos, cfgs)
        hm = HMMBaseline(n_states=5, random_state=seed, n_iter=hmm_n_iter).fit(demos, cfgs)

    # GPT / DMP — train on the first config's canonical demo.
    train_cfg = cfgs[0]
    canon = make_canonical_demo(train_cfg, n_points=60, seed=seed)

    # Top row — training-set config (cfgs[0]).
    # Bottom row — random test config.
    train_target = cfgs[0]
    test_target = _random_test_config(rng)
    test_gt = make_multiframe_demo(test_target, n_points=60,
                                   seed=seed + 999)["x"]

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.0))
    for col, name in enumerate(METHODS_ORDER):
        for row, cfg in enumerate([train_target, test_target]):
            S, T = get_frame_points(cfg)
            if name == "HMM":
                method = hm
                traj = _rollout(method, cfg, n_steps)
                gpt_band = None
            elif name == "TP-GMM":
                method = tp
                traj = _rollout(method, cfg, n_steps)
                gpt_band = None
            elif name == "DMP":
                method = DMPBaseline(n_iter_gp=n_iter_dmp_gp).fit(
                    S, T, canon["x"], canon["xdot"],
                )
                traj = _rollout(method, cfg, n_steps)
                gpt_band = None
            else:  # GPT
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    method = GPTBaseline(
                        n_iter_transport=n_iter_transport, n_iter_ds=n_iter_ds,
                    ).fit(S, T, canon["x"], canon["xdot"])
                traj = _rollout(method, cfg, n_steps)
                # Σ_x̂ along the *source* canonical demo, then scalar.
                Sigma = transportation_velocity_variance(
                    method.transport, canon["x"], canon["xdot"],
                )
                # Resample to length of traj.
                scalar = np.sqrt(Sigma.sum(axis=1))
                if len(scalar) != len(traj):
                    s_old = np.linspace(0, 1, len(scalar))
                    s_new = np.linspace(0, 1, len(traj))
                    scalar = np.interp(s_new, s_old, scalar)
                gpt_band = scalar

            gt = (demos[0]["x"] if row == 0 else test_gt)
            _plot_panel(axes[row, col], name, cfg, traj, gt, S, T,
                        gpt_band=gpt_band)
            if col == 0:
                row_label = "training set" if row == 0 else "test set"
                axes[row, col].set_ylabel(row_label)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9)
    fig.suptitle(
        "Phase 7 — Sec. V-B: Multi-frame qualitative comparison",
        y=0.99,
    )
    fig.text(
        0.5, 0.005,
        f"section=V-B  seed={seed}  "
        f"timestamp={_dt.datetime.now().isoformat(timespec='seconds')}",
        ha="center", fontsize=7, color="grey",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig_path = out_dir / "phase7_fig8_qualitative.png"
    if save:
        fig.savefig(fig_path, dpi=300)
        fig.savefig(fig_path.with_suffix(".pdf"))
    plt.close(fig)
    return {"fig_path": fig_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_dir", type=Path,
        default=_REPO_ROOT / "reports" / "figures",
    )
    args = parser.parse_args()
    res = make_figure(seed=args.seed, out_dir=args.out_dir, save=True)
    print(f"Saved figure: {res['fig_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
