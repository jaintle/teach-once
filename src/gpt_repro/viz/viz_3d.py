"""3-D visualisation helpers for Phase 9.

Provides :func:`plot_3d_trajectory_pair` (overlaying a demo trajectory and
its transported counterpart) and :func:`plot_generalization_trials`
(showing all trial rollouts together with a success indicator).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_3d_trajectory_pair(
    demo_x: np.ndarray,
    transported_x: np.ndarray,
    S: np.ndarray,
    T: np.ndarray,
    title: Optional[str] = None,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a demo trajectory alongside its transported counterpart in 3-D.

    Parameters
    ----------
    demo_x : (N, 3) array — original demonstration positions.
    transported_x : (N, 3) array — transported demonstration positions.
    S : (M, 3) array — source frame landmark points.
    T : (M, 3) array — target frame landmark points.
    title : str, optional — figure title.
    out_path : str or Path, optional — if provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(10, 5))

    # -- Left panel: original demo ------------------------------------------
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(demo_x[:, 0], demo_x[:, 1], demo_x[:, 2], "b-", linewidth=1.5,
             label="Demo")
    ax1.scatter(S[:, 0], S[:, 1], S[:, 2], c="red", s=30, zorder=5,
                label="Source pts")
    ax1.scatter(demo_x[0, 0], demo_x[0, 1], demo_x[0, 2], c="green", s=60,
                marker="o", zorder=6, label="Start")
    ax1.scatter(demo_x[-1, 0], demo_x[-1, 1], demo_x[-1, 2], c="black", s=60,
                marker="*", zorder=6, label="Goal")
    ax1.set_title("Source demo")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax1.legend(fontsize=7)

    # -- Right panel: transported demo -------------------------------------
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(transported_x[:, 0], transported_x[:, 1], transported_x[:, 2],
             "r-", linewidth=1.5, label="Transported")
    ax2.scatter(T[:, 0], T[:, 1], T[:, 2], c="blue", s=30, zorder=5,
                label="Target pts")
    ax2.scatter(transported_x[0, 0], transported_x[0, 1], transported_x[0, 2],
                c="green", s=60, marker="o", zorder=6, label="Start")
    ax2.scatter(transported_x[-1, 0], transported_x[-1, 1], transported_x[-1, 2],
                c="black", s=60, marker="*", zorder=6, label="Goal")
    ax2.set_title("Transported demo")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    ax2.legend(fontsize=7)

    if title is not None:
        fig.suptitle(title, fontsize=13)

    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)

    return fig


def plot_generalization_trials(
    all_rollouts: List[dict],
    goal_pos: np.ndarray,
    title: Optional[str] = None,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """Plot all trial rollouts overlaid in 3-D, coloured by success.

    Parameters
    ----------
    all_rollouts : list[dict]
        List of result dicts from :func:`~gpt_repro.transport.rollout_3d.
        transport_and_rollout_3d` (each has ``"rollout_x"`` and ``"success"``).
    goal_pos : (3,) array — goal position for reference marker.
    title : str, optional — figure title.
    out_path : str or Path, optional — save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    for res in all_rollouts:
        traj = res["rollout_x"]  # (N+1, 3)
        colour = "green" if res["success"] else "salmon"
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-",
                color=colour, linewidth=0.8, alpha=0.7)

    ax.scatter(*goal_pos, c="black", s=100, marker="*", zorder=10, label="Goal")

    # Legend proxies
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="green", lw=2, label="Success"),
        Line2D([0], [0], color="salmon", lw=2, label="Failure"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    if title is not None:
        ax.set_title(title)

    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)

    return fig
