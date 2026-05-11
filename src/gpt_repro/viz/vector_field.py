"""Vector-field visualization for a learned 2D dynamical system.

Reproduces the "field of arrows colored by epistemic uncertainty" look
of the paper's qualitative DS figures (Figs. 5, 6). Arrows show the
predicted velocity at each grid point; arrow color encodes the
predicted posterior standard deviation, so cool colors indicate
high-confidence regions near the demonstration and warm colors
indicate the GP's drift toward its zero-mean prior.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_vector_field(
    ds,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    n_grid: int = 20,
    ax: Optional[plt.Axes] = None,
    cmap_by_std: bool = True,
    demo: Optional[Mapping[str, np.ndarray]] = None,
    rollout: Optional[np.ndarray] = None,
    cmap: str = "hot",
    arrow_scale: Optional[float] = None,
    std_fn=None,
    cbar_label: Optional[str] = None,
) -> plt.Axes:
    """Plot the velocity field predicted by a fitted DS policy.

    Parameters
    ----------
    ds : object
        A fitted policy exposing ``predict(X, return_std=True) -> (mean, std)``
        with mean and std of shape (M, 2). Typically a
        :class:`gpt_repro.policies.ds_policy.GPDynamicalSystem`.
    x_range, y_range : (float, float)
        Plot extents in x and y.
    n_grid : int
        Number of grid points per axis (total arrows = ``n_grid**2``).
    ax : matplotlib Axes, optional
        Existing axes to draw on. A new figure/axes is created if None.
    cmap_by_std : bool
        If True, arrow color encodes the Euclidean norm of the per-axis
        predictive std at each grid point (paper Figs. 5/6 convention —
        "warm" colors mean high epistemic uncertainty). If False, arrows
        are drawn uniformly black.
    demo : dict, optional
        Demonstration dict (``{"x": (N,2), ...}``) — overlaid as red dots.
    rollout : (T, 2) array, optional
        DS rollout trajectory — overlaid as a black line.
    cmap : str
        Matplotlib colormap name used for the std encoding.
    arrow_scale : float, optional
        Forwarded to ``ax.quiver(scale=...)``. Auto-scaled by matplotlib
        when None.

    Returns
    -------
    ax : matplotlib Axes used for the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.0))

    xs = np.linspace(x_range[0], x_range[1], n_grid)
    ys = np.linspace(y_range[0], y_range[1], n_grid)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1)
    mean, std = ds.predict(grid, return_std=True)
    U = mean[:, 0]
    V = mean[:, 1]

    if cmap_by_std:
        # Euclidean norm of per-axis std → scalar uncertainty per arrow,
        # unless the caller provides an explicit ``std_fn`` (used by
        # Phase 5 to color arrows by Σ_total rather than ds-epistemic).
        if std_fn is not None:
            C = np.asarray(std_fn(grid)).reshape(-1)
            if C.shape[0] != grid.shape[0]:
                raise ValueError(
                    f"std_fn must return shape ({grid.shape[0]},); got {C.shape}"
                )
            default_label = "total std  ‖σ_total(x)‖"
        else:
            C = np.linalg.norm(std, axis=1)
            default_label = "predicted std  ‖σ(x)‖"
        q = ax.quiver(
            XX.ravel(), YY.ravel(), U, V, C,
            cmap=cmap, scale=arrow_scale, width=0.003,
            angles="xy", pivot="mid",
        )
        cbar = plt.colorbar(q, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label if cbar_label is not None else default_label)
    else:
        ax.quiver(
            XX.ravel(), YY.ravel(), U, V,
            color="black", scale=arrow_scale, width=0.003,
            angles="xy", pivot="mid",
        )

    if demo is not None and "x" in demo:
        pts = np.asarray(demo["x"])
        ax.plot(pts[:, 0], pts[:, 1], "o", color="red", ms=3.5,
                label="demonstration")
    if rollout is not None:
        traj = np.asarray(rollout)
        ax.plot(traj[:, 0], traj[:, 1], "-", color="black", lw=1.6,
                label="DS rollout")
        ax.plot(traj[0, 0], traj[0, 1], "s", color="black", ms=6,
                markerfacecolor="white", label="rollout start")
        ax.plot(traj[-1, 0], traj[-1, 1], "P", color="black", ms=7,
                label="rollout end")

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if demo is not None or rollout is not None:
        ax.legend(loc="best", fontsize=8)
    return ax
