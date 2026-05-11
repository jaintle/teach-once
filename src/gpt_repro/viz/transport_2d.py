"""Visualizations for 2D policy transportation (Sec. IV).

Reproduces the panel structure of Fig. 3 in Franzese et al. (2024):

1. *Distribution Match*  — paired source / target point sets.
2. *Source Distribution* — a regular grid in the source frame.
3. *Linear Transformation* — that grid mapped through γ.
4. *GP Transportation*  — same grid mapped through ϕ = γ + ψ (Phase 4).

Panels 1–3 are implemented here. Panel 4's underlying ψ is built in
Phase 4 of the reproduction; it consumes the same
:func:`plot_grid_under_transform` helper.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_distribution_match(
    S: np.ndarray,
    T: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    source_color: str = "tab:blue",
    target_color: str = "tab:red",
) -> plt.Axes:
    """Panel 1 of Fig. 3: paired source / target points with correspondences.

    Plots ``S`` (source) in one color, ``T`` (target) in another, and
    draws a thin grey line between each paired index ``(S[i], T[i])`` to
    make the correspondence explicit. This mirrors the leftmost panel
    of Fig. 3 in the paper.

    Parameters
    ----------
    S : (N, 2) source points.
    T : (N, 2) target points.
    ax : optional matplotlib Axes.
    title : optional axes title.
    """
    S = np.asarray(S)
    T = np.asarray(T)
    if S.shape != T.shape:
        raise ValueError(f"S and T must match shape: {S.shape} vs {T.shape}")
    if S.shape[1] != 2:
        raise ValueError(f"plot_distribution_match is 2D-only; got d={S.shape[1]}")

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    # Correspondence lines first (so they sit underneath the points).
    for i in range(S.shape[0]):
        ax.plot(
            [S[i, 0], T[i, 0]],
            [S[i, 1], T[i, 1]],
            color="0.6",
            lw=0.6,
            zorder=1,
        )
    ax.scatter(S[:, 0], S[:, 1], color=source_color, s=30, label="source S",
               edgecolor="white", linewidths=0.5, zorder=3)
    ax.scatter(T[:, 0], T[:, 1], color=target_color, s=30, label="target T",
               edgecolor="white", linewidths=0.5, zorder=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def plot_grid_under_transform(
    transform_fn: Callable[[np.ndarray], np.ndarray],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    n_grid: int = 15,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    overlay_points: Optional[Mapping[str, np.ndarray]] = None,
    source_color: str = "tab:blue",
    target_color: str = "tab:red",
) -> plt.Axes:
    """Panels 2-4 of Fig. 3: a regular grid before and after a transform.

    Builds a regular ``n_grid × n_grid`` mesh of points spanning
    ``[x_range] × [y_range]`` in source frame, then plots:

    * the original grid in light grey (panel 2 of Fig. 3 when used
      alone, with ``transform_fn = identity``),
    * the transformed grid in solid color (panels 3 / 4 when
      ``transform_fn`` is :meth:`LinearTransport.transform` or the
      full transportation map ϕ from Phase 4).

    The connectivity of the grid is preserved by drawing grid lines
    (horizontal and vertical) explicitly — this makes the deformation
    induced by the transport visually obvious.

    Parameters
    ----------
    transform_fn : callable taking (M, 2) → (M, 2).
    x_range, y_range : (float, float) extent of the source grid.
    n_grid : grid resolution per axis.
    ax : optional matplotlib Axes.
    title : optional axes title.
    overlay_points : optional dict mapping label → (N, 2) — typically
        ``{"source": S, "target": T}`` to overlay the paired point
        sets on top of the transformed grid.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    xs = np.linspace(x_range[0], x_range[1], n_grid)
    ys = np.linspace(y_range[0], y_range[1], n_grid)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1)
    transformed = np.asarray(transform_fn(grid)).reshape(n_grid, n_grid, 2)

    # Original grid (light grey lines + dots).
    for i in range(n_grid):
        ax.plot(XX[i, :], YY[i, :], color="0.85", lw=0.7, zorder=1)
        ax.plot(XX[:, i], YY[:, i], color="0.85", lw=0.7, zorder=1)
    ax.scatter(XX.ravel(), YY.ravel(), color="0.7", s=6, zorder=2)

    # Transformed grid (solid).
    Tx, Ty = transformed[..., 0], transformed[..., 1]
    for i in range(n_grid):
        ax.plot(Tx[i, :], Ty[i, :], color="tab:green", lw=0.9, zorder=3)
        ax.plot(Tx[:, i], Ty[:, i], color="tab:green", lw=0.9, zorder=3)
    ax.scatter(Tx.ravel(), Ty.ravel(), color="tab:green", s=8, zorder=4)

    if overlay_points is not None:
        if "source" in overlay_points:
            S = np.asarray(overlay_points["source"])
            ax.scatter(S[:, 0], S[:, 1], color=source_color, s=28,
                       edgecolor="white", linewidths=0.5,
                       label="source S", zorder=5)
        if "target" in overlay_points:
            T = np.asarray(overlay_points["target"])
            ax.scatter(T[:, 0], T[:, 1], color=target_color, s=28,
                       edgecolor="white", linewidths=0.5,
                       label="target T", zorder=5)
        if any(k in overlay_points for k in ("source", "target")):
            ax.legend(loc="best", fontsize=8)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)
    if title:
        ax.set_title(title)
    return ax
