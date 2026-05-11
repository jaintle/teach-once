"""Visualizations for 2D policy transportation (Sec. IV).

Reproduces the panel structure of Fig. 3 in Franzese et al. (2024):

1. *Distribution Match*  — paired source / target point sets.
2. *Source Distribution* — a regular grid in the source frame.
3. *Linear Transformation* — that grid mapped through γ.
4. *GP Transportation*  — same grid mapped through ϕ = γ + ψ.

Phase 3 added panels 1-3 via :func:`plot_distribution_match` and
:func:`plot_grid_under_transform`. Phase 4 fills in panel 4 by passing
``transform_fn = PolicyTransport.transform`` to the existing helper, and
adds :func:`plot_phi_scheme` — the 2×2 "transport scheme" of Fig. 5.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from gpt_repro.viz.vector_field import plot_vector_field


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


def plot_phi_scheme(
    demo_x: np.ndarray,
    demo_xdot: np.ndarray,
    S: np.ndarray,
    T: np.ndarray,
    transport,
    ax_grid: Sequence[plt.Axes],
    ds_source: Any = None,
    ds_target: Any = None,
    x_range_source: Optional[Tuple[float, float]] = None,
    y_range_source: Optional[Tuple[float, float]] = None,
    x_range_target: Optional[Tuple[float, float]] = None,
    y_range_target: Optional[Tuple[float, float]] = None,
    uncertainty_overlay: Optional[Mapping[str, Any]] = None,
) -> Sequence[plt.Axes]:
    """Fig. 5 "transport scheme" — 4 panels showing source/target demo+DS.

    Layout (matches Fig. 5 of the paper):

    * ``ax_grid[0]`` (top-left)     — source demo ``x`` overlaid on the
      source-frame vector field of ``ds_source`` (if provided).
    * ``ax_grid[1]`` (top-right)    — source distribution ``S``.
    * ``ax_grid[2]`` (bottom-left)  — transported demo ``ϕ(x)`` overlaid
      on the target-frame vector field of ``ds_target`` (if provided).
    * ``ax_grid[3]`` (bottom-right) — target distribution ``T``.

    Parameters
    ----------
    demo_x, demo_xdot : (N, 2) arrays of demonstration positions / velocities.
    S, T : (N_s, 2) source / target point clouds defining ϕ.
    transport : fitted :class:`PolicyTransport`-like object with
        ``transform`` and ``transform_velocity`` methods.
    ax_grid : sequence of 4 matplotlib axes (flattened from a 2×2 grid).
    ds_source : optional fitted source DS (must implement
        ``predict(X, return_std=True)``). Used to draw the source-frame
        vector field in panel (a).
    ds_target : optional fitted target DS. Used to draw the target-frame
        vector field in panel (c).
    x_range_source, y_range_source : optional plot extents for the
        source-frame panels. Defaults to the bounding box of the demo
        and ``S`` plus a margin.
    x_range_target, y_range_target : same for target-frame panels.
    uncertainty_overlay : optional dict — **Phase 5 hook**, currently
        unused. When provided in Phase 5 it will pass the propagated
        uncertainty fields (Eqs. 17–18) to overlay shaded confidence
        regions on the target-frame panels.

    Returns
    -------
    ax_grid : the same sequence passed in, for chaining.
    """
    if len(ax_grid) != 4:
        raise ValueError(f"plot_phi_scheme expects 4 axes; got {len(ax_grid)}")

    demo_x = np.asarray(demo_x)
    demo_xdot = np.asarray(demo_xdot)
    S = np.asarray(S)
    T = np.asarray(T)

    # Default plot extents.
    if x_range_source is None or y_range_source is None:
        pts = np.vstack([demo_x, S])
        margin = 0.3
        x_range_source = (
            float(pts[:, 0].min() - margin),
            float(pts[:, 0].max() + margin),
        )
        y_range_source = (
            float(pts[:, 1].min() - margin),
            float(pts[:, 1].max() + margin),
        )
    demo_x_hat = transport.transform(demo_x)
    if x_range_target is None or y_range_target is None:
        pts = np.vstack([demo_x_hat, T])
        margin = 0.3
        x_range_target = (
            float(pts[:, 0].min() - margin),
            float(pts[:, 0].max() + margin),
        )
        y_range_target = (
            float(pts[:, 1].min() - margin),
            float(pts[:, 1].max() + margin),
        )

    # Panel (a) — source demo + source DS field.
    ax_a = ax_grid[0]
    if ds_source is not None:
        plot_vector_field(
            ds_source,
            x_range=x_range_source, y_range=y_range_source,
            n_grid=16, ax=ax_a, cmap_by_std=True,
            demo={"x": demo_x},
        )
    else:
        ax_a.plot(demo_x[:, 0], demo_x[:, 1], "ro", ms=3)
        ax_a.set_xlim(x_range_source); ax_a.set_ylim(y_range_source)
        ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_title("(a) source demo x + DS f")

    # Panel (b) — source distribution.
    ax_b = ax_grid[1]
    ax_b.scatter(S[:, 0], S[:, 1], color="tab:blue", s=30,
                 edgecolor="white", linewidths=0.5)
    ax_b.set_xlim(x_range_source); ax_b.set_ylim(y_range_source)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_title("(b) source distribution S")
    ax_b.grid(True, alpha=0.3)

    # Panel (c) — transported demo + target DS field.
    ax_c = ax_grid[2]
    if ds_target is not None:
        plot_vector_field(
            ds_target,
            x_range=x_range_target, y_range=y_range_target,
            n_grid=16, ax=ax_c, cmap_by_std=True,
            demo={"x": demo_x_hat},
        )
    else:
        ax_c.plot(demo_x_hat[:, 0], demo_x_hat[:, 1], "ro", ms=3)
        ax_c.set_xlim(x_range_target); ax_c.set_ylim(y_range_target)
        ax_c.set_aspect("equal", adjustable="box")
    ax_c.set_title("(c) transported demo x̂ + DS f̂")

    # Panel (d) — target distribution.
    ax_d = ax_grid[3]
    ax_d.scatter(T[:, 0], T[:, 1], color="tab:red", s=30,
                 edgecolor="white", linewidths=0.5)
    ax_d.set_xlim(x_range_target); ax_d.set_ylim(y_range_target)
    ax_d.set_aspect("equal", adjustable="box")
    ax_d.set_title("(d) target distribution T")
    ax_d.grid(True, alpha=0.3)

    # TODO(phase5): wire `uncertainty_overlay` to shade the target-frame
    # panels with the propagated GP uncertainty from Eqs. (17)–(18).
    _ = uncertainty_overlay  # explicit no-op for now

    return ax_grid
