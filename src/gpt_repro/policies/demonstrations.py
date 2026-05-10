"""Synthetic 2D demonstration generators.

These are the canonical 2D toy demonstrations used throughout the paper's
qualitative figures (the letter-C trajectory of Figs. 2 / 5, and the
periodic cleaning trajectory of Fig. 7). The reference frames / surfaces
they live on are built by :func:`make_surface_2d` and are used in later
phases as the source / target manifolds for policy transportation.

This module does not implement any equation by itself — it generates the
state / velocity pairs that feed :class:`gpt_repro.policies.ds_policy.
GPDynamicalSystem`, which in turn implements **Eq. (1)** of Sec. III-A.

Velocities are computed deterministically given the seed: central
differences for the interior of the trajectory, one-sided differences at
the endpoints, followed by a single-pass length-3 moving average. This
matches the paper's stated "smoothing pass" preprocessing of human
demonstrations.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


# ---------------------------------------------------------------------------
# Velocity preprocessing
# ---------------------------------------------------------------------------
def _finite_diff_velocity(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Central finite-difference velocity with one-pass length-3 smoothing.

    Parameters
    ----------
    pos : (N, d) array of positions.
    t   : (N,)   array of strictly increasing timestamps.

    Returns
    -------
    vel : (N, d) array of velocities.
    """
    if pos.ndim != 2:
        raise ValueError(f"pos must be (N, d), got {pos.shape}")
    if t.shape != (pos.shape[0],):
        raise ValueError(f"t must be ({pos.shape[0]},), got {t.shape}")

    vel = np.zeros_like(pos)
    # Endpoints: one-sided differences.
    vel[0] = (pos[1] - pos[0]) / (t[1] - t[0])
    vel[-1] = (pos[-1] - pos[-2]) / (t[-1] - t[-2])
    # Interior: central differences using the actual (possibly non-uniform) dt.
    if pos.shape[0] > 2:
        dt_c = (t[2:] - t[:-2])[:, None]
        vel[1:-1] = (pos[2:] - pos[:-2]) / dt_c

    # Length-3 moving average smoothing, single pass, interior only.
    smoothed = vel.copy()
    smoothed[1:-1] = (vel[:-2] + vel[1:-1] + vel[2:]) / 3.0
    return smoothed


# ---------------------------------------------------------------------------
# Letter-C demonstration (Fig. 2 / Fig. 5)
# ---------------------------------------------------------------------------
def make_letter_C_demo(
    n_points: int = 100,
    noise: float = 0.0,
    seed: int = 0,
    radius: float = 1.0,
    duration: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Synthesize a 2D "letter C" demonstration.

    Open arc traversed counter-clockwise from the upper-right (θ = π/4)
    through the left (θ = π) to the lower-right (θ = 7π/4), spanning 270°
    on a circle of the given radius. This matches the C-shaped trajectory
    visualized in Figs. 2 and 5 of Franzese et al. (2024).

    Time parameterization is linear, which yields a roughly constant
    tangential speed — convenient for GP regression because the
    velocity targets are well-conditioned (no near-zero magnitudes).

    Parameters
    ----------
    n_points : int
        Number of samples along the trajectory.
    noise : float
        Std of additive isotropic Gaussian position noise (default 0).
    seed : int
        RNG seed for the noise (only matters when ``noise > 0``).
    radius : float
        Radius of the C arc.
    duration : float
        Total demonstration duration in seconds.

    Returns
    -------
    dict with keys
        ``"x"``    : (n_points, 2) positions,
        ``"xdot"`` : (n_points, 2) velocities,
        ``"t"``    : (n_points,)  timestamps in [0, duration].
    """
    if n_points < 3:
        raise ValueError("n_points must be >= 3.")
    rng = np.random.default_rng(int(seed))

    t = np.linspace(0.0, float(duration), n_points)
    theta_start, theta_end = np.pi / 4.0, 7.0 * np.pi / 4.0  # 270° sweep
    theta = theta_start + (theta_end - theta_start) * (t / duration)
    pos = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)

    if noise > 0.0:
        pos = pos + float(noise) * rng.standard_normal(pos.shape)

    vel = _finite_diff_velocity(pos, t)
    return {"x": pos, "xdot": vel, "t": t}


# ---------------------------------------------------------------------------
# Cleaning demonstration (Fig. 7)
# ---------------------------------------------------------------------------
def make_cleaning_demo(
    n_points: int = 120,
    noise: float = 0.0,
    seed: int = 0,
    n_cycles: int = 3,
    x_range: tuple = (-1.0, 1.0),
    lift_height: float = 0.4,
    surface_y: float = 0.0,
    duration: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Synthesize a 2D periodic "approach-clean-retreat" demonstration.

    The end-effector progresses linearly in x while oscillating in y
    between a lifted height and the surface — a stylized version of the
    cyclic cleaning trajectory shown in Fig. 7 of the paper. The
    trajectory descends to ``surface_y`` at the bottom of each cycle and
    rises to ``surface_y + lift_height`` at the top.

    Parameters
    ----------
    n_points : int
        Total number of samples along the trajectory.
    noise : float
        Additive isotropic Gaussian position noise std (default 0).
    seed : int
        RNG seed.
    n_cycles : int
        Number of full down-up oscillations across ``x_range``.
    x_range : (float, float)
        Sweep range in x.
    lift_height : float
        Distance lifted above the surface at the top of each cycle.
    surface_y : float
        y-coordinate of the cleaning surface (touched at each cycle min).
    duration : float
        Total demonstration duration in seconds.

    Returns
    -------
    dict with keys ``"x"``, ``"xdot"``, ``"t"`` of shapes
        (n_points, 2), (n_points, 2), (n_points,) respectively.
    """
    if n_points < 3:
        raise ValueError("n_points must be >= 3.")
    rng = np.random.default_rng(int(seed))

    t = np.linspace(0.0, float(duration), n_points)
    s = t / duration  # normalized progress in [0, 1]
    x_lo, x_hi = float(x_range[0]), float(x_range[1])
    x = x_lo + s * (x_hi - x_lo)
    # y oscillates: 0 at surface (when cos=1) to lift_height (when cos=-1).
    y = surface_y + 0.5 * lift_height * (1.0 - np.cos(2.0 * np.pi * n_cycles * s))
    pos = np.stack([x, y], axis=1)

    if noise > 0.0:
        pos = pos + float(noise) * rng.standard_normal(pos.shape)

    vel = _finite_diff_velocity(pos, t)
    return {"x": pos, "xdot": vel, "t": t}


# ---------------------------------------------------------------------------
# Surfaces for transportation (Phase 3+)
# ---------------------------------------------------------------------------
def make_surface_2d(
    kind: str = "flat",
    n_points: int = 40,
    x_range: tuple = (-1.2, 1.2),
    amplitude: float = 0.15,
) -> np.ndarray:
    """Generate a 2D point cloud sampling a "surface" curve.

    Used in later phases as source / target points for policy
    transportation (Sec. IV-A linear and IV-B nonlinear).

    Parameters
    ----------
    kind : {"flat", "curved"}
        ``"flat"``   — y = 0 line.
        ``"curved"`` — y = ``amplitude`` * sin(π x / x_max).
    n_points : int
        Number of points sampled along the surface.
    x_range : (float, float)
        x extent of the surface.
    amplitude : float
        Bump amplitude of the curved surface (ignored when flat).

    Returns
    -------
    (n_points, 2) array of surface points.
    """
    if n_points < 2:
        raise ValueError("n_points must be >= 2.")
    x_lo, x_hi = float(x_range[0]), float(x_range[1])
    x = np.linspace(x_lo, x_hi, n_points)
    if kind == "flat":
        y = np.zeros_like(x)
    elif kind == "curved":
        x_max = max(abs(x_lo), abs(x_hi))
        y = float(amplitude) * np.sin(np.pi * x / x_max)
    else:
        raise ValueError(f"kind must be 'flat' or 'curved', got {kind!r}")
    return np.stack([x, y], axis=1)
