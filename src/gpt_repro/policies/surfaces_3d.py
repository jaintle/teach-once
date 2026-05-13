"""3D surface generators for Phase 10 — Sec. VI-C analog.

Provides procedural point-cloud surfaces and demonstration generation
for the 3D surface cleaning experiment. All surfaces span roughly
0.3 m × 0.3 m in the XY plane.

Paper grounding: Sec. VI-C "Robot Surface Cleaning" — depth-camera
point clouds replaced by procedural generation (scope note in CLAUDE.md:
'No point-cloud SVGP for the cleaning task at 3D scale (paper Sec. VI-C);
we only do the 2D version in Sec. V-A.').
This module extends that note to provide a 3D analog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# SurfaceConfig
# ---------------------------------------------------------------------------

@dataclass
class SurfaceConfig:
    """Configuration for a 3D surface shape.

    Parameters
    ----------
    kind : str
        One of "flat", "tilted", "curved", "bumpy".
    center : (3,) array
        Center of the surface patch in world coordinates.
    normal : (3,) array
        Unit normal vector (used for flat/tilted).
    params : dict
        Shape-specific parameters:
        - "flat":   no extra params needed.
        - "tilted": no extra params (tilt from normal already captured).
        - "curved": "amplitude" (default 0.05), "frequency" (default 2π).
        - "bumpy":  "n_bumps" (default 5), "bump_amplitude" (default 0.04),
                    "bump_sigma" (default 0.05).
    """
    kind: str = "flat"
    center: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.0, 0.5]))
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    params: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.normal = np.asarray(self.normal, dtype=float)
        n = np.linalg.norm(self.normal)
        if n > 1e-10:
            self.normal = self.normal / n


# ---------------------------------------------------------------------------
# Surface point cloud generation
# ---------------------------------------------------------------------------

def make_surface_pointcloud(
    config: SurfaceConfig,
    n_points: int = 400,
    seed: int = 0,
) -> np.ndarray:
    """Generate a point cloud on a 3D surface.

    All surfaces span roughly 0.3 m × 0.3 m in XY with their center
    at ``config.center``.

    Parameters
    ----------
    config : SurfaceConfig
    n_points : int — number of output points.
    seed : int — random seed (used only for bumpy bump centers).

    Returns
    -------
    points : (N, 3) numpy array, row-major (row-scan order).
    """
    side = int(np.ceil(np.sqrt(n_points)))
    span = 0.3  # meters
    xs = np.linspace(-span / 2, span / 2, side)
    ys = np.linspace(-span / 2, span / 2, side)
    XX, YY = np.meshgrid(xs, ys)  # (side, side)
    x_flat = XX.ravel()[:n_points]
    y_flat = YY.ravel()[:n_points]

    kind = config.kind
    c = config.center
    p = config.params

    if kind == "flat":
        z_flat = np.zeros(len(x_flat))
        pts = np.column_stack([x_flat, y_flat, z_flat])

    elif kind == "tilted":
        # Tilt the flat plane so its normal matches config.normal.
        # Compute rotation from [0,0,1] to config.normal.
        z_flat = np.zeros(len(x_flat))
        pts_local = np.column_stack([x_flat, y_flat, z_flat])
        R = _rotation_to_normal(config.normal)
        pts = pts_local @ R.T

    elif kind == "curved":
        # Sinusoidal surface: z = A·sin(k·x)·cos(k·y)  — Sec. VI-C visual.
        A = float(p.get("amplitude", 0.05))
        k = float(p.get("frequency", 2 * np.pi))
        z_flat = A * np.sin(k * x_flat) * np.cos(k * y_flat)
        pts = np.column_stack([x_flat, y_flat, z_flat])

    elif kind == "bumpy":
        # Gaussian bumps scattered on a base plane.
        rng = np.random.default_rng(seed)
        n_bumps = int(p.get("n_bumps", 5))
        a = float(p.get("bump_amplitude", 0.04))
        sigma = float(p.get("bump_sigma", 0.05))
        span_half = 0.3 / 2
        centers = rng.uniform(-span_half, span_half, size=(n_bumps, 2))
        xy = np.column_stack([x_flat, y_flat])
        z_flat = np.zeros(len(x_flat))
        for ci in centers:
            dist2 = np.sum((xy - ci) ** 2, axis=1)
            z_flat += a * np.exp(-dist2 / (2 * sigma ** 2))
        pts = np.column_stack([x_flat, y_flat, z_flat])

    else:
        raise ValueError(f"Unknown surface kind: {kind!r}")

    # Translate to config.center
    pts = pts + c
    return pts.astype(float)


def _rotation_to_normal(normal: np.ndarray) -> np.ndarray:
    """Return 3×3 rotation matrix that takes [0,0,1] to ``normal``."""
    n = normal / (np.linalg.norm(normal) + 1e-10)
    z = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z, n)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-8:
        # Already aligned or anti-aligned
        if np.dot(z, n) > 0:
            return np.eye(3)
        else:
            return np.diag([1.0, -1.0, -1.0])
    axis = cross / cross_norm
    angle = np.arccos(np.clip(np.dot(z, n), -1, 1))
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


# ---------------------------------------------------------------------------
# Surface normal at a point
# ---------------------------------------------------------------------------

def _surface_normal_at(config: SurfaceConfig, xy: np.ndarray) -> np.ndarray:
    """Approximate outward surface normal at a 2D (x,y) position."""
    kind = config.kind
    p = config.params

    if kind in ("flat", "tilted"):
        return config.normal.copy()
    elif kind == "curved":
        A = float(p.get("amplitude", 0.05))
        k = float(p.get("frequency", 2 * np.pi))
        x, y = float(xy[0]) - config.center[0], float(xy[1]) - config.center[1]
        # ∂z/∂x = A·k·cos(kx)·cos(ky), ∂z/∂y = -A·k·sin(kx)·sin(ky)
        dzdx = A * k * np.cos(k * x) * np.cos(k * y)
        dzdy = -A * k * np.sin(k * x) * np.sin(k * y)
        n = np.array([-dzdx, -dzdy, 1.0])
        return n / (np.linalg.norm(n) + 1e-10)
    else:
        return np.array([0.0, 0.0, 1.0])


def _gram_schmidt_rotation(n: np.ndarray) -> np.ndarray:
    """Build SO(3) rotation matrix R whose third column is the unit normal n.

    Used to orient the end-effector z-axis with the surface normal.
    """
    n = n / (np.linalg.norm(n) + 1e-10)
    # Pick a non-parallel vector to build the tangent frame
    if abs(n[0]) < 0.9:
        t = np.array([1.0, 0.0, 0.0])
    else:
        t = np.array([0.0, 1.0, 0.0])
    t = t - np.dot(t, n) * n
    t /= np.linalg.norm(t) + 1e-10
    b = np.cross(n, t)
    b /= np.linalg.norm(b) + 1e-10
    R = np.column_stack([t, b, n])  # columns: tangent, binormal, normal
    return R


# ---------------------------------------------------------------------------
# Surface demonstration
# ---------------------------------------------------------------------------

def make_surface_demo(
    source_config: SurfaceConfig,
    n_points: int = 100,
    noise: float = 0.0,
    seed: int = 0,
    dt: float = 0.02,
) -> dict:
    """Generate a raster-scan cleaning trajectory on ``source_config``.

    Produces a back-and-forth raster scan covering the surface patch
    (analogous to the cyclic cleaning motion in Sec. VI-C).

    Stiffness profile: diagonal ``Ks`` with z-component varying
    sinusoidally along the trajectory (encodes increasing/decreasing force
    regions matching Fig. 16's trend). Per Sec. IV-D: K̂_s = J K_s Jᵀ.

    Orientation: EE z-axis aligned with surface normal at each waypoint
    via Gram-Schmidt (Eq. (15) convention).

    Parameters
    ----------
    source_config : SurfaceConfig
    n_points : int — number of waypoints.
    noise : float — Gaussian noise std added to positions.
    seed : int — RNG seed.
    dt : float — time step.

    Returns
    -------
    dict with keys:
        "x"           : (N, 3) positions on the surface.
        "xdot"        : (N, 3) velocities (central differences).
        "t"           : (N,)   time stamps.
        "stiffness"   : (N, 3, 3) diagonal stiffness matrices.
        "orientation" : (N, 3, 3) SO(3) orientation matrices.
    """
    rng = np.random.default_rng(seed)
    span = 0.3
    c = source_config.center

    # Build raster scan in 2D
    n_rows = max(2, int(np.sqrt(n_points)))
    xs = np.linspace(-span / 2, span / 2, n_rows)
    ys = np.linspace(-span / 2, span / 2, n_points // n_rows + 1)

    waypoints_2d = []
    for i, xi in enumerate(xs):
        col = ys if i % 2 == 0 else ys[::-1]
        for yj in col:
            waypoints_2d.append([xi, yj])
        if len(waypoints_2d) >= n_points:
            break
    waypoints_2d = np.array(waypoints_2d[:n_points])  # (N, 2)

    # Lift onto surface
    kind = source_config.kind
    p = source_config.params
    xy_local = waypoints_2d  # (N, 2) relative to center projected

    if kind == "flat":
        z = np.zeros(len(waypoints_2d))
    elif kind == "tilted":
        z = np.zeros(len(waypoints_2d))
        R = _rotation_to_normal(source_config.normal)
        pts3 = np.column_stack([xy_local[:, 0], xy_local[:, 1], z]) @ R.T
        x_arr = pts3 + c
        x_arr = x_arr[:n_points]
        if noise > 0:
            x_arr += rng.standard_normal(x_arr.shape) * noise
        return _build_demo_dict(x_arr, source_config, dt, n_points, rng, noise)
    elif kind == "curved":
        A = float(p.get("amplitude", 0.05))
        k = float(p.get("frequency", 2 * np.pi))
        x_rel = xy_local[:, 0]
        y_rel = xy_local[:, 1]
        z = A * np.sin(k * x_rel) * np.cos(k * y_rel)
    elif kind == "bumpy":
        n_bumps = int(p.get("n_bumps", 5))
        a = float(p.get("bump_amplitude", 0.04))
        sigma = float(p.get("bump_sigma", 0.05))
        span_half = 0.3 / 2
        bump_rng = np.random.default_rng(seed)  # same seed as pointcloud
        centers = bump_rng.uniform(-span_half, span_half, size=(n_bumps, 2))
        z = np.zeros(len(waypoints_2d))
        for ci in centers:
            dist2 = np.sum((xy_local - ci) ** 2, axis=1)
            z += a * np.exp(-dist2 / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unknown surface kind: {kind!r}")

    x_arr = np.column_stack([xy_local[:, 0], xy_local[:, 1], z]) + c
    return _build_demo_dict(x_arr, source_config, dt, n_points, rng, noise)


def _build_demo_dict(
    x_arr: np.ndarray,
    config: SurfaceConfig,
    dt: float,
    n_points: int,
    rng: np.random.Generator,
    noise: float,
) -> dict:
    """Compute velocity, stiffness, orientation for a trajectory array."""
    N = len(x_arr)
    if noise > 0:
        x_arr = x_arr + rng.standard_normal(x_arr.shape) * noise

    # Velocities via central differences
    xdot = np.empty_like(x_arr)
    xdot[1:-1] = (x_arr[2:] - x_arr[:-2]) / (2 * dt)
    xdot[0] = (x_arr[1] - x_arr[0]) / dt
    xdot[-1] = (x_arr[-1] - x_arr[-2]) / dt

    # Time stamps
    t = np.arange(N) * dt

    # Stiffness: diagonal with z-component sinusoidal (increasing/decreasing trend)
    # Matches Fig. 16: "force profile preserved across surfaces"
    Ks = np.zeros((N, 3, 3))
    phase = np.linspace(0, 2 * np.pi, N)
    kz_profile = 200.0 + 100.0 * np.sin(phase)  # 100–300 N/m
    for i in range(N):
        Ks[i] = np.diag([100.0, 100.0, kz_profile[i]])

    # Orientation: EE z-axis aligned with surface normal at each point
    Rots = np.empty((N, 3, 3))
    for i in range(N):
        n = _surface_normal_at(config, x_arr[i])
        Rots[i] = _gram_schmidt_rotation(n)

    return {"x": x_arr, "xdot": xdot, "t": t, "stiffness": Ks, "orientation": Rots}


# ---------------------------------------------------------------------------
# Cloud pairing
# ---------------------------------------------------------------------------

def pair_surface_clouds(
    S_cloud: np.ndarray,
    T_cloud: np.ndarray,
) -> tuple:
    """Pair two point clouds by nearest-neighbour matching (KD-tree).

    For each point in ``S_cloud``, finds the nearest point in ``T_cloud``
    and returns the paired arrays. This relaxes the "already paired"
    assumption of the paper for procedural clouds.

    NOTE: For very differently-shaped surfaces the NN pairing may be
    suboptimal. The paper uses paired point clouds from the depth camera
    which have an implicit spatial correspondence. Logged in Phase 10 entry.

    Parameters
    ----------
    S_cloud : (N, 3) source cloud.
    T_cloud : (M, 3) target cloud.

    Returns
    -------
    S_paired : (N, 3) — same as S_cloud.
    T_paired : (N, 3) — T_cloud points matched to each S_cloud point.
    """
    tree = cKDTree(T_cloud)
    _, idx = tree.query(S_cloud, k=1)
    return S_cloud.copy(), T_cloud[idx]
