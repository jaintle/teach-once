"""Visual asset helpers for Phase 12 MuJoCo environments.

All XML fragments must remain under 80 lines each.
Units: metres. Coordinates: MuJoCo world frame (Z-up).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Shared visual defaults (lighting, haze, shadow quality)
# ---------------------------------------------------------------------------

VISUAL_DEFAULTS = """\
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8"
               specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="2048"/>
  </visual>"""

# ---------------------------------------------------------------------------
# Floor plane
# ---------------------------------------------------------------------------

FLOOR_GEOM = (
    '  <geom name="floor" type="plane" size="2 2 0.1"'
    ' rgba="0.72 0.72 0.72 1" pos="0 0 0"/>'
)

# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


def fixed_camera_xml(pos: np.ndarray, target: np.ndarray) -> str:
    """Return a <camera> tag with name="fixed" pointing from pos toward target.

    Uses ``zaxis`` (normalized pos→target direction) which MuJoCo 3.x accepts.

    Parameters
    ----------
    pos : (3,) camera position.
    target : (3,) point the camera looks toward.

    Returns
    -------
    str — single <camera .../> XML line.
    """
    p = np.asarray(pos, dtype=float)
    t = np.asarray(target, dtype=float)
    z = t - p
    norm = np.linalg.norm(z)
    if norm > 1e-10:
        z = z / norm
    return (
        f'  <camera name="fixed" mode="fixed"'
        f' pos="{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}"'
        f' zaxis="{z[0]:.3f} {z[1]:.3f} {z[2]:.3f}"/>'
    )


# ---------------------------------------------------------------------------
# End-effector tool geom
# ---------------------------------------------------------------------------


def ee_tool_geom(radius: float = 0.025, color: str = "0.2 0.6 0.9 1") -> str:
    """Capsule geom representing the EE tool tip (Z-axis aligned)."""
    length = 0.06
    return (
        f'      <geom name="ee_geom" type="capsule"'
        f' size="{radius:.4f} {length/2:.4f}"'
        f' fromto="0 0 -{length/2:.4f} 0 0 {length/2:.4f}"'
        f' rgba="{color}"/>'
    )


# ---------------------------------------------------------------------------
# Primitive geom helpers
# ---------------------------------------------------------------------------


def box_geom(
    size: np.ndarray,
    pos: np.ndarray,
    color: str,
    name: str,
) -> str:
    """Return a <geom> box tag."""
    s = np.asarray(size, dtype=float)
    p = np.asarray(pos, dtype=float)
    return (
        f'  <geom name="{name}" type="box"'
        f' size="{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}"'
        f' pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}"'
        f' rgba="{color}"/>'
    )


def sphere_geom(
    radius: float,
    pos: np.ndarray,
    color: str,
    name: str,
) -> str:
    """Return a <geom> sphere tag."""
    p = np.asarray(pos, dtype=float)
    return (
        f'  <geom name="{name}" type="sphere"'
        f' size="{radius:.4f}"'
        f' pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}"'
        f' rgba="{color}"/>'
    )


def capsule_between(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: str,
    name: str,
) -> str:
    """Return a capsule geom between two points (arm link visual)."""
    s = np.asarray(start, dtype=float)
    e = np.asarray(end, dtype=float)
    return (
        f'  <geom name="{name}" type="capsule"'
        f' fromto="{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}'
        f' {e[0]:.4f} {e[1]:.4f} {e[2]:.4f}"'
        f' size="{radius:.4f}"'
        f' rgba="{color}"/>'
    )


# ---------------------------------------------------------------------------
# Surface mesh as tiny sphere array
# ---------------------------------------------------------------------------


def make_surface_mesh_xml(
    surface_points: np.ndarray,
    name: str = "surface",
    color: str = "0.3 0.7 0.4 0.6",
    cap: int = 150,
) -> str:
    """Return XML for a point-cloud surface visualization.

    Caps at ``cap`` geoms (subsampled uniformly if needed) to keep XML size
    manageable. Each point is a tiny sphere of radius 0.005 m.

    Parameters
    ----------
    surface_points : (N, 3) array.
    name : geom name prefix.
    color : RGBA string.
    cap : maximum number of geoms (default 150).

    Returns
    -------
    str — newline-separated geom tags.
    """
    pts = np.asarray(surface_points, dtype=float)
    if len(pts) > cap:
        idx = np.round(np.linspace(0, len(pts) - 1, cap)).astype(int)
        pts = pts[idx]
    lines = []
    for i, pt in enumerate(pts):
        lines.append(
            f'  <geom name="{name}_{i}" type="sphere" size="0.005"'
            f' pos="{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}"'
            f' rgba="{color}" contype="0" conaffinity="0"/>'
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Arrow (source→target correspondence line)
# ---------------------------------------------------------------------------


def make_arrow_xml(
    start: np.ndarray,
    end: np.ndarray,
    name: str,
    color: str = "1 0 0 1",
    radius: float = 0.005,
) -> str:
    """Return a capsule geom between start and end (correspondence arrow)."""
    return capsule_between(start, end, radius, color, name)
