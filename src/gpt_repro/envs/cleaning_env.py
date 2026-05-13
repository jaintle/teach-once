"""Surface cleaning environment — Phase 10 (Sec. VI-C analog).

Extends ``KinematicEndEffectorEnv`` for 3D surface cleaning tasks.
Provides:
- Surface-conditioned reset (EE placed at start of raster scan).
- Force estimation via Hooke's law proxy: ‖F‖ = ‖Ks · ẋ‖.
  NOTE: This is a proxy, not real MuJoCo contact force.
  Paper Sec. VI-C uses a force-torque sensor; we approximate it
  with the transported stiffness profile (no FT sensor in simulation).
- Coverage fraction computation.
- ``get_surface_points()`` returning the target cloud.

MuJoCo XML: flat base plane + grid of visual spheres for surface shape.
Kept under 60 embedded XML lines.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from gpt_repro.envs.base_env import KinematicEndEffectorEnv
from gpt_repro.policies.surfaces_3d import SurfaceConfig, make_surface_pointcloud


# ---------------------------------------------------------------------------
# XML builder
# ---------------------------------------------------------------------------

def _build_cleaning_xml(
    surface_config: SurfaceConfig,
    n_visual: int = 16,
) -> str:
    """Generate MuJoCo XML with a cleaning surface (visual only).

    Uses a flat base plane + a sparse grid of small spheres to indicate
    the surface shape. The number of visual spheres is capped to keep
    XML under 60 lines.
    """
    c = surface_config.center
    # Sample a few representative surface points for visual markers
    vis_pts = make_surface_pointcloud(surface_config, n_points=n_visual, seed=0)

    sphere_lines = []
    for i, pt in enumerate(vis_pts):
        sphere_lines.append(
            f'    <geom name="surf_{i}" type="sphere" size="0.008" '
            f'pos="{pt[0]:.3f} {pt[1]:.3f} {pt[2]:.3f}" '
            f'rgba="0.8 0.5 0.2 0.5"/>'
        )
    spheres_xml = "\n".join(sphere_lines)

    return f"""<mujoco model="cleaning">
  <option gravity="0 0 0" timestep="0.02"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1 1 0.05" rgba="0.7 0.7 0.7 1" pos="{c[0]:.3f} {c[1]:.3f} 0"/>
{spheres_xml}
    <body name="ee_body" pos="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f}">
      <joint name="ee_x" type="slide" axis="1 0 0" range="-2 2"/>
      <joint name="ee_y" type="slide" axis="0 1 0" range="-2 2"/>
      <joint name="ee_z" type="slide" axis="0 0 1" range="-2 2"/>
      <geom name="ee_geom" type="sphere" size="0.015" rgba="0.1 0.5 0.9 1"/>
    </body>
  </worldbody>
</mujoco>"""


# ---------------------------------------------------------------------------
# SurfaceCleaningEnv
# ---------------------------------------------------------------------------

class SurfaceCleaningEnv(KinematicEndEffectorEnv):
    """3D kinematic surface cleaning environment — Phase 10.

    The EE traverses a target surface following a transported cleaning
    demonstration. Force is estimated via Hooke's law proxy (no MuJoCo
    contact forces — documented in Phase 10 log).

    Parameters
    ----------
    surface_config : SurfaceConfig
        Target surface configuration.
    dt : float, optional
        Control time step. Defaults to 0.02 s.
    n_surface_pts : int, optional
        Number of surface points for coverage computation. Defaults to 400.
    render_mode : str or None, optional
        See ``KinematicEndEffectorEnv``.
    """

    def __init__(
        self,
        surface_config: SurfaceConfig,
        dt: float = 0.02,
        n_surface_pts: int = 400,
        render_mode: Optional[str] = None,
    ) -> None:
        self._surface_config = surface_config
        self._n_surface_pts = n_surface_pts

        # Generate surface cloud for coverage / distance computation
        self._surface_pts = make_surface_pointcloud(
            surface_config, n_points=n_surface_pts, seed=0
        )

        xml = _build_cleaning_xml(surface_config, n_visual=16)
        super().__init__(xml_string=xml, dt=dt, render_mode=render_mode)

        # Reset EE to the first surface point
        self._start_pos: np.ndarray = self._surface_pts[0].copy()

    # ------------------------------------------------------------------
    # gymnasium.Env overrides
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict]:
        if options is None:
            options = {}
        if "init_pos" not in options:
            options["init_pos"] = self._start_pos.copy()
        return super().reset(seed=seed, options=options)

    # ------------------------------------------------------------------
    # Surface-specific methods
    # ------------------------------------------------------------------

    def get_surface_points(self, n_points: int = 400) -> np.ndarray:
        """Return the target surface point cloud (N, 3)."""
        if n_points == self._n_surface_pts:
            return self._surface_pts.copy()
        return make_surface_pointcloud(
            self._surface_config, n_points=n_points, seed=0
        )

    def get_contact_force_norm(
        self,
        stiffness: np.ndarray,
        xdot: np.ndarray,
    ) -> float:
        """Estimate contact force norm via Hooke's law proxy.

        Computes ``‖F‖ = ‖K_s · ẋ‖`` as a scalar force proxy.

        NOTE: This is NOT a real MuJoCo contact force. The paper uses
        a force-torque sensor (Sec. VI-C). This proxy is used for the
        force profile reproduction (Fig. 16 analog) in simulation.

        Parameters
        ----------
        stiffness : (3, 3) transported stiffness matrix.
        xdot : (3,) velocity at the current timestep.

        Returns
        -------
        float — ‖K_s · ẋ‖.
        """
        K = np.asarray(stiffness, dtype=float)
        v = np.asarray(xdot, dtype=float)
        F = K @ v
        return float(np.linalg.norm(F))

    def is_on_surface(self, pos: np.ndarray, tol: float = 0.02) -> bool:
        """Return True if ``pos`` is within ``tol`` m of the nearest surface point."""
        from scipy.spatial import cKDTree
        tree = cKDTree(self._surface_pts)
        dist, _ = tree.query(pos, k=1)
        return bool(dist < tol)

    def coverage_fraction(
        self, rollout_x: np.ndarray, tol: float = 0.025
    ) -> float:
        """Fraction of surface grid cells visited within ``tol`` m.

        Uses ball-radius coverage: a surface cell is "visited" if any
        rollout position is within ``tol`` of it.

        Parameters
        ----------
        rollout_x : (N, 3) EE rollout trajectory.
        tol : float — coverage ball radius in metres.

        Returns
        -------
        float in [0, 1].
        """
        from scipy.spatial import cKDTree
        if len(rollout_x) == 0:
            return 0.0
        tree = cKDTree(rollout_x)
        distances, _ = tree.query(self._surface_pts, k=1)
        visited = np.sum(distances < tol)
        return float(visited) / len(self._surface_pts)
