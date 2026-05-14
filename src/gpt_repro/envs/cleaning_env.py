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
    n_visual: int = 150,
    source_pts: "np.ndarray | None" = None,
    target_pts: "np.ndarray | None" = None,
) -> str:
    """Generate MuJoCo XML for surface cleaning.

    Phase 12: added source cloud (blue), target cloud (orange), coloured base
    plane by surface kind, capsule EE, and a fixed camera.

    Source/target clouds are subsampled to ``n_visual`` geoms each (cap=150).
    """
    c = surface_config.center

    # Floor colour by surface kind
    floor_colours = {
        "flat": "0.80 0.80 0.80 1",
        "tilted": "0.75 0.82 0.90 1",
        "curved": "0.72 0.88 0.74 1",
        "bumpy": "0.90 0.88 0.70 1",
    }
    floor_rgba = floor_colours.get(surface_config.kind, "0.75 0.75 0.75 1")

    # Target cloud (orange) from surface config
    if target_pts is None:
        target_pts = make_surface_pointcloud(surface_config, n_points=n_visual, seed=0)

    # Subsample both clouds to cap
    def _subsample(pts: np.ndarray, cap: int) -> np.ndarray:
        if len(pts) > cap:
            idx = np.round(np.linspace(0, len(pts) - 1, cap)).astype(int)
            return pts[idx]
        return pts

    tgt = _subsample(target_pts, n_visual)

    tgt_lines = []
    for i, pt in enumerate(tgt):
        tgt_lines.append(
            f'    <geom name="tgt_{i}" type="sphere" size="0.004"'
            f' pos="{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}"'
            f' rgba="1.0 0.55 0.1 0.75" contype="0" conaffinity="0"/>'
        )

    src_lines = []
    if source_pts is not None:
        src = _subsample(source_pts, n_visual)
        for i, pt in enumerate(src):
            src_lines.append(
                f'    <geom name="src_{i}" type="sphere" size="0.004"'
                f' pos="{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}"'
                f' rgba="0.25 0.55 0.95 0.65" contype="0" conaffinity="0"/>'
            )

    all_sphere_xml = "\n".join(src_lines + tgt_lines)

    return f"""<mujoco model="cleaning">
  <option gravity="0 0 0" timestep="0.02"/>
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="2048"/>
  </visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" specular="0.3 0.3 0.3"/>
    <light pos="-1 -1 2" dir="1 1 -1" diffuse="0.5 0.5 0.5" specular="0 0 0"/>
    <camera name="fixed" mode="fixed" pos="0.0 -0.5 0.8" zaxis="0.000 0.908 -0.419"/>
    <geom name="floor" type="plane" size="1 1 0.05" rgba="{floor_rgba}" pos="{c[0]:.3f} {c[1]:.3f} 0"/>
{all_sphere_xml}
    <body name="ee_body" pos="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f}">
      <joint name="ee_x" type="slide" axis="1 0 0" range="-2 2"/>
      <joint name="ee_y" type="slide" axis="0 1 0" range="-2 2"/>
      <joint name="ee_z" type="slide" axis="0 0 1" range="-2 2"/>
      <geom name="ee_geom" type="capsule" size="0.015 0.020"
            fromto="0 0 -0.020 0 0 0.020" rgba="0.95 0.95 0.95 1"/>
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

    import numpy as _np
    _CAM_LOOKAT    = _np.array([0.5, 0.0, 0.5])
    _CAM_DISTANCE  = 1.0
    _CAM_ELEVATION = 30.0
    _CAM_AZIMUTH   = -90.0

    def __init__(
        self,
        surface_config: SurfaceConfig,
        dt: float = 0.02,
        n_surface_pts: int = 400,
        render_mode: Optional[str] = None,
        source_config: Optional[SurfaceConfig] = None,
    ) -> None:
        self._surface_config = surface_config
        self._n_surface_pts = n_surface_pts

        # Generate surface cloud for coverage / distance computation
        self._surface_pts = make_surface_pointcloud(
            surface_config, n_points=n_surface_pts, seed=0
        )

        # Source cloud (optional) shown as blue dots in render
        source_pts = None
        if source_config is not None:
            source_pts = make_surface_pointcloud(source_config, n_points=150, seed=0)

        xml = _build_cleaning_xml(
            surface_config,
            n_visual=150,
            source_pts=source_pts,
            target_pts=self._surface_pts,
        )
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
