"""Scene XML builder for Franka Panda environments — Phase 13.

Builds complete MuJoCo XML strings for three tasks:
  - "reshelving": arm + table + shelf + orange object + goal slot
  - "cleaning":   arm + table + cleaning surface mesh
  - "armpose":    arm + table + 4 coloured keypoint spheres

Usage::

    from gpt_repro.envs.franka_scene import build_scene_xml
    xml_str = build_scene_xml("reshelving")

The XML is written to a temp file in FRANKA_ASSETS_DIR before loading
so that the panda_with_site.xml's relative meshdir="assets" resolves
correctly.  Callers should use :func:`load_scene_model` rather than
calling ``MjModel.from_xml_string`` directly.
"""

from __future__ import annotations

import pathlib
import tempfile
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FRANKA_ASSETS_DIR: pathlib.Path = pathlib.Path(__file__).parent / "assets" / "franka"
_PANDA_XML = FRANKA_ASSETS_DIR / "panda_with_site.xml"

# ---------------------------------------------------------------------------
# Camera helpers — azimuth/elevation for MjvCamera (CAMERA_FREE mode)
# ---------------------------------------------------------------------------

# Each entry: (lookat_xyz, distance, elevation_deg, azimuth_deg)
CAMERAS = {
    "front":   (np.array([0.4, 0.0, 0.65]), 1.8, -25.0, 180.0),
    "side":    (np.array([0.4, 0.0, 0.65]), 1.8, -20.0, 90.0),
    "top":     (np.array([0.4, 0.0, 0.5]), 1.5, -65.0, 180.0),
    # 3/4 angle: arm AND surface visible simultaneously
    "quarter": (np.array([0.45, 0.15, 0.65]), 1.6, -35.0, 225.0),
}


# ---------------------------------------------------------------------------
# Task geometry helpers
# ---------------------------------------------------------------------------

def _table_xml(
    pos: tuple = (0.5, 0.0, 0.4),
    size: tuple = (0.3, 0.2, 0.02),
) -> str:
    """One flat tabletop + four legs.  All use material='table_mat'."""
    cx, cy, cz = pos
    hw, hd, _ = size          # half-width, half-depth of top
    leg_h = cz - 0.01         # floor to bottom of top
    leg_hw = 0.02              # leg cross-section half-size
    # Table top
    top = (
        f'    <geom name="table_top" type="box" size="{hw} {hd} {size[2]:.3f}"\n'
        f'          pos="{cx:.4f} {cy:.4f} {cz:.4f}" material="table_mat"/>\n'
    )
    # Four legs at corners
    legs = ""
    for idx, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        lx = cx + sx * (hw - leg_hw)
        ly = cy + sy * (hd - leg_hw)
        lz = leg_h / 2
        legs += (
            f'    <geom name="table_leg{idx}" type="box"'
            f' size="{leg_hw} {leg_hw} {lz:.4f}"\n'
            f'          pos="{lx:.4f} {ly:.4f} {lz:.4f}" material="table_mat"/>\n'
        )
    return top + legs


def _shelf_xml(pos: tuple = (0.0, 0.6, 0.8)) -> str:
    """Wall-mounted shelf: two planks + two vertical supports + goal slot."""
    sx, sy, sz = pos
    return (
        f'    <geom name="shelf_lower" type="box" size="0.25 0.08 0.008"\n'
        f'          pos="{sx:.4f} {sy:.4f} {sz - 0.12:.4f}" material="shelf_mat"/>\n'
        f'    <geom name="shelf_upper" type="box" size="0.25 0.08 0.008"\n'
        f'          pos="{sx:.4f} {sy:.4f} {sz + 0.12:.4f}" material="shelf_mat"/>\n'
        f'    <geom name="shelf_left"  type="box" size="0.008 0.08 0.14"\n'
        f'          pos="{sx - 0.25:.4f} {sy:.4f} {sz:.4f}" material="shelf_mat"/>\n'
        f'    <geom name="shelf_right" type="box" size="0.008 0.08 0.14"\n'
        f'          pos="{sx + 0.25:.4f} {sy:.4f} {sz:.4f}" material="shelf_mat"/>\n'
        f'    <geom name="shelf_back"  type="box" size="0.25 0.008 0.14"\n'
        f'          pos="{sx:.4f} {sy + 0.08:.4f} {sz:.4f}" material="shelf_mat"/>\n'
        f'    <geom name="goal_slot" type="box" size="0.06 0.06 0.06"\n'
        f'          pos="{sx:.4f} {sy - 0.02:.4f} {sz:.4f}" material="goal_mat"/>\n'
    )


def _object_xml(pos: tuple = (0.5, 0.0, 0.65), name: str = "object") -> str:
    """Orange box object with AprilTag-proxy face."""
    ox, oy, oz = pos
    return (
        f'    <geom name="{name}" type="box" size="0.025 0.025 0.025"\n'
        f'          pos="{ox:.4f} {oy:.4f} {oz:.4f}" material="object_mat"/>\n'
        f'    <geom name="{name}_tag" type="box" size="0.012 0.012 0.001"\n'
        f'          pos="{ox:.4f} {oy - 0.026:.4f} {oz:.4f}" rgba="0.1 0.1 0.1 1"/>\n'
    )


def _cleaning_surface_xml(pos: tuple = (0.5, 0.0, 0.62)) -> str:
    """Flat cleaning surface (thin plane on table top)."""
    cx, cy, cz = pos
    return (
        f'    <geom name="cleaning_surface" type="box" size="0.15 0.15 0.004"\n'
        f'          pos="{cx:.4f} {cy:.4f} {cz:.4f}" material="surface_mat"/>\n'
    )


def _armpose_spheres_xml(large: bool = False) -> str:
    """Four coloured keypoint spheres for arm-pose task.

    Parameters
    ----------
    large : bool
        If True, use larger radii (0.06/0.05/0.04/0.04) for the
        "best scene" single-scene render so spheres are clearly visible.
    """
    if large:
        r0, r1, r2, r3 = 0.060, 0.050, 0.040, 0.040
    else:
        r0, r1, r2, r3 = 0.030, 0.022, 0.018, 0.015
    return (
        f'    <geom name="kp_shoulder" type="sphere" size="{r0:.3f}"\n'
        f'          pos="0.35 0.0 0.70" rgba="0.0 0.9 0.9 0.9"/>\n'
        f'    <geom name="kp_elbow"    type="sphere" size="{r1:.3f}"\n'
        f'          pos="0.47 0.0 0.80" rgba="0.9 0.0 0.9 0.9"/>\n'
        f'    <geom name="kp_wrist"    type="sphere" size="{r2:.3f}"\n'
        f'          pos="0.57 0.0 0.75" rgba="1.0 0.9 0.0 0.9"/>\n'
        f'    <geom name="kp_hand"     type="sphere" size="{r3:.3f}"\n'
        f'          pos="0.62 0.0 0.65" rgba="0.2 0.4 0.9 0.9"/>\n'
    )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_scene_xml(
    task: str,
    table_pos: tuple = (0.5, 0.0, 0.4),
    shelf_pos: tuple = (0.0, 0.6, 0.8),
    object_pos: tuple = (0.5, 0.0, 0.65),
    camera_configs: Optional[list] = None,
    large_kp_spheres: bool = False,
) -> str:
    """Build a complete MuJoCo XML string for the given task.

    The Franka Panda arm is included via ``<include file="...panda_with_site.xml"/>``
    which merges the arm's worldbody, assets, and actuators into the scene.

    Parameters
    ----------
    task : str — one of "reshelving", "cleaning", "armpose".
    table_pos : (x, y, z) center of table top surface.
    shelf_pos : (x, y, z) center of shelf unit.
    object_pos : (x, y, z) center of orange task object.
    camera_configs : unused (cameras defined in CAMERAS constant).

    Returns
    -------
    str — complete MuJoCo XML.
    """
    panda_xml_path = str(_PANDA_XML.resolve())
    meshdir = str((FRANKA_ASSETS_DIR / "assets").resolve())

    # Task-specific geometry
    task_geoms = _table_xml(pos=table_pos)
    if task == "reshelving":
        task_geoms += _shelf_xml(pos=shelf_pos)
        task_geoms += _object_xml(pos=object_pos)
    elif task == "cleaning":
        surf_pos = (table_pos[0], table_pos[1], table_pos[2] + 0.022)
        task_geoms += _cleaning_surface_xml(pos=surf_pos)
    elif task == "armpose":
        task_geoms += _armpose_spheres_xml(large=large_kp_spheres)
    else:
        raise ValueError(f"Unknown task: {task!r}. Must be reshelving, cleaning, or armpose.")

    xml = f"""<mujoco model="gpt_scene_{task}">
  <compiler angle="radian" meshdir="{meshdir}" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <visual>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
    <rgba haze="0.1 0.15 0.2 1"/>
    <quality shadowsize="2048" offsamples="4"/>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient"
             rgb1="0.4 0.6 0.8" rgb2="0.1 0.2 0.4"
             width="512" height="512"/>
    <texture name="floor_tex" type="2d" builtin="checker"
             rgb1="0.72 0.72 0.72" rgb2="0.55 0.55 0.55"
             width="512" height="512"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="4 4" reflectance="0.1"/>
    <material name="table_mat" rgba="0.65 0.45 0.25 1" reflectance="0.05"/>
    <material name="shelf_mat" rgba="0.55 0.35 0.20 1" reflectance="0.05"/>
    <material name="object_mat" rgba="0.90 0.50 0.10 1" reflectance="0.1"/>
    <material name="goal_mat"   rgba="0.20 0.80 0.20 0.6" reflectance="0.05"/>
    <material name="surface_mat" rgba="0.80 0.80 0.90 1" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light name="sun"  pos="0 0 4" dir="0 0 -1"
           diffuse="0.9 0.9 0.9" castshadow="true"/>
    <light name="fill" pos="1.5 -1.5 3" dir="-0.5 0.5 -1"
           diffuse="0.4 0.4 0.5" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floor_mat" pos="0 0 0"/>
{task_geoms}  </worldbody>
  <include file="{panda_xml_path}"/>
</mujoco>
"""
    return xml


def load_scene_model(xml: str):
    """Write XML to a temp file in FRANKA_ASSETS_DIR and load it.

    This is required so that panda_with_site.xml's ``meshdir="assets"``
    resolves correctly relative to FRANKA_ASSETS_DIR.

    Parameters
    ----------
    xml : str — scene XML returned by :func:`build_scene_xml`.

    Returns
    -------
    (mujoco.MjModel, mujoco.MjData)
    """
    import mujoco  # local import to keep module importable without mujoco installed

    # Write to a deterministic temp file inside the assets dir (not /tmp)
    # so relative mesh paths in the included panda XML resolve correctly.
    tmp_path = FRANKA_ASSETS_DIR / "_scene_tmp.xml"
    tmp_path.write_text(xml, encoding="utf-8")
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    data = mujoco.MjData(model)
    return model, data
