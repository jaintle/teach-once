#!/usr/bin/env python
"""Smoke test for Phase 10 — 3D surface cleaning pipeline.

Runs run_cleaning_pipeline with tiny inputs to verify:
- All output keys present.
- rollout_x shape (N, 3).
- force_norms shape (N,).
- coverage in [0, 1].
- All outputs finite.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np

from gpt_repro.policies.surfaces_3d import SurfaceConfig
from gpt_repro.transport.cleaning_pipeline_3d import run_cleaning_pipeline

REQUIRED_KEYS = {
    "rollout_x", "force_norms", "demo_x", "transported_x",
    "S", "T", "coverage", "mean_surface_dist", "transport",
    "transported_stiffness",
}

src = SurfaceConfig(kind="flat", center=np.array([0.5, 0.0, 0.5]))
tgt = SurfaceConfig(kind="tilted", center=np.array([0.5, 0.0, 0.5]),
                    normal=np.array([0.0, 0.2, 1.0]))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = run_cleaning_pipeline(
        src, tgt,
        n_source_pts=20, n_target_pts=20,
        n_inducing=5, n_demo_pts=20,
        seed=0, n_iter=20, n_steps=20, gp_n_iter=20,
    )

# Key presence
missing = REQUIRED_KEYS - set(result.keys())
assert not missing, f"Missing keys: {missing}"

# Shapes
rx = result["rollout_x"]
assert rx.ndim == 2 and rx.shape[1] == 3, f"rollout_x shape: {rx.shape}"
print(f"rollout_x shape : {rx.shape}")

fn = result["force_norms"]
assert fn.ndim == 1, f"force_norms shape: {fn.shape}"
print(f"force_norms shape: {fn.shape}")

cov = result["coverage"]
assert 0.0 <= cov <= 1.0, f"coverage {cov} not in [0,1]"
print(f"coverage         : {cov:.3f}")

msd = result["mean_surface_dist"]
assert np.isfinite(msd), f"mean_surface_dist is not finite: {msd}"
print(f"mean_surface_dist: {msd:.4f} m")

# Finiteness
assert np.all(np.isfinite(rx)), "rollout_x has non-finite values"
assert np.all(np.isfinite(fn)), "force_norms has non-finite values"

print("\nPhase 10 smoke test: PASS")
