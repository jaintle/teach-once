"""Tests for the DS policy module (Sec. III-A, Eq. 1).

Covers:

* Letter-C demo geometry (shapes + non-degenerate path length).
* DS fit/predict quality on the demo (RMSE < 0.2).
* Euler rollout from the demo start passes near the demo endpoint
  (loose sanity — the paper does not enforce SEDS-style stability).
* Predictive std grows out-of-distribution (Fig. 5/6 claim).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from gpt_repro.policies import (
    GPDynamicalSystem,
    make_cleaning_demo,
    make_letter_C_demo,
    make_surface_2d,
)
from gpt_repro.utils import set_global_seed


# ---------------------------------------------------------------------------
# Demo geometry
# ---------------------------------------------------------------------------
def test_letter_C_shape():
    """letter-C demo has correct shapes, non-degenerate path, and roughly
    matches the analytic 270°-arc geometry."""
    demo = make_letter_C_demo(n_points=100, duration=1.0)
    assert set(demo.keys()) == {"x", "xdot", "t"}
    assert demo["x"].shape == (100, 2)
    assert demo["xdot"].shape == (100, 2)
    assert demo["t"].shape == (100,)

    path_len = float(np.sum(np.linalg.norm(np.diff(demo["x"], axis=0), axis=-1)))
    # Analytic arc length is 3π/2 ≈ 4.71 for a 270° unit arc; we polyline
    # it so we get slightly less. Anything well above 2 is non-degenerate.
    assert path_len > 2.0, f"letter-C path looks degenerate: length={path_len}"

    # Polyline should stay close to the unit circle (radius ≈ 1 at every point).
    radii = np.linalg.norm(demo["x"], axis=1)
    assert np.all(np.abs(radii - 1.0) < 1e-9)


def test_cleaning_demo_shape_and_surface():
    """Cleaning demo touches the surface and lifts off; surface generators
    return the expected shapes."""
    demo = make_cleaning_demo(n_points=60, n_cycles=3)
    assert demo["x"].shape == (60, 2)
    assert demo["xdot"].shape == (60, 2)
    # y must oscillate down to 0 (touches the flat surface) and up.
    y = demo["x"][:, 1]
    assert y.min() < 1e-6
    assert y.max() > 0.1

    flat = make_surface_2d("flat", n_points=20)
    assert flat.shape == (20, 2)
    assert np.allclose(flat[:, 1], 0.0)

    curved = make_surface_2d("curved", n_points=20)
    assert curved.shape == (20, 2)
    # Curved surface is bumpy: not all y values are zero.
    assert np.max(np.abs(curved[:, 1])) > 0.05


# ---------------------------------------------------------------------------
# Fit quality
# ---------------------------------------------------------------------------
def test_ds_fit_predict_shapes():
    """GPDynamicalSystem must fit the letter-C demo well: predicted
    velocities at the training states should have RMSE < 0.2."""
    set_global_seed(0)
    demo = make_letter_C_demo(n_points=80, duration=1.0)
    ds = GPDynamicalSystem(n_iter_default=200, lr=0.1).fit(
        demo["x"], demo["xdot"]
    )

    mean, std = ds.predict(demo["x"], return_std=True)
    assert mean.shape == demo["xdot"].shape
    assert std.shape == demo["xdot"].shape

    rmse = float(np.sqrt(np.mean((mean - demo["xdot"]) ** 2)))
    assert rmse < 0.2, f"DS fit RMSE too high: {rmse}"


# ---------------------------------------------------------------------------
# Rollout follows the demo (loose)
# ---------------------------------------------------------------------------
def test_ds_rollout_follows_demo():
    """Euler rollout from the demo's first state passes within 0.4 of
    the demo's last state at some point along the trajectory. This is
    deliberately loose — Sec. III-A does not enforce SEDS-style global
    stability so we just want to confirm the learned field tracks the
    demo over its support."""
    set_global_seed(0)
    demo = make_letter_C_demo(n_points=80, duration=1.0)
    ds = GPDynamicalSystem(n_iter_default=200, lr=0.1).fit(
        demo["x"], demo["xdot"]
    )

    # Roll out for roughly one demo duration's worth of simulated time.
    traj, _ = ds.rollout(demo["x"][0], dt=0.025, n_steps=80)
    dists = np.linalg.norm(traj - demo["x"][-1], axis=1)
    assert dists.min() < 0.4, (
        f"DS rollout never gets near the demo endpoint: "
        f"min dist = {dists.min():.3f}"
    )


# ---------------------------------------------------------------------------
# Out-of-distribution uncertainty growth
# ---------------------------------------------------------------------------
def test_ds_uncertainty_grows_ood():
    """Predicted std at points far from the demo trajectory must be
    strictly greater than the mean predicted std on the demo itself —
    the epistemic-uncertainty growth claim of Figs. 5/6."""
    set_global_seed(0)
    demo = make_letter_C_demo(n_points=80, duration=1.0)
    ds = GPDynamicalSystem(n_iter_default=200, lr=0.1).fit(
        demo["x"], demo["xdot"]
    )

    _, std_on_demo = ds.predict(demo["x"], return_std=True)
    mean_std_on_demo = float(np.linalg.norm(std_on_demo, axis=1).mean())

    far_points = np.array([[3.0, 3.0], [-3.0, -3.0], [4.0, 0.0]])
    _, std_far = ds.predict(far_points, return_std=True)
    min_std_far = float(np.linalg.norm(std_far, axis=1).min())

    assert min_std_far > mean_std_on_demo, (
        f"Far-field std {min_std_far} is not greater than on-demo std "
        f"{mean_std_on_demo}"
    )


# ---------------------------------------------------------------------------
# Guard rail: GPDynamicalSystem must enforce zero-mean prior
# ---------------------------------------------------------------------------
def test_ds_rejects_non_zero_mean():
    """Sec. III-A explicitly requires a zero-mean prior; the constructor
    must reject ``mean="constant"`` to avoid silent regressions."""
    with pytest.raises(ValueError):
        GPDynamicalSystem(mean="constant")


@pytest.fixture(autouse=True)
def _silence_gpytorch_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
