"""Unit tests for the GP module (Sec. III-B, Eqs. 2/3/16).

Covers:

* Exact GP: fit-to-sine RMSE, out-of-domain variance increase.
* Exact GP: analytical mean derivative matches central finite differences.
* SVGP: fit-to-sine RMSE (looser tolerance).
* Seeding: two seeded GP fits produce bit-identical predictions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from gpt_repro.gp import ExactGPRegressor, SVGPRegressor
from gpt_repro.utils import set_global_seed


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _sine_dataset(n: int = 40, noise_std: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.linspace(-3.0, 3.0, n).reshape(-1, 1)
    y = np.sin(X[:, 0]) + noise_std * rng.standard_normal(n)
    return X, y


# --------------------------------------------------------------------------
# Exact GP fit quality
# --------------------------------------------------------------------------
def test_exact_gp_fits_sine():
    """Exact GP fits noisy sin(x) with RMSE < 0.1, and variance grows away
    from the training support — basic sanity for Eqs. (2) and (3)."""
    set_global_seed(0)
    X, y = _sine_dataset(n=40, noise_std=0.05, seed=0)
    gp = ExactGPRegressor(n_iter_default=200, lr=0.1).fit(X, y)

    X_test = np.linspace(-3.0, 3.0, 60).reshape(-1, 1)
    mean, std = gp.predict(X_test)
    rmse = float(np.sqrt(np.mean((mean - np.sin(X_test[:, 0])) ** 2)))
    assert rmse < 0.1, f"Exact GP RMSE too high: {rmse}"

    # Variance must increase well outside the training support.
    _, std_in = gp.predict(np.array([[0.0]]))
    _, std_far = gp.predict(np.array([[8.0]]))
    assert std_far[0] > std_in[0] * 5, (
        f"Variance failed to grow out of domain: in={std_in[0]}, far={std_far[0]}"
    )


# --------------------------------------------------------------------------
# Eq. (16) — analytical mean derivative vs central finite difference
# --------------------------------------------------------------------------
def test_exact_gp_derivative_matches_finite_diff():
    """The autograd-derived mean derivative (Eq. 16) must match a central
    finite-difference of the GP mean within atol=1e-3 at 10 random points."""
    set_global_seed(0)
    X, y = _sine_dataset(n=40, noise_std=0.05, seed=0)
    gp = ExactGPRegressor(n_iter_default=200, lr=0.1).fit(X, y)

    rng = np.random.default_rng(123)
    X_test = rng.uniform(-2.5, 2.5, size=(10, 1))

    _, _, dmean_dx, dstd_dx = gp.predict_with_derivative(X_test)

    h = 1e-4
    mean_plus, std_plus = gp.predict(X_test + h)
    mean_minus, std_minus = gp.predict(X_test - h)
    fd_mean = (mean_plus - mean_minus) / (2.0 * h)
    fd_std = (std_plus - std_minus) / (2.0 * h)

    np.testing.assert_allclose(
        dmean_dx[:, 0], fd_mean, atol=1e-3, rtol=0,
        err_msg="GP mean derivative (Eq. 16) disagrees with finite difference.",
    )
    # The std derivative is also analytical — make sure it agrees too.
    np.testing.assert_allclose(
        dstd_dx[:, 0], fd_std, atol=1e-3, rtol=0,
        err_msg="GP std derivative (Eq. 16) disagrees with finite difference.",
    )


# --------------------------------------------------------------------------
# SVGP fit quality
# --------------------------------------------------------------------------
def test_svgp_fits_sine():
    """SVGP fits noisy sin(x) with RMSE < 0.15 — looser bound than exact."""
    set_global_seed(0)
    X, y = _sine_dataset(n=200, noise_std=0.05, seed=0)
    svgp = SVGPRegressor(
        n_inducing=32, n_iter_default=400, lr=0.05, batch_size=128
    ).fit(X, y)

    X_test = np.linspace(-3.0, 3.0, 80).reshape(-1, 1)
    mean, _ = svgp.predict(X_test)
    rmse = float(np.sqrt(np.mean((mean - np.sin(X_test[:, 0])) ** 2)))
    assert rmse < 0.15, f"SVGP RMSE too high: {rmse}"


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
def test_seeding_is_deterministic():
    """Two seeded runs of the GP pipeline must produce identical predictions."""
    X, y = _sine_dataset(n=40, noise_std=0.05, seed=0)
    X_test = np.linspace(-3.0, 3.0, 30).reshape(-1, 1)

    set_global_seed(42)
    gp1 = ExactGPRegressor(n_iter_default=80, lr=0.1).fit(X, y)
    m1, s1 = gp1.predict(X_test)

    set_global_seed(42)
    gp2 = ExactGPRegressor(n_iter_default=80, lr=0.1).fit(X, y)
    m2, s2 = gp2.predict(X_test)

    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_array_equal(s1, s2)

    # SVGP must also be deterministic — its inducing-point init and minibatch
    # shuffling both rely on the seeded RNG.
    set_global_seed(7)
    svgp1 = SVGPRegressor(n_inducing=16, n_iter_default=50, lr=0.05).fit(X, y)
    p1, _ = svgp1.predict(X_test)
    set_global_seed(7)
    svgp2 = SVGPRegressor(n_inducing=16, n_iter_default=50, lr=0.05).fit(X, y)
    p2, _ = svgp2.predict(X_test)
    np.testing.assert_array_equal(p1, p2)


# --------------------------------------------------------------------------
# Guard rail: SVGP derivatives are deferred to Phase 4.
# --------------------------------------------------------------------------
def test_svgp_derivative_raises_not_implemented():
    """Calling predict_with_derivative on SVGP must fail loudly in Phase 1."""
    set_global_seed(0)
    X, y = _sine_dataset(n=40, seed=0)
    svgp = SVGPRegressor(n_inducing=8, n_iter_default=5, lr=0.05).fit(X, y)
    with pytest.raises(NotImplementedError):
        svgp.predict_with_derivative(np.array([[0.0]]))


@pytest.fixture(autouse=True)
def _silence_gpytorch_warnings():
    """gpytorch warns when we predict at training inputs; harmless here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
