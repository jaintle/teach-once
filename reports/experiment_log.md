# Experiment Log

One entry per phase, in the format required by `CLAUDE.md`.

---

## Phase 1 — GP regression module (Sec. III-B)

Date: 2026-05-11
Paper section(s) implemented: Sec. III-B (Eqs. 2, 3, 16; SVGP approximation).

Files added/changed:
- `src/gpt_repro/utils/seeding.py` — `set_global_seed`.
- `src/gpt_repro/utils/__init__.py` — re-exports `set_global_seed`.
- `src/gpt_repro/gp/exact_gp.py` — `ExactGPRegressor` (ARD-RBF, exact MLL fit,
  analytical variance derivative).
- `src/gpt_repro/gp/svgp.py` — `SVGPRegressor` (CholeskyVariational +
  VariationalELBO; derivative deferred to Phase 4).
- `src/gpt_repro/gp/__init__.py` — re-exports both regressors.
- `tests/test_gp.py` — 5 tests (sine fit, derivative vs finite-diff,
  SVGP fit, determinism, SVGP derivative-not-implemented guard).
- `scripts/smoke_phase1.py` — Phase-1 demo, emits PASS/FAIL line, saves
  `reports/figures/phase1_gp_demo.png|.pdf` and
  `reports/results/phase1_gp_demo.{json,csv}`.
- `reports/experiment_log.md` — this file.

What works:
- Exact GP fits noisy `sin(1.2x) + 0.3 cos(2.5x)` with test RMSE ≈ 0.050.
- SVGP fits the same with test RMSE ≈ 0.054 using 20 inducing points.
- Mean derivative (autograd) matches a central finite-difference of the
  GP mean to ≈ 2e-9 absolute on the smoke run.
- Variance/std derivative (analytical, Eq. 16) matches finite-difference
  to ≈ 2e-7 absolute on the smoke run.
- `pytest -q tests/test_gp.py` → 5 passed.
- Predictions are bit-identical across two runs sharing the same seed.

What was tricky:
- `gpytorch.kernels.ScaleKernel(RBFKernel(...))`'s `covar_module(X).to_dense()`
  is the cleanest path to the raw train kernel matrix; pulling `outputscale`
  and `base_kernel.lengthscale` separately is needed when reconstructing
  `k_star` and its derivative by hand.
- Numerical stability of the analytical variance derivative required
  float64 throughout and a small jitter on `K + σ_n² I` before Cholesky;
  also a `clamp_min(eps)` on `std` to avoid divide-by-zero at training
  points where the posterior variance can be machine-epsilon.
- gpytorch emits a benign `GPInputWarning` whenever you predict at the
  training set; silenced in tests via an autouse fixture.

Math / equation references implemented:
- Eq. (2)  posterior mean        → `ExactGPRegressor.predict`, `SVGPRegressor.predict`.
- Eq. (3)  posterior variance    → same methods (return_std path).
- Eq. (16) mean derivative       → autograd through gpytorch in
  `ExactGPRegressor.predict_with_derivative`.
- Eq. (16) variance derivative   → analytical, same method:
  `∂σ²/∂x_*_d = -2 k_*^T (K+σ_n²I)^{-1} ∂k_*/∂x_*_d`, then
  `∂σ/∂x = (2σ)^{-1} ∂σ²/∂x`.

Numerical sanity checks passed:
- Exact-GP RMSE on sine < 0.1.
- SVGP RMSE on sine < 0.15.
- Mean derivative vs central finite-diff within 1e-3 (atol).
- Out-of-domain posterior std grows >5× relative to in-domain at `x=0`.
- Two seeded fits produce bit-identical predictions (exact and SVGP).
- SVGP `predict_with_derivative` raises `NotImplementedError` as required.

Open questions / deferred work:
- SVGP derivative (Eq. 16 under the variational posterior) deferred until
  Phase 4 if the cleaning-surface SVGP variant of Sec. V-A needs it.
- No Jacobian-of-multi-output GP yet; Phase 4 (Sec. IV-B) is where the
  multi-output transportation GP and Jacobian-of-GP code will land.
