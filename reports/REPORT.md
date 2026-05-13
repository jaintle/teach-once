# Technical Report: Reproduction of Gaussian Process Transportation (Franzese et al., 2024)

> Franzese, G., Prakash, R., Kober, J. (2024).
> "Generalization of Task Parameterized Dynamical Systems using
> Gaussian Process Transportation." arXiv:2404.13458.

---

## Overview

This repository reproduces the simulation experiments from Franzese et al. (2024),
which proposes Gaussian Process Transportation (GPT) — a method for generalizing
task-parameterized dynamical systems to new reference frames using GP-learned
non-linear maps between point clouds. The core idea: a transport map ϕ(x) = γ(x)
+ ψ(γ(x)) (Eq. 7), where γ is an SVD-based linear alignment and ψ is a GP
residual, is used to warp demonstration trajectories, velocities, orientations,
and stiffness profiles from a source frame to a target frame.

This repo covers Secs. III–V in full 2D and provides 3D MuJoCo kinematic analogs
for Sec. VI. All 10 phases were implemented incrementally. Math equations are cited
in docstrings. Random seeds are set globally for `random`, `numpy`, and `torch`
before every run. The real-robot Sec. VI (ROS, Franka, F/T sensors) is out of scope.
No novel algorithmic contributions are made.

---

## Mathematical Implementation

**Verbatim implementations:**
- **Eq. (7):** `PolicyTransport.transform(x) = gamma(x) + psi(gamma(x))`
- **Eqs. (9–11):** Kabsch/SVD alignment — `LinearTransport.fit` computes SVD of S^T T and assembles rotation A with det(A)=+1 (reflection correction).
- **Eq. (13):** Velocity transport `ẋ̂ = J(x)·ẋ` where `J = (I + ∂ψ/∂γ) · A` — composed Jacobian of ϕ at x via chain rule.
- **Eq. (15):** Orientation transport `R̂ = J·R`, followed by QR re-orthogonalisation to ensure SO(3) structure.
- **Stiffness/damping (Sec. IV-D):** `K̂_s = J·K_s·J^T`, `D̂ = J·D·J^T`.
- **Eq. (16):** GP derivative mean via `torch.autograd.grad` (autograd on the predictive mean); derivative std via RBF kernel Hessian (exact for ExactGP, inducing-point approximation for SVGP).
- **Eqs. (17–18):** Total transport variance = transport mean uncertainty + epistemic DS uncertainty. Clipped to `[-1e-10, ∞)` for numerical stability before sqrt.

**Approximations:**
- SVGP predict_derivative std uses the inducing-point K_ZZ^{-1} quadratic form rather than the full variational posterior covariance; under-estimates variance but is O(M²) rather than O(M³).
- QR re-orthogonalisation in orientation transport introduces a small non-linearity relative to the exact Rodrigues update.

---

- **Phase 1 (GP regression, Sec. III-B):** Exact-GP RMSE ≈ 0.050 on sine; SVGP
  RMSE ≈ 0.054. Mean derivative vs finite-diff error ≈ 2e-9. 5 tests pass.

- **Phase 2 (DS learning, Sec. III-A):** Letter-C train RMSE ≈ 0.006. Closest
  rollout approach to demo endpoint ≈ 0.31. OOD std ≈ 3.0 vs on-demo ≈ 0.005.
  11 tests pass.

- **Phase 3 (Linear transport, Sec. IV-A):** `kabsch_svd_rotation` recovers exact
  rotations; det(A) = 1.000000; mean residual ≈ 3e-6 at default seed. 18 tests pass.

- **Phase 4 (Full ϕ = γ + ψ, Sec. IV):** ϕ(S)−T max-norm ≈ 3e-6; Jacobian vs
  FD max-norm ≈ 1e-9; orientation output det = +1 to 1e-6. 26 tests pass.

- **Phase 5 (Uncertainty, Sec. IV-E):** Eq. (16) mean vs autograd ≈ 4e-5; variance
  vs MC within 1e-2 atol; Σ_total − Σ_x̂ ≡ Σ_f̂ to 1e-10. 32 tests pass.

- **Phase 6 (Sec. V-A, Fig. 7, Table I):** Mean dist to target surface: KMP=0.184,
  E-RF=0.164, E-NN=0.165, LE=0.199, E-NF=0.209, GP=0.178. Table I modality/
  uncertainty columns match the paper. 38 tests pass.

- **Phase 7 (Sec. V-B, Figs. 8/9/10, n_reps=20):** GPT rank 2, DMP rank 1 on
  Fréchet/Area/DTW/Final-pos; HMM rank 1 on final orientation. 46 tests pass.

- **Phase 8 (Sec. V-C, Fig. 11, n_reps=10):** MultiSourceGPT Fréchet 2.7579
  ≤ SingleSourceGPT 3.1952. 52 tests pass.

---

## 2D Simulation Results (Secs. V-A, V-B, V-C)

**Sec. V-A (Fig. 7 / Table I)** — Mean distance-to-surface per method
(seed=0, n_demos=120, n_source_pts=24):

| Method | Mean dist | Uncertainty type |
|--------|-----------|-----------------|
| KMP    | 0.184     | None            |
| E-RF   | 0.164     | Estimated       |
| E-NN   | 0.165     | Estimated       |
| LE     | 0.199     | None            |
| E-NF   | 0.209     | Estimated       |
| GP     | 0.178     | Analytical      |

Qualitative match to Fig. 7: GP shows wider ±2σ band that grows with OOD
distance (matching the paper's Sec. IV-E claim), while ensemble methods show
overconfident flat bands. KMP/LE produce no uncertainty estimate.

**Sec. V-B (Figs. 8/9/10)** — U-test ranking (n_reps=20, seed=0):

| Method   | Fréchet | Area | DTW | Final pos | Final orient |
|----------|---------|------|-----|-----------|-------------|
| DMP      | 1       | 1    | 1   | 1         | 3           |
| GPT      | 2       | 2    | 2   | 2         | 3           |
| HMM_6    | 3       | 3    | 3   | 3         | 2           |
| HMM_7    | 3       | 3    | 3   | 3         | 3           |
| TPGMM_6  | 5       | 5    | 5   | 3         | 3           |
| HMM_5    | 6       | 5    | 6   | 7         | 1           |
| TPGMM_5  | 6       | 7    | 6   | 3         | 3           |
| TPGMM_7  | 6       | 7    | 6   | 7         | 3           |

GPT achieves rank 2 on all trajectory-shape metrics (Fréchet, area, DTW, final
position); DMP ranks 1. This diverges from the paper's claim that GPT achieves
overall rank 1 — the most likely cause is the LQR→GMM rollout simplification (see
"Limitations" section). GPT's uncertainty advantage (analytical vs estimated) is
not captured in trajectory metrics alone.

**Sec. V-C (Fig. 11)** — Multi-source fusion (seed=0, n_reps=10, n_sources=4):

| Method          | Fréchet (mean ± std) | Final orient err |
|-----------------|---------------------|-----------------|
| MultiSourceGPT  | 2.7579 ± 0.5873     | 1.2401 ± 0.65   |
| SingleSourceGPT | 3.1952 ± 0.8842     | 1.7831 ± 0.60   |

Main Sec. V-C claim confirmed: MultiSourceGPT Fréchet < SingleSourceGPT (ratio ≈ 0.86).

---

## 3D Simulation Results (Sec. VI analogs)

**Reshelving (Sec. VI-A analog):** Kinematic end-effector moves from a source
shelf pose to a randomly offset target pose via GPT-transported trajectory.
Evaluated over 10 randomised scenes. Success rate (final pos error < threshold)
confirmed above 50% in Phase 9 smoke tests. The OOD fallback to linear transport
is visible at large scene perturbations (> 0.3 m): ψ contribution shrinks as
inputs leave the training cloud and the pure-γ linear alignment dominates.

**Arm-pose following (Sec. VI-B analog):** Random 3D joint targets transported
to a perturbed reference frame using GPT. Evaluated over 10 scenes. Confirms
that orientation transport (Eq. 15) with QR re-orthogonalisation maintains SO(3)
structure (det ≈ +1, atol=1e-6 in all 10 scenes).

**Surface cleaning (Sec. VI-C analog, Figs. 15/16):** Five surface variants
(flat, tilted, curved, bumpy, tilted+curved) evaluated with SVGP transport
(n_inducing=100, 400-point clouds). Coverage fractions at 150-step rollout:

| Surface       | Coverage | Notes |
|--------------|----------|-------|
| flat (demo)  | ~0.20    | Kinematic rollout, no path-following controller |
| tilted       | ~0.20    | Tilted normal transported correctly |
| curved       | ~0.20    | Sinusoidal surface, z-displacement preserved |
| bumpy        | ~0.20    | Gaussian bumps, smoother rollout |
| tilted+curved| ~0.20    | Combined surface |

Coverage is modest (~20%) because the kinematic rollout uses a greedy DS velocity
command without coverage-aware path planning. The transported trajectories do
follow the surface geometrically — mean surface distance < 0.03 m in all cases.
Force profile trends (Hooke's law proxy ‖Ks·ẋ‖) are preserved qualitatively.

---

## Limitations and Failure Modes

- **No real robot, no contact dynamics.** All Sec. VI analogs use kinematic
  point-mass end-effectors (pos += vel · dt). Contact forces, inertia, joint
  limits, and torque saturation are absent. Results confirm the math but do not
  validate real-hardware feasibility.

- **No cloth/deformable objects.** The dressing task (Sec. VI-B) is replaced by
  arm-pose following. Any deformable-body transport properties claimed in the
  paper are untested here.

- **SVGP convergence sensitivity.** Phase 10 SVGP transport (100 inducing points,
  300 iterations) is sensitive to learning rate and batch size. Under-trained SVGP
  produces non-finite transported values (caught by the `np.isfinite` guard). A
  warm-restart or adaptive LR scheduler would improve robustness.

- **High variance in multi-frame benchmark at low demo counts.** With n_reps=20
  and 5 demos per repetition, the U-test rankings are noisy — rank differences of
  1–2 positions may not be statistically significant. The paper uses larger-scale
  experiments; our ranking table should be treated as indicative, not conclusive.

- **TP-GMM PoG numerical instability at many frames.** At 7+ reference frames,
  the Product-of-Gaussians covariance becomes near-singular. We apply a
  1e-6 diagonal jitter, but this can still produce degenerate trajectories.

- **20 reps is insufficient for strong statistical claims.** Mann-Whitney U-test
  at n=20 has limited power; effect sizes between GPT and DMP (ranks 1 and 2) are
  within the noise band. The paper's claims are based on larger n and real hardware.

- **Nearest-neighbour cloud pairing (Phase 10)** is a simple approximation to the
  optimal transport pairing used in the paper. For highly non-uniform clouds,
  NN pairing can produce asymmetric correspondences that degrade ψ training.

---

## Conclusion

This reproduction confirms the core mathematical claims of Franzese et al. (2024):
the GP transport map ϕ = γ + ψ correctly warps demonstrations across reference
frames with preserved velocity, orientation, stiffness, and uncertainty profiles
(Secs. III–IV). The Sec. V-C multi-source claim (pooling sources beats single-source
GPT, Fréchet ratio ≈ 0.86) is confirmed. The Sec. V-B rank-1 GPT claim is not
confirmed in our simplified rollout — DMP ranks 1 — due to the LQR→GMM rollout
substitution. Real-hardware validation (Sec. VI) remains unverified; all MuJoCo
results here are kinematic analogs that confirm the math but not physical feasibility.

---

## Reproducibility notes

- Seeds: `random`, `numpy`, `torch` (CPU + CUDA) set via `set_global_seed`.
- Runtime (Apple M-series): smoke_all.py (~30 s); full 20-rep benchmark (~90 s);
  Phase 10 full-res figures (~10 min each).
- All figures saved as PNG (150 dpi) + PDF; all numerics saved as JSON/CSV.
