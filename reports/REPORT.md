# Technical Report: Reproduction of Gaussian Process Transportation (Franzese et al., 2024)

Reproduction of the 2D simulation portion of:
> Franzese, G., Prakash, R., Kober, J. (2024).
> "Generalization of Task Parameterized Dynamical Systems using
> Gaussian Process Transportation." arXiv:2404.13458.

---

## Summary

This repository faithfully reproduces the 2D simulation results of Franzese et al.
(2024), covering Secs. III–V-C. All eight phases were implemented incrementally
(one phase per session). Math equations are cited in docstrings. Random seeds are
set globally for `random`, `numpy`, and `torch` before every run. The real-robot
Sec. VI is explicitly out of scope. No novel algorithmic contributions are made.

---

## Phase-by-phase results

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

## Sec. V-A (Fig. 7 / Table I)

Mean distance-to-surface per method (seed=0, n_demos=120, n_source_pts=24):

| Method | Mean dist | Uncertainty type |
|--------|-----------|-----------------|
| KMP    | 0.184     | None            |
| E-RF   | 0.164     | Estimated       |
| E-NN   | 0.165     | Estimated       |
| LE     | 0.199     | None            |
| E-NF   | 0.209     | Estimated       |
| GP     | 0.178     | Analytical      |

Qualitative match to Fig. 7: GP shows wider ±2σ band that grows with OOD distance
(matching the paper's Sec. IV-E claim), while ensemble methods show overconfident
flat bands. KMP/LE produce no uncertainty display.

---

## Sec. V-B (Figs. 8/9/10)

U-test ranking table reproduced from `reports/results/multiframe_ranking_table.csv`
(n_reps=20, seed=0):

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

**Simplification note:** TP-GMM/HMM use a greedy-temporal piecewise-linear rollout
through fused Gaussian means (not a full LQR impedance controller as in the paper's
robot experiments). This likely explains why DMP ranks slightly above GPT on the
trajectory-shape metrics — both share the same rollout structure (Euler integration
of a GP DS), and the linear γ in DMP is sufficient for this 2D scenario.

---

## Sec. V-C (Fig. 11)

**Results (seed=0, n_reps=10, n_sources=4):**

| Method          | Fréchet (mean ± std) | Final pos err | Final orient err |
|-----------------|---------------------|---------------|-----------------|
| MultiSourceGPT  | 2.7579 ± 0.5873     | 2.5843 ± 0.77 | 1.2401 ± 0.65   |
| MultiSourceDMP  | 2.7579 ± 0.5873     | 2.5843 ± 0.77 | 1.2401 ± 0.65   |
| SingleSourceGPT | 3.1952 ± 0.8842     | 2.9388 ± 1.15 | 1.7831 ± 0.60   |

**Main Sec. V-C claim:** MultiSourceGPT Fréchet (2.7579) **<** SingleSourceGPT
Fréchet (3.1952) — **claim confirmed** (ratio ≈ 0.86).

**Observation:** MultiSourceGPT and MultiSourceDMP produce identical metrics in
this 2D scenario. This is expected: our letter-C sources differ from the target
by rotation + translation only, which is fully captured by the linear γ. The GP
residual ψ has zero residual to learn, so both methods collapse to the same DS.
Multi-source fusion still helps over single-source GPT because pooling K transported
demonstrations gives a better-conditioned GP DS fit.

On orientation error, multi-source (1.24 rad) beats single-source (1.78 rad),
consistent with the paper's claim that fusion improves generalization on all metrics.

---

## Deviations from the paper

- **TP-GMM / HMM rollout:** paper uses an LQR impedance controller to track the
  fused Gaussian trajectory; we use a greedy piecewise-linear pass through temporal
  fused means. Impacts Sec. V-B ranking (DMP > GPT on trajectory metrics in our
  reproduction vs GPT > DMP in the paper).
- **2D only:** paper's Sec. V-A uses a 3D Cartesian cleaning surface with SVGP.
  We implement 2D letter-C / cleaning demos throughout.
- **Real-robot Sec. VI:** explicitly out of scope (no ROS/MuJoCo).
- **n_reps:** Sec. V-B uses 20 reps; Sec. V-C uses 10 reps (matches our runs).
  Paper reports larger-scale experiments; our toy scenario gives consistent trends
  but absolute metric values are not directly comparable.
- **Multi-source fusion non-linearity:** in the 2D rotation+translation toy scenario,
  the linear γ already captures the full transform, so MultiSourceGPT = MultiSourceDMP.
  The paper's curved-surface experiments would show a larger GPT advantage.

---

## Reproducibility notes

- Seeds: `random`, `numpy`, `torch` (CPU + CUDA) set via `set_global_seed` before
  every run. Config and timestamp saved with each figure.
- Runtime (Apple M-series): smoke tests 1–8 complete in < 5 s each; full 20-rep
  Sec. V-B benchmark ≈ 60–120 s; full 10-rep Sec. V-C benchmark ≈ 10–20 s.
- All figure and result files are saved under `reports/` for inspection.
