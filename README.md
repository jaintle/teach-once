# Gaussian Process Transportation — Paper Reproduction

Reproduction of:
> Franzese, G., Prakash, R., Kober, J. (2024).
> "Generalization of Task Parameterized Dynamical Systems using
> Gaussian Process Transportation." arXiv:2404.13458

---

## What is reproduced

| Paper Section | Content | Status |
|---|---|---|
| Sec. III-B | GP regression (exact + SVGP) | ✅ Full |
| Sec. III-A | DS learning from demonstrations | ✅ Full |
| Sec. IV-A | Linear transport via SVD (Eqs. 8–11) | ✅ Full |
| Sec. IV-B | Non-linear GP transportation (Eq. 12) | ✅ Full |
| Sec. IV-C | Velocity transport via Jacobian (Eq. 13) | ✅ Full |
| Sec. IV-D | Orientation + stiffness transport (Eqs. 14–15) | ✅ Full |
| Sec. IV-E | Uncertainty propagation (Eqs. 16–18) | ✅ Full |
| Sec. V-A | 2D surface cleaning comparison + Fig. 7 | ✅ Full |
| Sec. V-B | Multi-frame benchmark + Figs. 8–10 | ⚠️ Simplified |
| Sec. V-C | Multi-source single-target + Fig. 11 | ✅ Full |
| Sec. VI-A | Reshelving (3D MuJoCo kinematic analog) | ✅ Analog |
| Sec. VI-B | Dressing → arm-pose following (3D analog) | ✅ Analog |
| Sec. VI-C | Surface cleaning (3D SVGP, Figs. 15–16 analog) | ✅ Analog |

### What was simplified

**Sec. V-B:** The paper uses HMM-LQR rollout for TP-GMM and HMM baselines;
this repo replaces LQR with greedy GMM following. TP-GMM is susceptible to
numerical instability at high frame counts. U-test shows DMP rank-1 on final
position and GPT rank-1 on Fréchet/area/DTW — diverges from the paper's
rank-1 GPT overall, likely due to the LQR→GMM simplification.

**Sec. VI (all robot tasks):** Replaced by MuJoCo kinematic point-mass
end-effector (pos += vel · dt, no contact/inertia). Dressing → arm-pose
following (no cloth). Surface cleaning force estimation uses Hooke's law proxy
rather than real F/T sensor. Cloud pairing uses nearest-neighbour rather than
optimal transport.

---

## Setup

Requires **Python 3.11**.

```bash
git clone <repo>
cd gpt-paper-repro
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

---

## Reproduce all 2D figures (Secs. III–V)

```bash
# GP regression quality check
python scripts/smoke_phase1.py
# Output: reports/figures/phase1_gp_demo.png

# DS learning demo
python scripts/smoke_phase2.py
# Output: reports/figures/phase2_letter_C_field.png
#         reports/figures/phase2_cleaning_demo.png

# Fig. 3 (partial) — Linear transportation
python scripts/figure3_linear.py --seed 0
# Output: reports/figures/phase3_fig3_partial.png

# Fig. 3 (full) — GP transportation
python scripts/figure3_full.py --seed 0
# Output: reports/figures/phase4_fig3_full.png

# Fig. 5 scheme — Transport overview
python scripts/figure5_scheme.py --seed 0
# Output: reports/figures/phase4_fig5_scheme.png

# Fig. 5 (full) — Uncertainty (smoke generates)
python scripts/smoke_phase5.py
# Output: reports/figures/phase5_fig5_full.png

# Fig. 6 — Uncertainty propagation
python scripts/figure6_uncertainty.py --seed 0
# Output: reports/figures/phase5_fig6_uncertainty.png

# Fig. 7 — 2D surface cleaning comparison
python scripts/figure7_cleaning_comparison.py --seed 0 --n_demos 5
# Output: reports/figures/phase6_fig7_comparison.png

# Figs. 8–10 — Multi-frame benchmark (run benchmark first)
python scripts/run_multiframe_benchmark.py --seed 0 --n_reps 20
python scripts/figure8_qualitative.py --seed 0
# Output: reports/figures/phase7_fig8_qualitative.png
python scripts/figure9_boxplots.py --seed 0
# Output: reports/figures/phase7_fig9_boxplots.png
python scripts/figure10_test_boxplots.py --seed 0
# Output: reports/figures/phase7_fig10_test_boxplots.png

# Fig. 11 — Multi-source single-target
python scripts/smoke_phase8.py --seed 0
# Output: reports/figures/phase8_fig11_multisource.png
```

---

## Reproduce all 3D simulation figures (Sec. VI analogs)

```bash
# Phase 9 — Reshelving + arm-pose 3D rollout
python scripts/smoke_phase9.py
# Output: reports/figures/phase9_reshelving_3d.png
#         reports/figures/phase9_armpose_3d.png

# Phase 10 — Surface cleaning (Figs. 15/16 analogs)
# --fast: n_pts=100, n_inducing=20 (~2 min)
python scripts/figure15_cleaning_surfaces.py --seed 0 --fast
# Output: reports/figures/phase10_fig15_cleaning.png

python scripts/figure16_force_profile.py --seed 0 --fast
# Output: reports/figures/phase10_fig16_force.png

# Full resolution (~10 min):
python scripts/figure15_cleaning_surfaces.py --seed 0
python scripts/figure16_force_profile.py --seed 0
```

---

## Run all tests

```bash
pytest -q                    # 77 tests
pytest -q -m "not slow"      # skip slow regression tests
```

---

## Run all smoke tests

```bash
python scripts/smoke_all.py
# Output: reports/results/smoke_all_output.txt
```

---

## Directory structure

```
src/gpt_repro/
  gp/          # Sec. III-B: GP regression (ExactGPRegressor, SVGPRegressor)
  transport/   # Sec. IV:    Policy transportation math (linear, nonlinear, uncertainty)
  policies/    # Sec. III-A: DS learning, demo generators, 3D surface generators
  baselines/   # Sec. V:     KMP, LE, ensembles (RF/NN/NF), TP-GMM, HMM, DMP
  metrics/     # Sec. V-B:   Fréchet, area, DTW, final pos/orient error, U-test
  viz/         # Figures: vector fields, transport diagrams, uncertainty, 3D rollouts
  envs/        # Sec. VI analogs: MuJoCo kinematic environments
  utils/       # Seeding, IO helpers

scripts/       # Figure scripts (figure<N>_*.py) and smoke tests (smoke_phase<N>.py)
tests/         # 77 pytest unit + integration tests
configs/       # YAML experiment configs
data/          # Generated 2D demonstration trajectories
reports/
  figures/     # All reproduced figures (PNG + PDF)
  results/     # CSV/JSON numerical results per phase
  REPORT.md    # Technical report
  FIGURE_INDEX.md  # Figure → script → paper mapping
  experiment_log.md  # Per-phase implementation notes
```

---

## Citation

```bibtex
@article{franzese2024gpt,
  title={Generalization of Task Parameterized Dynamical Systems
         using Gaussian Process Transportation},
  author={Franzese, Giovanni and Prakash, Ravi and Kober, Jens},
  journal={arXiv preprint arXiv:2404.13458},
  year={2024}
}
```
