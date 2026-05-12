# Gaussian Process Transportation — 2D Reproduction

Faithful reproduction of the 2D simulation experiments from:

> Franzese, G., Prakash, R., Kober, J. (2024).
> "Generalization of Task Parameterized Dynamical Systems using
> Gaussian Process Transportation." arXiv:2404.13458.

Sections covered: Sec. III-A/B (GP regression, DS learning), Sec. IV (policy
transportation — linear SVD, GP residual, Jacobian, uncertainty), Sec. V-A
(cleaning comparison, Fig. 7, Table I), Sec. V-B (multi-frame benchmark,
Figs. 8/9/10), Sec. V-C (multi-source single-target, Fig. 11). Real-robot
Sec. VI is explicitly out of scope.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

---

## Run all smoke tests

```bash
for i in 1 2 3 4 5 6 7 8; do python scripts/smoke_phase$i.py; done
pytest -q
```

---

## Reproduce figures

| Figure | Script | Output |
|--------|--------|--------|
| Fig. 3 (linear γ) | `python scripts/figure3_linear.py` | `reports/figures/phase3_fig3_linear.png` |
| Fig. 3 (full ϕ) | `python scripts/figure3_full.py` | `reports/figures/phase4_fig3_full.png` |
| Fig. 5 (scheme) | `python scripts/figure5_scheme.py` | `reports/figures/phase5_fig5_full.png` |
| Fig. 6 (uncertainty) | `python scripts/figure6_uncertainty.py` | `reports/figures/phase5_fig6_uncertainty.png` |
| Fig. 7 (cleaning) | `python scripts/figure7_cleaning_comparison.py` | `reports/figures/phase6_fig7_cleaning.png` |
| Fig. 8 (qualitative) | `python scripts/figure8_qualitative.py` | `reports/figures/phase7_fig8_qualitative.png` |
| Fig. 9 (boxplots) | `python scripts/figure9_boxplots.py` | `reports/figures/phase7_fig9_boxplots.png` |
| Fig. 10 (test boxplots) | `python scripts/figure10_test_boxplots.py` | `reports/figures/phase7_fig10_test_boxplots.png` |
| Fig. 11 (multi-source) | `python scripts/figure11_multisource.py --seed 0` | `reports/figures/phase8_fig11_multisource.png` |

---

## Run full benchmarks

```bash
python scripts/run_multiframe_benchmark.py --seed 0 --n_reps 20
python scripts/run_multisource_benchmark.py --seed 0 --n_reps 10
```

---

## Directory layout

```
src/gpt_repro/
  gp/              # Sec. III-B — Gaussian Process regression
  policies/        # Sec. III-A — DS learning from demonstrations
  transport/       # Sec. IV    — Policy Transportation (linear + nonlinear + Jacobian)
  baselines/       # Sec. V     — KMP, LE, E-RF, E-NN, E-NF, TP-GMM, HMM, DMP
  metrics/         # Sec. V-B   — Frechet, area, DTW, final pos/orient error, U-test
  viz/             # Figure reproduction utilities
  utils/           # Seeding, IO, config, geometry helpers
scripts/           # Entry-point scripts: figure_<N>.py, smoke_phase<N>.py
tests/             # pytest unit + integration tests
configs/           # YAML configs for each experiment
data/demonstrations/   # Generated 2D demo trajectories
reports/
  figures/         # All paper-figure reproductions
  results/         # CSV/NPZ of numerical results
  experiment_log.md
```

---

## Non-goals

- No real-robot code (no ROS, no franka_ros, no MuJoCo).
- No 3D Cartesian impedance control implementation.
- No AprilTag detection.
- No point-cloud SVGP for the cleaning task at 3D scale
  (paper Sec. VI-C); we only do the 2D version in Sec. V-A.
- No novel algorithmic contributions.
