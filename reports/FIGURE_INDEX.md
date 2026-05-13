# Figure Index

All output figures produced by this repository.

| File | Script | Paper Fig. | Description |
|------|--------|------------|-------------|
| `phase1_gp_demo.png` | `scripts/smoke_phase1.py` | N/A | Exact-GP and SVGP fit quality on 1D sine; includes derivative uncertainty |
| `phase2_letter_C_field.png` | `scripts/smoke_phase2.py` | N/A | GP velocity field learned from letter-C demonstrations |
| `phase2_cleaning_demo.png` | `scripts/smoke_phase2.py` | N/A | Cleaning raster-scan demo on 2D surface |
| `phase3_fig3_partial.png` | `scripts/figure3_linear.py` | Fig. 3 (panels 1–3) | Linear SVD transport γ: source cloud → target cloud with rotation A |
| `phase4_fig3_full.png` | `scripts/figure3_full.py` | Fig. 3 (all panels) | Full transport ϕ = γ + ψ: GP residual adds non-linear correction |
| `phase4_fig5_scheme.png` | `scripts/figure5_scheme.py` | Fig. 5 (scheme) | Schematic of the transport pipeline (demo, γ, ψ, transported demo) |
| `phase5_fig5_full.png` | `scripts/smoke_phase5.py` | Fig. 5 (full) | Full ϕ transport with ±2σ uncertainty band |
| `phase5_fig6_uncertainty.png` | `scripts/figure6_uncertainty.py` | Fig. 6 | Epistemic + transport uncertainty fields; OOD uncertainty growth |
| `phase6_fig7_comparison.png` | `scripts/figure7_cleaning_comparison.py` | Fig. 7 | Side-by-side comparison of GPT vs KMP/LE/E-RF/E-NN/E-NF on cleaning task |
| `phase7_fig8_qualitative.png` | `scripts/figure8_qualitative.py` | Fig. 8 | Qualitative multi-frame transport: source + 5/6/7-frame rollouts |
| `phase7_fig9_boxplots.png` | `scripts/figure9_boxplots.py` | Fig. 9 | Train-split metric boxplots (Fréchet, area, DTW) across methods |
| `phase7_fig10_test_boxplots.png` | `scripts/figure10_test_boxplots.py` | Fig. 10 | Test-split metric boxplots (final pos error, final orient error) |
| `phase8_fig11_multisource.png` | `scripts/smoke_phase8.py` | Fig. 11 | Multi-source single-target: MultiSourceGPT vs SingleSourceGPT rollouts |
| `phase9_reshelving_3d.png` | `scripts/smoke_phase9.py` | N/A (Sec. VI-A analog) | 3D reshelving rollout in MuJoCo kinematic env; source + transported demo |
| `phase9_armpose_3d.png` | `scripts/smoke_phase9.py` | N/A (Sec. VI-B analog) | 3D arm-pose following rollout; orientation preserved under transport |
| `phase9_reshelving_traj.png` | `scripts/smoke_phase9.py` | N/A | XYZ trajectory components over time for reshelving rollout |
| `phase9_armpose_traj.png` | `scripts/smoke_phase9.py` | N/A | XYZ trajectory components over time for arm-pose rollout |
| `phase10_fig15_cleaning.png` | `scripts/figure15_cleaning_surfaces.py` | Fig. 15 analog (Sec. VI-C) | 2-row 3D surface cleaning: rollout overlays (top) and point clouds (bottom) for 5 surface variants |
| `phase10_fig16_force.png` | `scripts/figure16_force_profile.py` | Fig. 16 analog (Sec. VI-C) | Force norm ‖Ks·ẋ‖ over time for 5 surface variants with demo overlay |
