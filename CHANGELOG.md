# Changelog

## v1.0.0 — Initial public release

### Paper reproduction (Python)
- Full TP-GPT algorithm: Secs. III–V (Franzese et al. 2024)
- GP regression (exact + SVGP), Eqs. (2),(3),(16)
- Linear transport via SVD Kabsch, Eqs. (8)–(11)
- Nonlinear GP transportation, Eq. (12)
- Velocity, orientation, stiffness, damping transport,
  Eqs. (13)–(15)
- Transportation + epistemic uncertainty, Eqs. (16)–(18)
- All 6 baselines: KMP, LE, E-RF, E-NN, E-NF, GP
- Multi-frame benchmark (Sec. V-B): TP-GMM, HMM, DMP, GPT
- 3D MuJoCo kinematic analog: reshelving, arm-pose, cleaning

### Interactive web demo
- Three.js 3D scene with Franka Panda arm (DLS IK)
- Pure-JavaScript TP-GPT inference (no server)
- Three interactive modes: reshelving, cleaning, arm-pose
- Pre-computed GIF fallback
- Deployed on GitHub Pages

### Known simplifications vs paper
- Sec. V-B: LQR replaced by greedy GMM rollout
- Sec. VI: Impedance control replaced by kinematic IK
- Sec. VI-B: Dressing replaced by arm-pose following
