"""Phase 3 — Sec. IV-A linear policy transportation: Fig. 3 (panels 1–3).

Generates a paired source/target 2D point cloud whose mapping is a
known rigid transform PLUS a small per-point non-linear perturbation
(so the linear transport leaves a visible residual — this motivates
Phase 4 / panel 4 of Fig. 3).

Produces a single figure with three side-by-side subplots:

1. *Distribution Match*  — source S vs target T with correspondence lines.
2. *Source Distribution* — a regular grid in the source frame.
3. *Linear Transformation* — that grid mapped through γ, with S and T
   overlaid for context.

Prints to stdout:

* recovered rotation matrix A
* its determinant
* centroid shift  T̄ − S̄
* mean residual   ‖T − γ(S)‖₂ / N

CLI (per CLAUDE.md):

* ``--seed``             RNG seed (default 0)
* ``--n_source_points``  number of paired source/target points (default 18)
* ``--out_dir``          figure directory (default ``reports/figures/``)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gpt_repro.transport import LinearTransport  # noqa: E402
from gpt_repro.utils import set_global_seed  # noqa: E402
from gpt_repro.viz import (  # noqa: E402
    plot_distribution_match,
    plot_grid_under_transform,
)


def _rot_2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _build_source_target(
    n: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Synthesize paired (S, T) with a known rotation+translation and a
    deterministic non-linear perturbation that makes the linear fit imperfect.

    Returns (S, T, R_true, t_true, perturbation_amplitude).
    """
    rng = np.random.default_rng(seed)
    # Source: roughly a unit-ish box.
    S = rng.uniform(low=-1.0, high=1.0, size=(n, 2))

    theta_true = 0.55  # ≈ 31.5°
    R = _rot_2d(theta_true)
    t = np.array([1.6, -0.4])

    # Deterministic non-linear perturbation: small sinusoid in S
    # coordinates. Because it depends only on S, two different runs
    # with the same seed and n produce identical T (reproducibility).
    nl_amp = 0.07
    perturb = nl_amp * np.stack(
        [np.sin(1.8 * S[:, 1]), np.cos(2.1 * S[:, 0])], axis=1
    )
    T = S @ R.T + t + perturb
    return S, T, R, t, nl_amp


def _save_metadata(fig: plt.Figure, *, section: str, seed: int, n_source: int) -> None:
    footer = (
        f"section={section}  seed={seed}  n_source_points={n_source}  "
        f"timestamp={_dt.datetime.now().isoformat(timespec='seconds')}"
    )
    fig.text(0.5, 0.005, footer, ha="center", fontsize=7, color="grey")


def make_figure(
    seed: int = 0,
    n_source_points: int = 18,
    out_dir: Path = _REPO_ROOT / "reports" / "figures",
    save: bool = True,
) -> dict:
    """Build Fig. 3 (panels 1–3) and return numerical results.

    This is split out from ``main`` so :mod:`smoke_phase3` can re-use it.
    """
    set_global_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    S, T, R_true, t_true, nl_amp = _build_source_target(n_source_points, seed)
    lt = LinearTransport().fit(S, T)
    A = lt.A
    det_A = float(np.linalg.det(A))
    centroid_shift = lt.T_bar - lt.S_bar
    residual = T - lt.transform(S)
    mean_residual = float(np.mean(np.linalg.norm(residual, axis=1)))

    if save:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
        # Panel 1
        plot_distribution_match(
            S, T, ax=axes[0],
            title="1. Distribution Match",
        )
        # Use a common source-frame grid extent for panels 2 and 3 based on
        # S's bounding box plus a margin, so the grids are visually
        # comparable.
        margin = 0.4
        x_range = (float(S[:, 0].min() - margin), float(S[:, 0].max() + margin))
        y_range = (float(S[:, 1].min() - margin), float(S[:, 1].max() + margin))

        # Panel 2: identity grid in source frame.
        plot_grid_under_transform(
            transform_fn=lambda X: X,
            x_range=x_range, y_range=y_range, n_grid=12, ax=axes[1],
            title="2. Source Distribution",
            overlay_points={"source": S},
        )
        # Panel 3: grid mapped through γ; overlay S and T.
        plot_grid_under_transform(
            transform_fn=lt.transform,
            x_range=x_range, y_range=y_range, n_grid=12, ax=axes[2],
            title="3. Linear Transformation (γ)",
            overlay_points={"source": S, "target": T},
        )

        fig.suptitle("Phase 3 — Sec. IV-A: Linear component γ of ϕ", y=0.98)
        fig.text(
            0.5, 0.93,
            "Panel 4 ('GP Transportation', Sec. IV-B) deferred to Phase 4.",
            ha="center", fontsize=8, color="0.4",
        )
        _save_metadata(fig, section="IV-A", seed=seed,
                       n_source=n_source_points)
        fig.tight_layout(rect=(0, 0.04, 1, 0.9))

        fig_path = out_dir / "phase3_fig3_partial.png"
        fig.savefig(fig_path, dpi=300)
        fig.savefig(fig_path.with_suffix(".pdf"))
        plt.close(fig)
    else:
        fig_path = None

    return {
        "A": A,
        "det_A": det_A,
        "centroid_shift": centroid_shift,
        "mean_residual": mean_residual,
        "reflection_fixed": lt.reflection_fixed,
        "R_true": R_true,
        "t_true": t_true,
        "perturbation_amplitude": nl_amp,
        "fig_path": fig_path,
        "n_source_points": n_source_points,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_source_points", type=int, default=18)
    parser.add_argument(
        "--out_dir", type=Path, default=_REPO_ROOT / "reports" / "figures"
    )
    parser.add_argument("--n_demos", type=int, default=0,
                        help="Unused in Phase 3; kept for CLI consistency.")
    args = parser.parse_args()

    res = make_figure(
        seed=args.seed,
        n_source_points=args.n_source_points,
        out_dir=args.out_dir,
        save=True,
    )

    np.set_printoptions(precision=4, suppress=True)
    print("Recovered rotation A:")
    print(res["A"])
    print(f"det(A)               : {res['det_A']:.6f}")
    print(f"centroid shift T̄-S̄  : {res['centroid_shift']}")
    print(f"reflection_fixed     : {res['reflection_fixed']}")
    print(f"mean residual ‖T-γ(S)‖₂/N : {res['mean_residual']:.6f}")
    print(f"true R (synthetic)   :")
    print(res["R_true"])
    print(f"true t (synthetic)   : {res['t_true']}")
    print(f"perturbation amplitude (non-linear) : {res['perturbation_amplitude']:.4f}")
    print(f"Saved figure: {res['fig_path']}")

    # Save numerical results.
    results_dir = _REPO_ROOT / "reports" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": 3,
        "section": "IV-A",
        "seed": res["seed"],
        "n_source_points": res["n_source_points"],
        "A": res["A"].tolist(),
        "det_A": res["det_A"],
        "centroid_shift": res["centroid_shift"].tolist(),
        "mean_residual": res["mean_residual"],
        "reflection_fixed": bool(res["reflection_fixed"]),
        "perturbation_amplitude": float(res["perturbation_amplitude"]),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    (results_dir / "phase3_linear.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved results: {results_dir / 'phase3_linear.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
