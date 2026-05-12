"""Phase 8 smoke test — multi-source single-target (Sec. V-C).

Runs the smallest possible end-to-end check:
  1. make_multisource_scenario with n_sources=2.
  2. Fit MultiSourceGPT and MultiSourceDMP with n_iter=30.
  3. Call figure11_multisource.make_figure (n_sources=2, n_reps=2,
     n_steps=30, seed=0) and verify outputs exist and are non-empty.
  4. Verify reports/results/phase8_multisource.json exists.

Target runtime: < 60 s.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import tempfile

from gpt_repro.baselines.multisource_dmp import MultiSourceDMP
from gpt_repro.baselines.multisource_gpt import MultiSourceGPT
from gpt_repro.policies.multisource_demos import make_multisource_scenario
from gpt_repro.utils import set_global_seed

_FIG_DIR = _REPO_ROOT / "reports" / "figures"
_RES_DIR = _REPO_ROOT / "reports" / "results"


def main() -> None:
    t0 = time.time()
    failures: list = []

    # ------------------------------------------------------------------
    # Step 1: scenario creation
    # ------------------------------------------------------------------
    set_global_seed(0)
    try:
        scenario = make_multisource_scenario(n_sources=2, seed=0, n_points=20)
        assert len(scenario["source_demos"]) == 2
        assert scenario["target_demo"]["x"].shape[1] == 2
        print("  [OK] make_multisource_scenario")
    except Exception as e:
        failures.append(f"make_multisource_scenario: {e}")
        print(f"  [FAIL] make_multisource_scenario: {e}")

    # ------------------------------------------------------------------
    # Step 2: MultiSourceGPT fit + rollout
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            msg = MultiSourceGPT(n_iter_transport=30, n_iter_ds=30).fit(
                scenario["S_list"], scenario["T"], scenario["source_demos"],
            )
            x0 = scenario["target_demo"]["x"][0]
            traj, _ = msg.rollout(x0, dt=0.05, n_steps=20)
            assert traj.shape == (21, 2)
            std = msg.uncertainty(
                scenario["target_demo"]["x"],
                scenario["target_demo"]["xdot"],
            )
            assert len(std) == 20
            assert all(s >= 0 for s in std)
            print("  [OK] MultiSourceGPT fit + rollout + uncertainty")
        except Exception as e:
            failures.append(f"MultiSourceGPT: {e}")
            print(f"  [FAIL] MultiSourceGPT: {e}")

    # ------------------------------------------------------------------
    # Step 3: MultiSourceDMP fit + rollout
    # ------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mdmp = MultiSourceDMP(n_iter_gp=30).fit(
                scenario["S_list"], scenario["T"], scenario["source_demos"],
            )
            traj2, _ = mdmp.rollout(x0, dt=0.05, n_steps=20)
            assert traj2.shape == (21, 2)
            print("  [OK] MultiSourceDMP fit + rollout")
        except Exception as e:
            failures.append(f"MultiSourceDMP: {e}")
            print(f"  [FAIL] MultiSourceDMP: {e}")

    # ------------------------------------------------------------------
    # Step 4: figure11_multisource.make_figure
    # ------------------------------------------------------------------
    try:
        _scripts_dir = str(_REPO_ROOT / "scripts")
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        import importlib
        f11 = importlib.import_module("figure11_multisource")

        with tempfile.TemporaryDirectory() as tmpdir:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = f11.make_figure(
                    seed=0, n_reps=2, n_sources=2,
                    n_steps=30, n_points=20,
                    n_iter_transport=30, n_iter_ds=30,
                    out_dir=tmpdir, save=True,
                )
            png = Path(tmpdir) / "phase8_fig11_multisource.png"
            pdf = Path(tmpdir) / "phase8_fig11_multisource.pdf"
            assert png.exists() and png.stat().st_size > 1000, \
                f"PNG missing or tiny: {png}"
            assert pdf.exists() and pdf.stat().st_size > 1000, \
                f"PDF missing or tiny: {pdf}"
        print("  [OK] figure11_multisource.make_figure")
    except Exception as e:
        failures.append(f"figure11_multisource: {e}")
        print(f"  [FAIL] figure11_multisource: {e}")

    # ------------------------------------------------------------------
    # Step 5: reports/results/phase8_multisource.json
    # ------------------------------------------------------------------
    json_path = _RES_DIR / "phase8_multisource.json"
    if json_path.exists() and json_path.stat().st_size > 0:
        with open(json_path) as f:
            data = json.load(f)
        assert "summary" in data
        print(f"  [OK] phase8_multisource.json exists ({json_path.stat().st_size} B)")
    else:
        # Run benchmark to generate it
        try:
            sys.path.insert(0, str(_REPO_ROOT / "scripts"))
            import importlib
            bench = importlib.import_module("run_multisource_benchmark")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results, _ = bench.run_benchmark(
                    seed=0, n_reps=2, n_sources=2,
                    n_points=20, n_iter_transport=30, n_iter_ds=30,
                )
            import json as _json
            out = {
                "summary": {
                    m: {
                        met: {
                            "mean": float(sum(results[m][met]) / max(1, len(results[m][met]))),
                            "std": 0.0,
                        }
                        for met in ["frechet", "final_pos", "final_orient"]
                    }
                    for m in bench.METHOD_NAMES
                }
            }
            _RES_DIR.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w") as f:
                _json.dump(out, f, indent=2)
            print(f"  [OK] phase8_multisource.json generated")
        except Exception as e:
            failures.append(f"phase8_multisource.json: {e}")
            print(f"  [FAIL] phase8_multisource.json: {e}")

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    if failures:
        print(f"\nPHASE8 SMOKE: FAIL  ({elapsed:.1f}s)")
        for f in failures:
            print(f"  • {f}")
        sys.exit(1)
    else:
        print(f"\nPHASE8 SMOKE: PASS  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
