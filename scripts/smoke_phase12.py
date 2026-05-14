"""
Smoke test for Phase 12 — MuJoCo visualisation & animation scripts.

Checks:
1. All 3 envs instantiate and render a (480,480,3) uint8 frame.
2. animate_reshelving.py runs in fast mode and produces a non-empty GIF.
3. animate_armpose.py runs in fast mode and produces a non-empty GIF.
4. animate_cleaning.py runs in fast mode and produces a non-empty GIF.

Run:
    python scripts/smoke_phase12.py

Exit code 0 = all PASS, non-zero = at least one FAIL.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gpt_repro.envs.reshelving_env import ReshelvingEnv  # noqa: E402
from gpt_repro.envs.armpose_env import ArmPoseEnv  # noqa: E402
from gpt_repro.envs.cleaning_env import SurfaceCleaningEnv  # noqa: E402
from gpt_repro.policies.surfaces_3d import SurfaceConfig  # noqa: E402

PYTHON = sys.executable
SCRIPTS = Path(__file__).resolve().parent


def _check(condition: bool, label: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


# ── 1. Render shape tests ─────────────────────────────────────────────────────

def check_renders() -> bool:
    print("\n=== 1. Render resolution check ===")
    ok = True
    envs_and_labels = [
        (lambda: ReshelvingEnv(render_mode="rgb_array"), "ReshelvingEnv"),
        (lambda: ArmPoseEnv(render_mode="rgb_array"), "ArmPoseEnv"),
        (
            lambda: SurfaceCleaningEnv(
                SurfaceConfig(kind="flat", center=np.array([0.5, 0.0, 0.5])),
                render_mode="rgb_array",
            ),
            "SurfaceCleaningEnv",
        ),
    ]
    for factory, label in envs_and_labels:
        try:
            env = factory()
            env.reset(seed=0)
            frame = env.render()
            env.close()
            ok &= _check(
                frame is not None and frame.shape == (480, 480, 3) and frame.dtype == np.uint8,
                f"{label} render() → (480,480,3) uint8",
            )
        except Exception as exc:
            ok &= _check(False, f"{label} render() raised: {exc}")
    return ok


# ── 2. Animation script tests ─────────────────────────────────────────────────

def _run_script(script_name: str, extra_args: list[str], out_dir: Path) -> tuple[bool, str]:
    """Run an animation script and return (success, output)."""
    cmd = [
        PYTHON,
        str(SCRIPTS / script_name),
        "--seed", "0",
        "--fps", "10",
        "--out_dir", str(out_dir),
        "--fast",
    ] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout + result.stderr


def check_animation(script_name: str, gif_name: str, extra_args: list[str] | None = None) -> bool:
    print(f"\n=== {script_name} (fast mode) ===")
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        ok, output = _run_script(script_name, extra_args or [], out_dir)
        if not ok:
            print("  Script stderr/stdout:")
            for line in output.splitlines()[-20:]:
                print(f"    {line}")
        gif_path = out_dir / gif_name
        gif_ok = gif_path.exists() and gif_path.stat().st_size > 1024
        ok &= _check(ok, f"exit code 0")
        ok &= _check(gif_ok, f"{gif_name} exists and > 1 KB")
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    results = []
    results.append(check_renders())
    results.append(check_animation("animate_reshelving.py", "reshelving_rollout.gif"))
    results.append(check_animation("animate_armpose.py", "armpose_rollout.gif"))
    results.append(check_animation("animate_cleaning.py", "cleaning_rollout.gif"))

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{'='*50}")
    print(f"Phase 12 smoke: {n_pass}/{n_total} sections PASS")
    if n_pass == n_total:
        print("ALL PASS")
    else:
        print("SOME FAILURES — see above")
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
