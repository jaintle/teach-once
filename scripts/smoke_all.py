#!/usr/bin/env python
"""smoke_all.py — Run all 10 phase smoke tests and report pass/fail.

Each smoke is run via subprocess with a 120-second timeout.
Failures are printed but do not stop the run.
Output is also saved to reports/results/smoke_all_output.txt.

Usage:
    python scripts/smoke_all.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _REPO_ROOT / "reports" / "results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SMOKES = [
    ("Phase 1",  "scripts/smoke_phase1.py"),
    ("Phase 2",  "scripts/smoke_phase2.py"),
    ("Phase 3",  "scripts/smoke_phase3.py"),
    ("Phase 4",  "scripts/smoke_phase4.py"),
    ("Phase 5",  "scripts/smoke_phase5.py"),
    ("Phase 6",  "scripts/smoke_phase6.py"),
    ("Phase 7",  "scripts/smoke_phase7.py"),
    ("Phase 8",  "scripts/smoke_phase8.py"),
    ("Phase 9",  "scripts/smoke_phase9.py"),
    ("Phase 10", "scripts/smoke_phase10.py"),
]

_PYTHON = str(_REPO_ROOT / ".venv" / "bin" / "python")


def run_smoke(label: str, script: str) -> tuple[bool, float, str]:
    """Run a smoke script.  Returns (passed, elapsed_s, output)."""
    script_path = _REPO_ROOT / script
    if not script_path.exists():
        return False, 0.0, f"SCRIPT NOT FOUND: {script}"

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            [_PYTHON, str(script_path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(_REPO_ROOT),
        )
        elapsed = time.monotonic() - t0
        combined = result.stdout + result.stderr
        passed = result.returncode == 0
        return passed, elapsed, combined
    except subprocess.TimeoutExpired as e:
        elapsed = time.monotonic() - t0
        return False, elapsed, f"TIMEOUT after {elapsed:.0f}s"
    except Exception as e:
        elapsed = time.monotonic() - t0
        return False, elapsed, f"ERROR: {e}"


def main():
    lines: list[str] = []

    def emit(s: str = ""):
        print(s)
        lines.append(s)

    emit("=" * 60)
    emit("smoke_all.py — Gaussian Process Transportation reproduction")
    emit("=" * 60)
    emit()

    results = []
    for label, script in SMOKES:
        emit(f"[{label}] {script} ...")
        passed, elapsed, output = run_smoke(label, script)
        status = "PASS" if passed else "FAIL"
        emit(f"  → {status}  ({elapsed:.1f}s)")
        if not passed:
            # Print last 10 lines of output for diagnosis
            last_lines = [l for l in output.strip().splitlines() if l.strip()][-10:]
            for l in last_lines:
                emit(f"    {l}")
        results.append((label, passed, elapsed))
        emit()

    # Summary
    n_pass = sum(1 for _, p, _ in results if p)
    n_total = len(results)
    emit("=" * 60)
    emit(f"SUMMARY: {n_pass}/{n_total} smoke tests passed")
    emit("=" * 60)
    emit()
    for label, passed, elapsed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        emit(f"  {status}  {label:<12} ({elapsed:.1f}s)")
    emit()
    emit(f"Final: {n_pass}/{n_total} smoke tests passed")

    # Save output
    out_path = _RESULTS_DIR / "smoke_all_output.txt"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nOutput saved to {out_path}")

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
