"""Sec. V-A Table I — Method / Velocity Generalization / Uncertainty.

Builds the table directly from the baselines' class attributes so it
stays in sync with the implementation. Prints an ASCII version and
saves a CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from gpt_repro.baselines import BASELINE_NAMES, BASELINES


def build_table1() -> List[Dict[str, str]]:
    """Read class attributes of every baseline and return Table I as a
    list of ordered-dict-like rows.

    Columns: ``Method`` / ``Velocity Gen.`` / ``Transp. Uncertainty``.
    """
    rows: List[Dict[str, str]] = []
    for key, cls in BASELINES.items():
        rows.append({
            "Method": BASELINE_NAMES[key],
            "Velocity Gen.": "Yes" if cls.has_velocity_generalization else "No",
            "Transp. Uncertainty": cls.uncertainty_type.capitalize(),
        })
    return rows


def format_ascii(rows: List[Dict[str, str]]) -> str:
    """Format a list-of-dicts as a borderless ASCII table."""
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(r[c]) for r in rows)) for c in cols}
    sep = "  ".join("-" * widths[c] for c in cols)
    header = "  ".join(f"{c:<{widths[c]}}" for c in cols)
    body = "\n".join(
        "  ".join(f"{r[c]:<{widths[c]}}" for c in cols) for r in rows
    )
    return f"{header}\n{sep}\n{body}"


def save_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(r[c] for c in cols) + "\n")


def print_and_save(out_path: Path) -> List[Dict[str, str]]:
    rows = build_table1()
    print(format_ascii(rows))
    save_csv(rows, out_path)
    return rows


if __name__ == "__main__":  # pragma: no cover
    from pathlib import Path as _P
    repo_root = _P(__file__).resolve().parents[3]
    print_and_save(repo_root / "reports" / "results" / "table1.csv")
