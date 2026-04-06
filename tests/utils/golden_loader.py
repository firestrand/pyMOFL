"""Utility for loading golden JSONL test cases from the cec-benchmarks repository."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_golden_cases(
    year: str,
    func_id: int,
    dim: int,
    golden_root: str | Path,
) -> list[dict[str, Any]]:
    """Load golden test cases from JSONL.

    Parameters
    ----------
    year : str
        Suite year identifier, e.g. "CEC2014".
    func_id : int
        Function number (1-based).
    dim : int
        Dimensionality.
    golden_root : str or Path
        Root path to the cec-benchmarks/datasets/ directory.

    Returns
    -------
    list[dict]
        Each dict has keys: "func_id", "dim", "case", "x", "value".
        Returns empty list if the file doesn't exist.
    """
    golden_root = Path(golden_root)
    golden_dir = golden_root / "datasets" / year / f"func_{func_id}_D{dim}"
    golden_file = golden_dir / "golden.jsonl"

    if not golden_file.exists():
        return []

    cases: list[dict[str, Any]] = []
    with golden_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases
