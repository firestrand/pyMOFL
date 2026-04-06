"""Tests for golden_loader utility."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.utils.golden_loader import load_golden_cases

GOLDEN_ROOT = os.environ.get(
    "CEC_BENCHMARKS_PATH",
    str(Path(__file__).resolve().parents[2].parent / "cec-benchmarks"),
)


class TestLoadGoldenCases:
    """Test golden JSONL loading."""

    def test_returns_empty_for_nonexistent_path(self, tmp_path):
        """Returns empty list when golden file doesn't exist."""
        cases = load_golden_cases("CEC9999", 1, 10, str(tmp_path))
        assert cases == []

    def test_loads_valid_jsonl(self, tmp_path):
        """Correctly loads JSONL with expected structure."""
        golden_dir = tmp_path / "datasets" / "CEC_TEST" / "func_1_D10"
        golden_dir.mkdir(parents=True)
        golden_file = golden_dir / "golden.jsonl"
        entries = [
            {"func_id": 1, "dim": 10, "case": "zeros", "x": [0.0] * 10, "value": 42.0},
            {"func_id": 1, "dim": 10, "case": "shift", "x": [1.0] * 10, "value": 100.0},
        ]
        golden_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        cases = load_golden_cases("CEC_TEST", 1, 10, str(tmp_path))
        assert len(cases) == 2
        assert cases[0]["case"] == "zeros"
        assert cases[0]["value"] == 42.0
        assert cases[1]["case"] == "shift"
        assert len(cases[1]["x"]) == 10

    @pytest.mark.skipif(
        not Path(GOLDEN_ROOT).exists(),
        reason="Golden datasets not available",
    )
    def test_loads_real_cec2014_data(self):
        """Loads actual CEC2014 golden data if available."""
        cases = load_golden_cases("CEC2014", 1, 10, GOLDEN_ROOT)
        assert len(cases) == 5
        for case in cases:
            assert "func_id" in case
            assert "dim" in case
            assert "case" in case
            assert "x" in case
            assert "value" in case
            assert case["dim"] == 10
            assert case["func_id"] == 1
