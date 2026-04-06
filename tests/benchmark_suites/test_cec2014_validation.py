"""
Validation tests for CEC 2014 benchmark suite against golden datasets.

Uses golden JSONL files from the cec-benchmarks repository to validate
pyMOFL's CEC 2014 function implementations against reference C outputs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pyMOFL.factories.function_factory import DataLoader, FunctionFactory, FunctionRegistry
from pyMOFL.utils import inject_dimension, load_suite_config
from tests.utils.golden_loader import load_golden_cases

GOLDEN_ROOT = os.environ.get(
    "CEC_BENCHMARKS_PATH",
    str(Path(__file__).resolve().parents[2].parent / "cec-benchmarks"),
)

SUITE_JSON = "src/pyMOFL/constants/cec/2014/cec2014_suite.json"
DATA_DIR = "src/pyMOFL/constants/cec/2014"

# Skip all tests if golden datasets not available
pytestmark = pytest.mark.skipif(
    not Path(GOLDEN_ROOT).exists(),
    reason="Golden datasets not available (set CEC_BENCHMARKS_PATH)",
)


@pytest.fixture(scope="module")
def suite_config():
    """Load CEC 2014 suite configuration."""
    return load_suite_config(SUITE_JSON)


@pytest.fixture(scope="module")
def factory():
    """Create factory with CEC 2014 data loader."""
    loader = DataLoader(base_path=DATA_DIR)
    registry = FunctionRegistry()
    return FunctionFactory(data_loader=loader, registry=registry)


def _find_config_by_func_id(suite_config: dict, func_id: int) -> dict | None:
    """Find function config by CEC function number (1-based)."""
    suffix = f"f{func_id:02d}_"
    for func_cfg in suite_config["functions"]:
        if suffix in func_cfg["id"]:
            return func_cfg
    return None


def _create_function(factory: FunctionFactory, suite_config: dict, func_id: int, dim: int):
    """Create a CEC 2014 function by ID and dimension."""
    func_cfg = _find_config_by_func_id(suite_config, func_id)
    if func_cfg is None:
        pytest.skip(f"No config for F{func_id}")
    return factory.create_function(inject_dimension(func_cfg["function"], dim))


# ---------------------------------------------------------------------------
# Unimodal functions F1-F3
# ---------------------------------------------------------------------------
class TestCEC2014Unimodal:
    """Test unimodal functions F1-F3 against golden data."""

    @pytest.mark.parametrize("func_id", [1, 2, 3])
    @pytest.mark.parametrize("dim", [10, 30, 50])
    def test_function(self, factory, suite_config, func_id, dim):
        cases = load_golden_cases("CEC2014", func_id, dim, GOLDEN_ROOT)
        if not cases:
            pytest.skip(f"No golden data for F{func_id} D{dim}")

        func = _create_function(factory, suite_config, func_id, dim)
        for case in cases:
            x = np.array(case["x"], dtype=np.float64)
            expected = case["value"]
            actual = func.evaluate(x)
            atol = 1e-6 * max(1.0, abs(expected))
            assert abs(actual - expected) < atol, (
                f"F{func_id} D{dim} case={case['case']}: "
                f"expected={expected}, got={actual}, diff={abs(actual - expected)}"
            )


# ---------------------------------------------------------------------------
# Simple multimodal functions F4-F16
# ---------------------------------------------------------------------------
class TestCEC2014SimpleMultimodal:
    """Test simple multimodal functions F4-F16 against golden data."""

    @pytest.mark.parametrize("func_id", list(range(4, 17)))
    @pytest.mark.parametrize("dim", [10, 30, 50])
    def test_function(self, factory, suite_config, func_id, dim):
        cases = load_golden_cases("CEC2014", func_id, dim, GOLDEN_ROOT)
        if not cases:
            pytest.skip(f"No golden data for F{func_id} D{dim}")

        func = _create_function(factory, suite_config, func_id, dim)
        for case in cases:
            x = np.array(case["x"], dtype=np.float64)
            expected = case["value"]
            actual = func.evaluate(x)
            atol = 1e-6 * max(1.0, abs(expected))
            assert abs(actual - expected) < atol, (
                f"F{func_id} D{dim} case={case['case']}: "
                f"expected={expected}, got={actual}, diff={abs(actual - expected)}"
            )


# ---------------------------------------------------------------------------
# Hybrid functions F17-F22
# ---------------------------------------------------------------------------
class TestCEC2014Hybrid:
    """Test hybrid functions F17-F22 against golden data."""

    @pytest.mark.parametrize("func_id", list(range(17, 23)))
    @pytest.mark.parametrize("dim", [10, 30, 50])
    def test_function(self, factory, suite_config, func_id, dim):
        cases = load_golden_cases("CEC2014", func_id, dim, GOLDEN_ROOT)
        if not cases:
            pytest.skip(f"No golden data for F{func_id} D{dim}")

        func = _create_function(factory, suite_config, func_id, dim)
        for case in cases:
            x = np.array(case["x"], dtype=np.float64)
            expected = case["value"]
            actual = func.evaluate(x)
            atol = 1e-6 * max(1.0, abs(expected))
            assert abs(actual - expected) < atol, (
                f"F{func_id} D{dim} case={case['case']}: "
                f"expected={expected}, got={actual}, diff={abs(actual - expected)}"
            )


# ---------------------------------------------------------------------------
# Composition functions F23-F30
# ---------------------------------------------------------------------------
class TestCEC2014Composition:
    """Test composition functions F23-F30 against golden data."""

    @pytest.mark.parametrize("func_id", list(range(23, 31)))
    @pytest.mark.parametrize("dim", [10, 30, 50])
    def test_function(self, factory, suite_config, func_id, dim):
        cases = load_golden_cases("CEC2014", func_id, dim, GOLDEN_ROOT)
        if not cases:
            pytest.skip(f"No golden data for F{func_id} D{dim}")

        func = _create_function(factory, suite_config, func_id, dim)
        for case in cases:
            x = np.array(case["x"], dtype=np.float64)
            expected = case["value"]
            actual = func.evaluate(x)
            atol = 1e-6 * max(1.0, abs(expected))
            assert abs(actual - expected) < atol, (
                f"F{func_id} D{dim} case={case['case']}: "
                f"expected={expected}, got={actual}, diff={abs(actual - expected)}"
            )


# ---------------------------------------------------------------------------
# Quick smoke test: can all 30 functions be constructed?
# ---------------------------------------------------------------------------
class TestCEC2014Construction:
    """Test that all 30 CEC 2014 functions can be constructed."""

    @pytest.mark.parametrize("func_id", list(range(1, 31)))
    def test_construct(self, factory, suite_config, func_id):
        """Each function should construct without error at D10."""
        func = _create_function(factory, suite_config, func_id, 10)
        assert func is not None
        # Smoke evaluate at zeros
        x = np.zeros(10, dtype=np.float64)
        result = func.evaluate(x)
        assert np.isfinite(result), f"F{func_id}: non-finite result at zeros"
