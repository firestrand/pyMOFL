"""
Validation tests for CEC 2017 benchmark suite against golden datasets.

Uses golden JSONL files from the cec-benchmarks repository to validate
pyMOFL's CEC 2017 function implementations against reference C outputs.
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

SUITE_JSON = "src/pyMOFL/constants/cec/2017/cec2017_suite.json"
DATA_DIR = "src/pyMOFL/constants/cec/2017"

# Skip all tests if golden datasets not available
pytestmark = pytest.mark.skipif(
    not Path(GOLDEN_ROOT).exists(),
    reason="Golden datasets not available (set CEC_BENCHMARKS_PATH)",
)


@pytest.fixture(scope="module")
def suite_config():
    """Load CEC 2017 suite configuration."""
    return load_suite_config(SUITE_JSON)


@pytest.fixture(scope="module")
def factory():
    """Create factory with CEC 2017 data loader."""
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
    """Create a CEC 2017 function by ID and dimension."""
    func_cfg = _find_config_by_func_id(suite_config, func_id)
    if func_cfg is None:
        pytest.skip(f"No config for F{func_id}")
    return factory.create_function(inject_dimension(func_cfg["function"], dim))


# CEC 2017 F2 is excluded from the competition (ill-defined)
SIMPLE_IDS = [1, 3, 4, 5, 6, 7, 8, 9, 10]
MULTIMODAL_IDS = list(range(11, 21))
HYBRID_IDS = list(range(21, 31))
# No composition functions in CEC 2017 (F21-F30 are hybrids; there are no F31+)
# CEC 2017 has 30 functions: F1-F10 simple, F11-F20 multimodal(?), F21-F30 hybrid/composition
# Actually: F1-F3 unimodal, F4-F10 simple multimodal, F11-F20 hybrid, F21-F30 composition

# Functions that are known to have formula differences from CEC C reference
# and cannot be matched via config-driven transforms
XFAIL_FUNCS = {}

# Functions with known marginal precision issues due to CEC C reference bugs
# (global y[] array stale in schaffer_F7_func within hybrids)
XFAIL_HYBRID_FUNCS = {
    14: "CEC C reference schaffer_F7_func uses stale global y[] in hybrids (marginal precision)",
    20: "CEC C reference schaffer_F7_func uses stale global y[] in hybrids",
}


# ---------------------------------------------------------------------------
# Simple functions F1-F10 (excluding F2)
# ---------------------------------------------------------------------------
class TestCEC2017Simple:
    """Test simple functions F1-F10 against golden data."""

    @pytest.mark.parametrize("func_id", list(range(1, 11)))
    @pytest.mark.parametrize("dim", [10, 30, 50])
    def test_function(self, factory, suite_config, func_id, dim):
        if func_id == 2:
            pytest.skip("F2 excluded from CEC 2017")
        if func_id in XFAIL_FUNCS:
            pytest.xfail(XFAIL_FUNCS[func_id])

        cases = load_golden_cases("CEC2017", func_id, dim, GOLDEN_ROOT)
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
# Multimodal/Hybrid functions F11-F20
# ---------------------------------------------------------------------------
class TestCEC2017Hybrid:
    """Test hybrid functions F11-F20 against golden data."""

    @pytest.mark.parametrize("func_id", list(range(11, 21)))
    @pytest.mark.parametrize("dim", [10, 30, 50])
    def test_function(self, factory, suite_config, func_id, dim):
        if func_id in XFAIL_HYBRID_FUNCS:
            pytest.xfail(XFAIL_HYBRID_FUNCS[func_id])

        cases = load_golden_cases("CEC2017", func_id, dim, GOLDEN_ROOT)
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
# Composition functions F21-F30
# ---------------------------------------------------------------------------
class TestCEC2017Composition:
    """Test composition functions F21-F30 against golden data."""

    @pytest.mark.parametrize("func_id", list(range(21, 31)))
    @pytest.mark.parametrize("dim", [10, 30, 50])
    def test_function(self, factory, suite_config, func_id, dim):
        cases = load_golden_cases("CEC2017", func_id, dim, GOLDEN_ROOT)
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
# Quick smoke test: can all functions be constructed?
# ---------------------------------------------------------------------------
class TestCEC2017Construction:
    """Test that all CEC 2017 functions can be constructed."""

    @pytest.mark.parametrize("func_id", [i for i in range(1, 31) if i != 2])
    def test_construct(self, factory, suite_config, func_id):
        """Each function should construct without error at D10."""
        func = _create_function(factory, suite_config, func_id, 10)
        assert func is not None
        x = np.zeros(10, dtype=np.float64)
        result = func.evaluate(x)
        assert np.isfinite(result), f"F{func_id}: non-finite result at zeros"
