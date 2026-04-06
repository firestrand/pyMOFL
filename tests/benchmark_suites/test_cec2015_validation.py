"""
Validation tests for CEC 2015 benchmark suite against golden datasets.
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

SUITE_JSON = "src/pyMOFL/constants/cec/2015/cec2015_suite.json"
DATA_DIR = "src/pyMOFL/constants/cec/2015"

pytestmark = pytest.mark.skipif(
    not Path(GOLDEN_ROOT).exists(),
    reason="Golden datasets not available (set CEC_BENCHMARKS_PATH)",
)


@pytest.fixture(scope="module")
def suite_config():
    return load_suite_config(SUITE_JSON)


@pytest.fixture(scope="module")
def factory():
    loader = DataLoader(base_path=DATA_DIR)
    registry = FunctionRegistry()
    return FunctionFactory(data_loader=loader, registry=registry)


def _find_config_by_func_id(suite_config: dict, func_id: int) -> dict | None:
    suffix = f"f{func_id:02d}_"
    for func_cfg in suite_config["functions"]:
        if suffix in func_cfg["id"]:
            return func_cfg
    return None


def _create_function(factory: FunctionFactory, suite_config: dict, func_id: int, dim: int):
    func_cfg = _find_config_by_func_id(suite_config, func_id)
    if func_cfg is None:
        pytest.skip(f"No config for F{func_id}")
    return factory.create_function(inject_dimension(func_cfg["function"], dim))


XFAIL_FUNCS: dict[int, str] = {}


class TestCEC2015Validation:
    """Test all CEC 2015 functions against golden data."""

    @pytest.mark.parametrize("func_id", list(range(1, 16)))
    @pytest.mark.parametrize("dim", [10, 30, 50])
    def test_function(self, factory, suite_config, func_id, dim):
        if func_id in XFAIL_FUNCS:
            pytest.xfail(XFAIL_FUNCS[func_id])

        cases = load_golden_cases("CEC2015", func_id, dim, GOLDEN_ROOT)
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


class TestCEC2015Construction:
    """Test that all CEC 2015 functions can be constructed."""

    @pytest.mark.parametrize("func_id", list(range(1, 16)))
    def test_construct(self, factory, suite_config, func_id):
        func = _create_function(factory, suite_config, func_id, 10)
        assert func is not None
        x = np.zeros(10, dtype=np.float64)
        result = func.evaluate(x)
        assert np.isfinite(result), f"F{func_id}: non-finite result at zeros"
