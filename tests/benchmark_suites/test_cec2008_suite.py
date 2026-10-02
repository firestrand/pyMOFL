"""
Tests for CEC 2008 Large-Scale Global Optimization benchmark suite.

Validates suite configuration, function construction via FunctionFactory,
global minimum values at shifted optima, and reference numerical parity across
dimensions D in {100, 500, 1000}.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyMOFL.factories.function_factory import DataLoader, FunctionFactory, FunctionRegistry
from pyMOFL.utils import inject_dimension, load_suite_config

SUITE_JSON = Path("src/pyMOFL/constants/cec/2008/cec2008_suite.json")
DATA_DIR = Path("src/pyMOFL/constants/cec/2008")

KNOWN_OPTIMA = {
    1: -450.0,
    2: -450.0,
    3: -390.0,
    4: -330.0,
    5: -180.0,
    6: -140.0,
}

PINNED_ORIGIN_VALUES_D100 = {
    1: 359246.7931655968,
    2: -350.35397290000003,
    3: 101086626292.55115,
    4: 1757.0191156539822,
    5: 2679.8377086382256,
    6: -118.95082745026707,
    7: -561.3365108174407,
}


@pytest.fixture(scope="module")
def suite_config():
    return load_suite_config(SUITE_JSON)


@pytest.fixture(scope="module")
def loader():
    return DataLoader(base_path=DATA_DIR)


@pytest.fixture(scope="module")
def factory(loader):
    registry = FunctionRegistry()
    return FunctionFactory(data_loader=loader, registry=registry)


def _find_config_by_func_id(suite_config: dict, func_id: int) -> dict:
    suffix = f"f{func_id:02d}_"
    for func_cfg in suite_config["functions"]:
        if suffix in func_cfg["id"]:
            return func_cfg
    raise ValueError(f"No config for F{func_id}")


def _create_function(factory: FunctionFactory, suite_config: dict, func_id: int, dim: int):
    func_cfg = _find_config_by_func_id(suite_config, func_id)
    return factory.create_function(inject_dimension(func_cfg["function"], dim))


class TestCEC2008SuiteConfig:
    """Test suite configuration integrity."""

    def test_metadata(self, suite_config):
        assert suite_config["suite_id"] == "cec2008"
        assert len(suite_config["functions"]) == 7

    def test_supported_dimensions(self, suite_config):
        for func_cfg in suite_config["functions"]:
            assert func_cfg["dimensions"]["supported"] == [100, 500, 1000]
            assert func_cfg["dimensions"]["default"] == 500

    def test_bounds(self, suite_config):
        expected_bounds = {
            1: (-100.0, 100.0),
            2: (-100.0, 100.0),
            3: (-100.0, 100.0),
            4: (-5.0, 5.0),
            5: (-600.0, 600.0),
            6: (-32.0, 32.0),
            7: (-1.0, 1.0),
        }
        for func_id, (low, high) in expected_bounds.items():
            cfg = _find_config_by_func_id(suite_config, func_id)
            assert cfg["search_space"]["low"] == low
            assert cfg["search_space"]["high"] == high


class TestCEC2008NumericalParity:
    """Verify numerical parity and global optima across dimensions."""

    @pytest.mark.parametrize("func_id", [1, 2, 3, 4, 5, 6])
    @pytest.mark.parametrize("dim", [100, 500, 1000])
    def test_optimum_value(self, factory, suite_config, loader, func_id, dim):
        func = _create_function(factory, suite_config, func_id, dim)
        f_num = f"f{func_id:02d}"
        o = loader.load_vector(f"{f_num}/vector_shift_D1000.txt", dim)
        val = func.evaluate(o)
        expected = KNOWN_OPTIMA[func_id]
        assert np.isclose(val, expected, atol=1e-10)

    @pytest.mark.parametrize("func_id", range(1, 8))
    def test_pinned_origin_d100(self, factory, suite_config, func_id):
        func = _create_function(factory, suite_config, func_id, 100)
        val = func.evaluate(np.zeros(100))
        expected = PINNED_ORIGIN_VALUES_D100[func_id]
        assert np.isclose(val, expected, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("func_id", range(1, 8))
    def test_batch_evaluation_matches_single(self, factory, suite_config, func_id):
        func = _create_function(factory, suite_config, func_id, 100)
        rng = np.random.default_rng(func_id * 100)
        cfg = _find_config_by_func_id(suite_config, func_id)
        low = cfg["search_space"]["low"]
        high = cfg["search_space"]["high"]

        X = rng.uniform(low, high, size=(5, 100))
        batch_vals = func.evaluate_batch(X)
        single_vals = np.array([func.evaluate(row) for row in X])

        np.testing.assert_allclose(batch_vals, single_vals, rtol=1e-10, atol=1e-10)
