"""
Tests for CEC 2010 Large-Scale Global Optimization benchmark suite.

Validates suite configuration, function construction via FunctionFactory,
global minimum values at shifted optima, and pinned reference numerical parity
at D = 1000.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyMOFL.factories.function_factory import DataLoader, FunctionFactory, FunctionRegistry
from pyMOFL.utils import inject_dimension, load_suite_config

SUITE_JSON = Path("src/pyMOFL/constants/cec/2010/cec2010_suite.json")
DATA_DIR = Path("src/pyMOFL/constants/cec/2010")

PINNED_ORIGIN_VALUES_D1000 = {
    1: 200013574823.1994,
    2: 17053.18650630713,
    3: 21.056672817164554,
    4: 7688021793189008.0,
    5: 1010097574.061646,
    6: 20927444.78573728,
    7: 20462163874762.37,
    8: 6.71906326544901e16,
    9: 240853971221.92047,
    10: 17426.670905750347,
    11: 231.68201493645788,
    12: 33824183.13459679,
    13: 701236472002.1222,
    14: 272900539536.46182,
    15: 17402.178851791195,
    16: 419.58943225210203,
    17: 76484601.8181398,
    18: 1475640453543.9058,
    19: 3347846871.121293,
    20: 1656753149555.2407,
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


def _create_function(factory: FunctionFactory, suite_config: dict, func_id: int, dim: int = 1000):
    func_cfg = _find_config_by_func_id(suite_config, func_id)
    return factory.create_function(inject_dimension(func_cfg["function"], dim))


class TestCEC2010SuiteConfig:
    """Test suite configuration integrity."""

    def test_metadata(self, suite_config):
        assert suite_config["suite_id"] == "cec2010"
        assert len(suite_config["functions"]) == 20

    def test_dimensions(self, suite_config):
        for func_cfg in suite_config["functions"]:
            assert func_cfg["dimensions"]["supported"] == [1000]
            assert func_cfg["dimensions"]["default"] == 1000

    def test_search_spaces(self, suite_config):
        for func_id in range(1, 21):
            cfg = _find_config_by_func_id(suite_config, func_id)
            if func_id in {2, 5, 10, 15}:
                assert (cfg["search_space"]["low"], cfg["search_space"]["high"]) == (-5.0, 5.0)
            elif func_id in {3, 6, 11, 16}:
                assert (cfg["search_space"]["low"], cfg["search_space"]["high"]) == (-32.0, 32.0)
            else:
                assert (cfg["search_space"]["low"], cfg["search_space"]["high"]) == (-100.0, 100.0)


class TestCEC2010NumericalParity:
    """Verify numerical parity, global minima, and pinned points at D = 1000."""

    @pytest.mark.parametrize("func_id", range(1, 21))
    def test_global_optimum_evaluates_to_zero(self, factory, suite_config, loader, func_id):
        func = _create_function(factory, suite_config, func_id, 1000)
        f_num = f"f{func_id:02d}"
        o = loader.load_vector(f"{f_num}/vector_shift_D1000.txt", 1000)

        if func_id == 8:
            p = loader.load_vector(f"{f_num}/vector_permutation_D1000.txt", 1000).astype(int)
            x_opt = o.copy()
            x_opt[p[:50]] += 1.0
        elif func_id == 13:
            p = loader.load_vector(f"{f_num}/vector_permutation_D1000.txt", 1000).astype(int)
            x_opt = o.copy()
            x_opt[p[:500]] += 1.0
        elif func_id in {18, 20}:
            x_opt = o + 1.0
        else:
            x_opt = o.copy()

        val_opt = func.evaluate(x_opt)
        assert np.isclose(val_opt, 0.0, atol=1e-10)

    @pytest.mark.parametrize("func_id", range(1, 21))
    def test_pinned_origin_d1000(self, factory, suite_config, func_id):
        func = _create_function(factory, suite_config, func_id, 1000)
        val = func.evaluate(np.zeros(1000))
        expected = PINNED_ORIGIN_VALUES_D1000[func_id]
        assert np.isclose(val, expected, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("func_id", range(1, 21))
    def test_batch_evaluation_matches_single(self, factory, suite_config, func_id):
        func = _create_function(factory, suite_config, func_id, 1000)
        cfg = _find_config_by_func_id(suite_config, func_id)
        low = cfg["search_space"]["low"]
        high = cfg["search_space"]["high"]

        rng = np.random.default_rng(func_id * 1000)
        X = rng.uniform(low, high, size=(3, 1000))
        batch_vals = func.evaluate_batch(X)
        single_vals = np.array([func.evaluate(row) for row in X])

        np.testing.assert_allclose(batch_vals, single_vals, rtol=1e-10, atol=1e-10)
