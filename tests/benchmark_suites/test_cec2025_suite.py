"""
Validation tests for CEC 2025 (GNBG-II) benchmark suite.

Validates suite configuration, function construction via FunctionFactory and GNBGSuiteFactory,
and numerical evaluation consistency across all 24 problem instances.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyMOFL.factories.function_factory import DataLoader, FunctionFactory, FunctionRegistry
from pyMOFL.factories.gnbg_suite_factory import GNBGSuiteFactory
from pyMOFL.utils import inject_dimension, load_suite_config

SUITE_JSON = "src/pyMOFL/constants/cec/2025/cec2025_suite.json"
DATA_DIR = "src/pyMOFL/constants/cec/2025"


@pytest.fixture(scope="module")
def suite_config():
    return load_suite_config(SUITE_JSON)


@pytest.fixture(scope="module")
def factory():
    loader = DataLoader(base_path=DATA_DIR)
    registry = FunctionRegistry()
    return FunctionFactory(data_loader=loader, registry=registry)


@pytest.fixture(scope="module")
def gnbg_factory():
    return GNBGSuiteFactory(data_path=DATA_DIR)


def _find_config_by_func_id(suite_config: dict, func_id: int) -> dict:
    suffix = f"f{func_id:02d}"
    for func_cfg in suite_config["functions"]:
        if func_cfg["id"].endswith(suffix):
            return func_cfg
    raise ValueError(f"No config for F{func_id}")


def _create_function(
    factory: FunctionFactory, suite_config: dict, func_id: int, dim: int | None = None
):
    func_cfg = _find_config_by_func_id(suite_config, func_id)
    if dim is None:
        dim = func_cfg["dimensions"]["default"]
    return factory.create_function(inject_dimension(func_cfg["function"], dim))


class TestCEC2025SuiteConfig:
    """Test suite configuration integrity."""

    def test_metadata(self, suite_config):
        assert suite_config["suite_id"] == "cec2025"
        assert len(suite_config["functions"]) == 24

    def test_expected_dimensions(self, suite_config):
        for fid in range(1, 25):
            cfg = _find_config_by_func_id(suite_config, fid)
            assert cfg["dimensions"]["default"] == 10
            assert cfg["dimensions"]["supported"] == [2, 10, 30, 50, 100]

    def test_search_space(self, suite_config):
        for fid in range(1, 25):
            cfg = _find_config_by_func_id(suite_config, fid)
            bounds = cfg["search_space"]["default_bounds"]
            assert bounds["min"] < bounds["max"]
            assert np.isfinite(bounds["min"])
            assert np.isfinite(bounds["max"])

    def test_gnbg_factory_info(self, gnbg_factory):
        info_all = gnbg_factory.get_function_info()
        assert len(info_all) == 24
        info_int = gnbg_factory.get_function_info(1)
        info_str = gnbg_factory.get_function_info("cec25_f01")
        info_gnbg = gnbg_factory.get_function_info("gnbg_f01")
        assert info_int["id"] == "cec25_f01"
        assert info_str["id"] == "cec25_f01"
        assert info_gnbg["id"] == "cec25_f01"


class TestCEC2025Evaluation:
    """Validate numerical evaluations for CEC 2025 functions."""

    # Reference evaluation values for all 24 GNBG-II instances at x = np.ones(10) (D=10)
    CEC2025_REFERENCE_AT_ONES_D10 = {
        1: 1.1207278860,
        2: 0.0984742710,
        3: 2.5273255722,
        4: 1.7563298905,
        5: 2.2000184059,
        6: 0.4727485438,
        7: 16.0037938043,
        8: 13.3362364240,
        9: 38.5820704099,
        10: 34.5049532650,
        11: 58.2667668630,
        12: 19.3973465881,
        13: 51.4181256286,
        14: 100.5346839498,
        15: 31.3686065733,
        16: 57.2567511589,
        17: 47.1614215605,
        18: 20.4495112368,
        19: 19.8275561000,
        20: 2.1742185587,
        21: 15.1881794307,
        22: 8.1663351540,
        23: 15.6838199948,
        24: 30.2797040816,
    }

    @pytest.mark.parametrize("fid", range(1, 25))
    def test_construction_finite_at_zeros(self, factory, suite_config, fid):
        """Construction check: verify all 24 functions construct and evaluate to finite values at origin."""
        func = _create_function(factory, suite_config, fid, dim=10)
        assert func.dimension == 10
        val = func.evaluate(np.zeros(10))
        assert np.isfinite(val)

    @pytest.mark.parametrize("fid", range(1, 25))
    def test_pinned_reference_non_optimum_vector(self, factory, suite_config, fid):
        """Pin each function at a non-optimum vector (ones) against independent reference."""
        func = _create_function(factory, suite_config, fid, dim=10)
        val = func.evaluate(np.ones(10))
        expected = self.CEC2025_REFERENCE_AT_ONES_D10[fid]
        np.testing.assert_allclose(val, expected, rtol=1e-8, atol=1e-8)

    @pytest.mark.parametrize("fid", range(1, 25))
    def test_gnbg_suite_factory_equivalence(self, factory, gnbg_factory, suite_config, fid):
        """Verify GNBGSuiteFactory creates identical functions to FunctionFactory."""
        f_direct = _create_function(factory, suite_config, fid, dim=10)
        f_gnbg = gnbg_factory.create_function(fid, dim=10)

        rng = np.random.default_rng(fid * 7)
        x = rng.uniform(-100.0, 100.0, size=10)
        val_direct = f_direct.evaluate(x)
        val_gnbg = f_gnbg.evaluate(x)
        np.testing.assert_allclose(val_gnbg, val_direct, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("fid", range(1, 10))
    def test_batch_evaluation_consistency(self, factory, suite_config, fid):
        """Verify vectorized batch evaluation matches single evaluations."""
        func = _create_function(factory, suite_config, fid, dim=10)
        rng = np.random.default_rng(fid * 42)
        X = rng.uniform(-100.0, 100.0, size=(5, 10))
        batch_vals = func.evaluate_batch(X)
        single_vals = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch_vals, single_vals, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 10, 30, 50, 100])
    def test_multi_dimension_construction(self, gnbg_factory, dim):
        """Verify construction across all supported dimensions for F1."""
        func = gnbg_factory.create_function(1, dim=dim)
        assert func.dimension == dim
        val = func.evaluate(np.zeros(dim))
        assert np.isfinite(val)
