"""
Validation tests for CEC 2024 benchmark suite.

Validates suite configuration, function construction via FunctionFactory,
and numerical correctness against official CEC 2024 competition definition.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyMOFL.factories.function_factory import DataLoader, FunctionFactory, FunctionRegistry
from pyMOFL.utils import inject_dimension, load_suite_config

SUITE_JSON = "src/pyMOFL/constants/cec/2024/cec2024_suite.json"
DATA_DIR = "src/pyMOFL/constants/cec/2024"


@pytest.fixture(scope="module")
def suite_config():
    return load_suite_config(SUITE_JSON)


@pytest.fixture(scope="module")
def factory():
    loader = DataLoader(base_path=DATA_DIR)
    registry = FunctionRegistry()
    return FunctionFactory(data_loader=loader, registry=registry)


def _find_config_by_func_id(suite_config: dict, func_id: int) -> dict:
    suffix = f"f{func_id:02d}_"
    for func_cfg in suite_config["functions"]:
        if suffix in func_cfg["id"]:
            return func_cfg
    raise ValueError(f"No config for F{func_id}")


def _create_function(
    factory: FunctionFactory, suite_config: dict, func_id: int, dim: int | None = None
):
    func_cfg = _find_config_by_func_id(suite_config, func_id)
    if dim is None:
        dim = func_cfg["dimensions"]["default"]
    return factory.create_function(inject_dimension(func_cfg["function"], dim))


# F2 is excluded from CEC 2017/2024 competition
ACTIVE_FUNC_IDS = [1, *range(3, 31)]


class TestCEC2024SuiteConfig:
    """Test suite configuration integrity."""

    def test_metadata(self, suite_config):
        assert suite_config["suite_id"] == "cec2024"
        assert len(suite_config["functions"]) == 29

    def test_default_dimensions_are_30(self, suite_config):
        for fid in ACTIVE_FUNC_IDS:
            cfg = _find_config_by_func_id(suite_config, fid)
            assert cfg["dimensions"]["default"] == 30

    def test_search_space(self, suite_config):
        for fid in ACTIVE_FUNC_IDS:
            cfg = _find_config_by_func_id(suite_config, fid)
            bounds = cfg["search_space"]["default_bounds"]
            assert bounds["min"] == -100
            assert bounds["max"] == 100


class TestCEC2024Evaluation:
    """Validate numerical evaluations for CEC 2024 functions."""

    # Independent reference values at x = -100.0 (bounds_min) from official reference C binaries
    CEC2024_REFERENCE_AT_BOUNDS_MIN = {
        1: 449043383041.3316,
        3: 5.865317547309101e16,
        4: 1316248.2790229856,
        5: 2641.8062665755947,
        6: 961.42507955078,
        7: 11410.835741037838,
        8: 2101.881035314964,
        9: 189875.09854970485,
        10: 11529.854390283044,
        # Hybrids (F11 - F20) away from shift:
        11: 83702961078.28694,
        12: 205264372885.45984,
        13: 232727897864.74832,
        14: 6873684408.188775,
        15: 203532033313.4193,
        16: 65535.32221145019,
        17: 122805913.25340798,
        18: 36888355562.95779,
        19: 240675078513.6156,
        20: 5772.853915507563,
        # Compositions (F21 - F30):
        21: 3997.3245879294927,
        22: 14039.849217127798,
        23: 4616.856516264229,
        24: 4688.637271692413,
        25: 256118.5588638908,
        26: 24117.200409997036,
        27: 12532.339743484863,
        28: 124312.48874610609,
        29: 13450730.600643024,
        30: 65702069310.977974,
    }

    @pytest.mark.parametrize("fid", ACTIVE_FUNC_IDS)
    def test_construction_finite_at_zeros(self, factory, suite_config, fid):
        """Construction check: verify each function can be instantiated and evaluated at origin."""
        func = _create_function(factory, suite_config, fid, dim=30)
        assert func.dimension == 30
        val = func.evaluate(np.zeros(30))
        assert np.isfinite(val)

    @pytest.mark.parametrize("fid", ACTIVE_FUNC_IDS)
    def test_pinned_reference_non_optimum_vector(self, factory, suite_config, fid):
        """Pin each function at a non-optimum vector (bounds_min) against independent reference."""
        if fid == 20:
            pytest.xfail("Documented SchafferF7 buffer aliasing reference C bug in Hybrid 10")
        func = _create_function(factory, suite_config, fid, dim=30)
        x_non_opt = np.full(30, -100.0)
        val = func.evaluate(x_non_opt)
        expected = self.CEC2024_REFERENCE_AT_BOUNDS_MIN[fid]
        np.testing.assert_allclose(val, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("fid", [1, 3, 4, 5, 6, 7, 8, 10])
    def test_simple_functions_optima_at_shift(self, factory, suite_config, fid):
        """Verify functions with zero-optimum base evaluate to exact bias at shift vector."""
        func = _create_function(factory, suite_config, fid, dim=30)
        shift_file = f"src/pyMOFL/constants/cec/2017/f{fid:02d}/shift_data.txt"
        shift_vec = np.loadtxt(shift_file)[:30]
        val = func.evaluate(shift_vec)
        expected_bias = float(fid * 100)
        np.testing.assert_allclose(val, expected_bias, rtol=1e-5, atol=1e-5)

    def test_f9_levy_at_shift(self, factory, suite_config):
        """Verify F9 (Levy) at shift vector equals bias 900 + Levy(0)."""
        func = _create_function(factory, suite_config, 9, dim=30)
        shift_file = "src/pyMOFL/constants/cec/2017/f09/shift_data.txt"
        shift_vec = np.loadtxt(shift_file)[:30]
        val = func.evaluate(shift_vec)
        # Levy optimum is at 1, so at shift (z=0) Levy adds ~3.259
        assert val > 900.0
        np.testing.assert_allclose(val, 903.259492, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("fid", ACTIVE_FUNC_IDS[:10])
    def test_batch_evaluation_consistency(self, factory, suite_config, fid):
        """Verify vectorized batch evaluation matches single evaluations."""
        func = _create_function(factory, suite_config, fid, dim=30)
        rng = np.random.default_rng(fid * 123)
        X = rng.uniform(-100.0, 100.0, size=(5, 30))
        batch_vals = func.evaluate_batch(X)
        single_vals = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch_vals, single_vals, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("dim", [10, 30, 50, 100])
    def test_multi_dimension_construction(self, factory, suite_config, dim):
        """Verify construction across all competition dimensions for F1."""
        func = _create_function(factory, suite_config, 1, dim=dim)
        assert func.dimension == dim
        val = func.evaluate(np.zeros(dim))
        assert np.isfinite(val)
