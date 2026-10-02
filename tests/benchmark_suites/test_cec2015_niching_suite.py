"""
Tests for CEC 2015 Niching benchmark suite.

Validates suite configuration, function construction via FunctionFactory,
and numerical correctness against official CEC 2015 Niching reference implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyMOFL.factories.function_factory import DataLoader, FunctionFactory, FunctionRegistry
from pyMOFL.utils import inject_dimension, load_suite_config

SUITE_JSON = "src/pyMOFL/constants/cec/2015_niching/cec2015_niching_suite.json"
DATA_DIR = "src/pyMOFL/constants/cec/2015_niching"


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


class TestCEC2015NichingSuiteConfig:
    """Test suite configuration integrity."""

    def test_metadata(self, suite_config):
        assert suite_config["suite_id"] == "cec2015_niching"
        assert len(suite_config["functions"]) == 15

    def test_expected_default_dimensions(self, suite_config):
        dims = {
            1: 5,
            2: 2,
            3: 2,
            4: 5,
            5: 2,
            6: 4,
            7: 6,
            8: 2,
            9: 10,
            10: 10,
            11: 10,
            12: 10,
            13: 10,
            14: 10,
            15: 10,
        }
        for fid, expected_dim in dims.items():
            cfg = _find_config_by_func_id(suite_config, fid)
            assert cfg["dimensions"]["default"] == expected_dim


class TestCEC2015NichingReferenceValues:
    """Validate numerical evaluations against official CEC 2015 Niching reference."""

    # Reference values computed by the compiled CEC 2015 Niching C reference code at x = 0
    EXPECTED_AT_ZERO_F1_F8 = {
        1: 11316.95643043,
        2: 5338.53910485,
        3: 304.31003692,
        4: 446.37951507,
        5: 513.43899999,
        6: 64197.14975967,
        7: 7041.32390871,
        8: 848.87983530,
    }

    @pytest.mark.parametrize("fid", range(1, 9))
    def test_evaluate_f1_f8_at_zeros(self, factory, suite_config, fid):
        """Verify F1-F8 match CEC 2015 Niching C reference at x = 0."""
        func = _create_function(factory, suite_config, fid)
        x_zero = np.zeros(func.dimension)
        val = func.evaluate(x_zero)
        expected = self.EXPECTED_AT_ZERO_F1_F8[fid]
        np.testing.assert_allclose(val, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("fid", range(1, 16))
    def test_evaluate_at_shift_optimum(self, factory, suite_config, fid):
        """Verify all functions evaluate to exact bias (fid * 100.0) at shifted optimum."""
        func = _create_function(factory, suite_config, fid)
        shift_file = f"{DATA_DIR}/f{fid:02d}/shift_data.txt"
        shift_raw = np.loadtxt(shift_file)
        if shift_raw.ndim == 2:
            x_opt = shift_raw[0, : func.dimension]
        else:
            x_opt = shift_raw[: func.dimension]
        val = func.evaluate(x_opt)
        expected = float(fid * 100.0)
        np.testing.assert_allclose(val, expected, rtol=1e-5, atol=1e-5)

    def test_f2_niching_internal_and_local_peaks(self, factory, suite_config):
        """Verify F2 (Five-Uneven-Peak Trap) at points inside (5, 7.5) and local peaks."""
        func = _create_function(factory, suite_config, 2, dim=2)
        # In F2: input x is shifted by o and rotated by M: z = M^T @ (x - o)
        # Conversely, x = M @ z + o gives internal coordinate z.
        o = np.loadtxt(f"{DATA_DIR}/f02/shift_data.txt")[:2]
        M = np.loadtxt(f"{DATA_DIR}/f02/M_D2.txt")

        # Point inside (5, 7.5) with z = [6.0, 6.0]:
        # trap1d(6.0) = 200 - 64 * (7.5 - 6.0) = 104.0; 2D sum = 208.0; plus bias 200.0 = 408.0
        z_mid = np.array([6.0, 6.0])
        x_mid = M.T @ z_mid + o
        np.testing.assert_allclose(func.evaluate(x_mid), 408.0, rtol=1e-10)

        # Local peaks (minima in minimization):
        # z = 5.0 -> trap = 40.0; 2D sum = 80.0; plus bias 200.0 = 280.0
        x_peak5 = M.T @ np.array([5.0, 5.0]) + o
        np.testing.assert_allclose(func.evaluate(x_peak5), 280.0, rtol=1e-10)

        # z = 12.5 -> trap = 60.0; 2D sum = 120.0; plus bias 200.0 = 320.0
        x_peak12 = M.T @ np.array([12.5, 12.5]) + o
        np.testing.assert_allclose(func.evaluate(x_peak12), 320.0, rtol=1e-10)

        # z = 22.5 -> trap = 40.0; 2D sum = 80.0; plus bias 200.0 = 280.0
        x_peak22 = M.T @ np.array([22.5, 22.5]) + o
        np.testing.assert_allclose(func.evaluate(x_peak22), 280.0, rtol=1e-10)

    @pytest.mark.parametrize("fid", range(1, 16))
    def test_batch_evaluation_consistency(self, factory, suite_config, fid):
        """Verify vectorized batch evaluation matches single evaluations."""
        func = _create_function(factory, suite_config, fid)
        rng = np.random.default_rng(fid * 42)
        X = rng.uniform(-1.0, 1.0, size=(5, func.dimension))
        batch_vals = func.evaluate_batch(X)
        single_vals = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch_vals, single_vals, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("fid", range(1, 16))
    def test_all_supported_dimensions_construct(self, factory, suite_config, fid):
        """Verify function construction across all advertised supported dimensions."""
        cfg = _find_config_by_func_id(suite_config, fid)
        for dim in cfg["dimensions"]["supported"]:
            func = _create_function(factory, suite_config, fid, dim)
            assert func.dimension == dim
            val = func.evaluate(np.zeros(dim))
            assert np.isfinite(val)
