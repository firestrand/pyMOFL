"""
Tests for CEC 2019 "100-Digit Challenge" benchmark suite.

Validates suite configuration, function construction via FunctionFactory,
and numerical correctness against official CEC 2019 reference implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyMOFL.factories.function_factory import DataLoader, FunctionFactory, FunctionRegistry
from pyMOFL.utils import inject_dimension, load_suite_config

SUITE_JSON = "src/pyMOFL/constants/cec/2019/cec2019_suite.json"
DATA_DIR = "src/pyMOFL/constants/cec/2019"


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


def _create_function(factory: FunctionFactory, suite_config: dict, func_id: int):
    func_cfg = _find_config_by_func_id(suite_config, func_id)
    dim = func_cfg["dimensions"]["default"]
    return factory.create_function(inject_dimension(func_cfg["function"], dim))


class TestCEC2019SuiteConfig:
    """Test suite configuration integrity."""

    def test_metadata(self, suite_config):
        assert suite_config["suite_id"] == "cec2019"
        assert len(suite_config["functions"]) == 10

    def test_expected_dimensions(self, suite_config):
        dims = {
            1: 9,
            2: 16,
            3: 18,
            4: 10,
            5: 10,
            6: 10,
            7: 10,
            8: 10,
            9: 10,
            10: 10,
        }
        for fid, expected_dim in dims.items():
            cfg = _find_config_by_func_id(suite_config, fid)
            assert cfg["dimensions"]["default"] == expected_dim


class TestCEC2019ReferenceValues:
    """Validate numerical evaluations against official CEC 2019 reference implementation."""

    # Reference values computed directly by the compiled CEC 2019 C reference code at x = 0
    EXPECTED_AT_ZERO = {
        1: 1.0,
        2: 5.0,
        3: 1.5e21,  # Overlapping penalty at origin
        4: 153.8133110510052,
        5: 227.98210333738817,
        6: 18.246775281680595,
        7: 3730.260049380988,
        8: 6.332640088240733,
        9: 7.580031067555258,
        10: 22.210959804664068,
    }

    @pytest.mark.parametrize("fid", range(1, 11))
    def test_evaluate_at_zeros(self, factory, suite_config, fid):
        """Verify each function matches CEC 2019 C reference at x = 0."""
        func = _create_function(factory, suite_config, fid)
        x_zero = np.zeros(func.dimension)
        val = func.evaluate(x_zero)
        expected = self.EXPECTED_AT_ZERO[fid]
        np.testing.assert_allclose(val, expected, rtol=1e-8, atol=1e-8)

    @pytest.mark.parametrize("fid", range(4, 11))
    def test_evaluate_at_shift_optimum(self, factory, suite_config, fid):
        """Verify shifted & rotated functions F4-F10 achieve optimal bias 1.0 at shift vector."""
        func = _create_function(factory, suite_config, fid)
        # Shift vector for F{fid}
        shift_vec = np.loadtxt(f"{DATA_DIR}/f{fid:02d}/shift_data.txt")[: func.dimension]
        val = func.evaluate(shift_vec)
        # At shift vector, (x - o) = 0, so rotation and base function evaluate at 0 (or base optimum)
        # For standard base functions with f(0)=0, result equals 1.0 bias
        np.testing.assert_allclose(val, 1.0, rtol=1e-8, atol=1e-8)

    def test_f1_chebyshev_optimum(self, factory, suite_config):
        """Verify F1 Chebyshev evaluates to bias 1.0 at analytical Chebyshev coefficients."""
        func = _create_function(factory, suite_config, 1)
        # T_8 polynomial coefficients
        x_opt = np.array([128.0, 0.0, -256.0, 0.0, 160.0, 0.0, -32.0, 0.0, 1.0])
        val = func.evaluate(x_opt)
        np.testing.assert_allclose(val, 1.0, rtol=1e-8, atol=1e-8)

    def test_f2_hilbert_optimum(self, factory, suite_config):
        """Verify F2 Hilbert evaluates to bias 1.0 at 4x4 inverse Hilbert matrix."""
        func = _create_function(factory, suite_config, 2)
        inv_h4 = np.array(
            [
                [16.0, -120.0, 240.0, -140.0],
                [-120.0, 1200.0, -2700.0, 1680.0],
                [240.0, -2700.0, 6480.0, -4200.0],
                [-140.0, 1680.0, -4200.0, 2800.0],
            ]
        ).ravel()
        val = func.evaluate(inv_h4)
        np.testing.assert_allclose(val, 1.0, rtol=1e-8, atol=1e-8)

    @pytest.mark.parametrize("fid", range(1, 11))
    def test_batch_evaluation_consistency(self, factory, suite_config, fid):
        """Verify vectorized batch evaluation matches single evaluations."""
        func = _create_function(factory, suite_config, fid)
        rng = np.random.default_rng(fid * 42)
        X = rng.uniform(-10.0, 10.0, size=(5, func.dimension))
        batch_vals = func.evaluate_batch(X)
        single_vals = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch_vals, single_vals, rtol=1e-10, atol=1e-10)
