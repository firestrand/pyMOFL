"""
Tests for CEC 2013 Large-Scale Global Optimization benchmark suite.

Validates suite configuration, function construction via FunctionFactory,
global minimum values at shifted optima, and pinned reference numerical parity
against the official IEEE CEC 2013 LSGO C++ reference implementation at
D = 1000 (F1-F12, F15) and D = 905 (F13-F14).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyMOFL.factories.function_factory import DataLoader, FunctionFactory, FunctionRegistry
from pyMOFL.utils import inject_dimension, load_suite_config

SUITE_JSON = Path("src/pyMOFL/constants/cec/2013_lsgo/cec2013_lsgo_suite.json")
DATA_DIR = Path("src/pyMOFL/constants/cec/2013_lsgo")

# Official ground-truth values at origin X = 0 from compiled C++ benchmark (Li et al., 2013)
PINNED_ORIGIN_VALUES = {
    1: 2.09833896353343505859e11,
    2: 4.76203116166061372496e04,
    3: 2.17290025349525848242e01,
    4: 1.07955147656065968750e14,
    5: 4.84191483329246491194e07,
    6: 1.07773246530947810970e06,
    7: 9.93826981321071500000e14,
    8: 5.72227150187806412800e18,
    9: 6.00160320250193786621e09,
    10: 9.81154816487027555704e07,
    11: 1.04485201647212032000e17,
    12: 1.71135423694972119141e12,
    13: 8.27380048985965600000e16,
    14: 4.40797968120962406400e18,
    15: 2.39389233661550200000e15,
}

# Official ground-truth values at all-ones vector X = 1 from compiled C++ benchmark
PINNED_ONES_VALUES = {
    1: 2.09946678145388153076e11,
    2: 7.00495371043751511024e04,
    3: 2.17108415925775766198e01,
    4: 1.07162206769653843750e14,
    5: 5.87148888268808200955e07,
    6: 1.07977197180324303918e06,
    7: 9.29113705518043375000e14,
    8: 5.60788325599984947200e18,
    9: 9.44072284529277229309e09,
    10: 9.78947871248587518930e07,
    11: 1.01442464039521104000e17,
    12: 1.71217696529957055664e12,
    13: 9.69220815693189600000e16,
    14: 4.37551256977279180800e18,
    15: 2.75152052424948000000e15,
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


class TestCEC2013LSGOSuiteStructure:
    """Validate suite metadata and JSON schema integrity."""

    def test_suite_metadata(self, suite_config):
        assert suite_config["suite_id"] == "cec2013_lsgo"
        assert len(suite_config["functions"]) == 15

    def test_function_dimensions_and_search_space(self, suite_config):
        for i, f_cfg in enumerate(suite_config["functions"], start=1):
            assert "id" in f_cfg
            assert "category" in f_cfg
            assert "dimensions" in f_cfg
            assert "search_space" in f_cfg

            # F13 and F14 are overlapping 905D; all others are 1000D
            expected_dim = 905 if i in (13, 14) else 1000
            assert f_cfg["dimensions"]["default"] == expected_dim
            assert expected_dim in f_cfg["dimensions"]["supported"]

            # Bounds validation
            low = f_cfg["search_space"]["low"]
            high = f_cfg["search_space"]["high"]
            assert low < high
            if i in (2, 5, 9):
                assert (low, high) == (-5.0, 5.0)  # Rastrigin
            elif i in (3, 6, 10):
                assert (low, high) == (-32.0, 32.0)  # Ackley
            else:
                assert (low, high) == (-100.0, 100.0)  # Elliptic, Schwefel, Rosenbrock


class TestCEC2013LSGONumericalParity:
    """Validate exact numerical agreement against C++ reference code."""

    @pytest.mark.parametrize("fid", range(1, 16))
    def test_origin_parity(self, fid, suite_config, factory):
        f_cfg = suite_config["functions"][fid - 1]
        dim = f_cfg["dimensions"]["default"]
        func = factory.create_function(inject_dimension(f_cfg["function"], dim))

        val = func.evaluate(np.zeros(dim))
        target = PINNED_ORIGIN_VALUES[fid]
        rel_err = abs(val - target) / abs(target)
        assert rel_err < 1e-12, f"F{fid} origin parity failed: rel_err={rel_err:.2e}"

    @pytest.mark.parametrize("fid", range(1, 16))
    def test_ones_vector_parity(self, fid, suite_config, factory):
        f_cfg = suite_config["functions"][fid - 1]
        dim = f_cfg["dimensions"]["default"]
        func = factory.create_function(inject_dimension(f_cfg["function"], dim))

        val = func.evaluate(np.ones(dim))
        target = PINNED_ONES_VALUES[fid]
        rel_err = abs(val - target) / abs(target)
        assert rel_err < 1e-12, f"F{fid} ones vector parity failed: rel_err={rel_err:.2e}"

    @pytest.mark.parametrize("fid", range(1, 16))
    def test_optimum_recovery(self, fid, suite_config, factory, loader):
        f_cfg = suite_config["functions"][fid - 1]
        dim = f_cfg["dimensions"]["default"]
        func = factory.create_function(inject_dimension(f_cfg["function"], dim))

        if fid == 12:
            # Shifted Rosenbrock minimum is at x* = o + 1.0
            o = loader.load_vector(f"f{fid:02d}/vector_shift_D{dim}.txt")
            val = func.evaluate(o + 1.0)
            assert abs(val) < 1e-8
        elif fid == 14:
            # Overlapping with conflicting subcomponent optima;
            # global optimum is strictly non-zero
            val = func.evaluate(np.zeros(dim))
            assert val > 0.0
            assert np.isfinite(val)
        else:
            # All other functions have global minimum at x* = o with f(x*) = 0.0
            o = loader.load_vector(f"f{fid:02d}/vector_shift_D{dim}.txt")
            val = func.evaluate(o)
            assert abs(val) < 1e-8

    @pytest.mark.parametrize("fid", range(1, 16))
    def test_batch_evaluation_consistency(self, fid, suite_config, factory):
        f_cfg = suite_config["functions"][fid - 1]
        dim = f_cfg["dimensions"]["default"]
        func = factory.create_function(inject_dimension(f_cfg["function"], dim))

        p1 = np.zeros(dim)
        p2 = np.ones(dim)
        batch = np.vstack([p1, p2])

        batch_vals = func.evaluate_batch(batch)
        v1 = func.evaluate(p1)
        v2 = func.evaluate(p2)

        np.testing.assert_allclose(batch_vals, [v1, v2], rtol=1e-10, atol=1e-10)
