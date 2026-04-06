"""Regression test for CEC 2005 composition functions (F15-F25).

Verifies that all composition functions produce known values at the origin
and at a fixed random point. This test guards against config or builder
changes that would silently alter function values.

F17 is excluded from exact value comparison because it applies stochastic
noise (multiplicative normal), making its output non-deterministic.
"""

from pathlib import Path

import numpy as np
import pytest

from pyMOFL.factories import BenchmarkFactory
from pyMOFL.utils import find_suite_function_config, inject_dimension, load_suite_config

CONSTANTS_DIR = (
    Path(__file__).parent.parent.parent.parent / "src" / "pyMOFL" / "constants" / "cec" / "2005"
)

DIM = 10

# Fixed random point (np.random.RandomState(42).uniform(-5, 5, 10))
FIXED_POINT = np.array(
    [
        -1.254598811526375,
        4.507143064099162,
        2.3199394181140507,
        0.986584841970366,
        -3.439813595575635,
        -3.4400547966379733,
        -4.419163878318005,
        3.6617614577493516,
        1.011150117432088,
        2.080725777960455,
    ]
)

# Expected values captured from current (verbose) configs.
# F17 is non-deterministic (noise) and excluded from exact checks.
EXPECTED_AT_ORIGIN = {
    "f15": 1666.7225273397953,
    "f16": 1697.7279016695818,
    # f17 excluded — stochastic noise
    "f18": 910.0,
    "f19": 910.0,
    "f20": 910.0,
    "f21": 2058.4137783224405,
    "f22": 2705.706323225983,
    "f23": 2058.4137783224405,
    "f24": 1977.5764604093733,
    "f25": 1977.5764604093733,
}

RTOL = 1e-9
ATOL = 1e-8

EXPECTED_AT_FIXED = {
    "f15": 1849.7077799284934,
    "f16": 1764.9395257502943,
    # f17 excluded — stochastic noise
    "f18": 3042.2079796502803,
    "f19": 3060.274064676511,
    "f20": 3059.9937612939248,
    "f21": 2683.2607832441154,
    "f22": 5830.025335309107,
    "f23": 2642.521237693406,
    "f24": 2186.123209530319,
    "f25": 2186.123209530319,
}


def _build_function(suite, fid: int, dim: int = DIM):
    """Build a CEC 2005 composition function from suite config."""
    func_id = f"f{fid:02d}"
    function_cfg = find_suite_function_config(suite, func_id)
    func_cfg = inject_dimension(function_cfg, dim)
    factory = BenchmarkFactory(data_path=CONSTANTS_DIR)
    return factory.create_function(func_cfg)


class TestCompositionRegressionAtOrigin:
    """Verify F15-F25 at the origin produce known values."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.suite = load_suite_config(CONSTANTS_DIR / "cec2005_suite.json")

    @pytest.mark.parametrize("fid", sorted(EXPECTED_AT_ORIGIN.keys()))
    def test_at_origin(self, fid):
        fnum = int(fid[1:])
        func = _build_function(self.suite, fnum)
        origin = np.zeros(DIM)
        result = func.evaluate(origin)
        expected = EXPECTED_AT_ORIGIN[fid]
        np.testing.assert_allclose(
            result, expected, rtol=RTOL, atol=ATOL, err_msg=f"{fid} at origin"
        )


class TestCompositionRegressionAtFixed:
    """Verify F15-F25 at a fixed random point produce known values."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.suite = load_suite_config(CONSTANTS_DIR / "cec2005_suite.json")

    @pytest.mark.parametrize("fid", sorted(EXPECTED_AT_FIXED.keys()))
    def test_at_fixed(self, fid):
        fnum = int(fid[1:])
        func = _build_function(self.suite, fnum)
        result = func.evaluate(FIXED_POINT.copy())
        expected = EXPECTED_AT_FIXED[fid]
        np.testing.assert_allclose(
            result, expected, rtol=RTOL, atol=ATOL, err_msg=f"{fid} at fixed point"
        )


class TestCompositionF17Loads:
    """F17 has stochastic noise — verify it loads and evaluates without error."""

    def test_f17_loads_and_evaluates(self):
        suite = load_suite_config(CONSTANTS_DIR / "cec2005_suite.json")
        func = _build_function(suite, 17)
        origin = np.zeros(DIM)
        result = func.evaluate(origin)
        assert np.isfinite(result)
        assert isinstance(result, float)
