"""Tests for the high-level pyMOFL API facade: load and get_suite."""

import numpy as np
import pytest

import pyMOFL
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.functions.benchmark.sphere import SphereFunction
from pyMOFL.functions.transformations.composed import ComposedFunction
from pyMOFL.loader import BenchmarkSuite, get_suite, load


class TestLoadClassicalFunctions:
    """Test pyMOFL.load with classical benchmark functions from registry."""

    def test_load_sphere_by_alias(self):
        func = load("sphere", dimension=5)
        assert isinstance(func, SphereFunction)
        assert func.dimension == 5
        val = func(np.zeros(5))
        assert val == 0.0

    def test_load_sphere_by_class_name(self):
        func = load("SphereFunction", dimension=10)
        assert isinstance(func, SphereFunction)
        assert func.dimension == 10

    def test_load_rosenbrock(self):
        func = load("rosenbrock", dimension=3)
        assert func.dimension == 3
        val = func(np.ones(3))
        assert np.isclose(val, 0.0)

    def test_load_requires_dimension_for_scalable(self):
        with pytest.raises(ValueError, match="requires a 'dimension'"):
            load("sphere")

    def test_load_fixed_2d_function_without_dimension(self):
        func = load("beale")
        assert func.dimension == 2
        val = func(np.array([3.0, 0.5]))
        assert np.isclose(val, 0.0, atol=1e-5)


class TestLoadSuiteFunctions:
    """Test pyMOFL.load with CEC, BBOB, and GNBG suite functions."""

    def test_load_cec2017_by_prefixed_id(self):
        func = load("cec17_f01", dimension=10)
        assert isinstance(func, OptimizationFunction)
        assert func.dimension == 10
        x = np.zeros(10)
        val = func(x)
        assert np.isfinite(val)

    def test_load_cec2017_by_suite_and_number(self):
        func = load(1, suite="cec2017", dimension=10)
        assert isinstance(func, OptimizationFunction)
        assert func.dimension == 10
        x = np.zeros(10)
        val = func(x)
        assert np.isfinite(val)

    def test_load_cec2005_function(self):
        func = load("cec05_f01_shifted_sphere", dimension=10)
        assert isinstance(func, OptimizationFunction)
        assert func.dimension == 10
        val = func(np.zeros(10))
        assert np.isfinite(val)

    def test_load_bbob_by_prefixed_id(self):
        func = load("bbob_f01", dimension=5, instance=1)
        assert isinstance(func, ComposedFunction)
        assert func.dimension == 5
        val = func(np.zeros(5))
        assert np.isfinite(val)

    def test_load_bbob_by_suite_and_fid(self):
        func = load(1, suite="bbob", dimension=3, instance=1)
        assert isinstance(func, ComposedFunction)
        assert func.dimension == 3
        val = func(np.zeros(3))
        assert np.isfinite(val)

    def test_load_gnbg_by_prefixed_id(self):
        func = load("gnbg_f01", dimension=10)
        assert isinstance(func, OptimizationFunction)
        assert func.dimension == 10
        val = func(np.zeros(10))
        assert np.isfinite(val)

    def test_load_gnbg_by_suite_and_fid(self):
        func = load(1, suite="gnbg", dimension=10)
        assert isinstance(func, OptimizationFunction)
        assert func.dimension == 10
        val = func(np.zeros(10))
        assert np.isfinite(val)


class TestGetSuite:
    """Test pyMOFL.get_suite for full suite instantiation."""

    def test_get_suite_cec2017(self):
        suite = get_suite("cec2017", dimension=10)
        assert isinstance(suite, BenchmarkSuite)
        assert isinstance(suite, list)
        assert len(suite) == 29
        assert suite.suite_id == "cec2017"
        assert suite.dimension == 10

        # Indexing by integer
        f0 = suite[0]
        assert isinstance(f0, OptimizationFunction)
        assert f0.dimension == 10
        val = f0(np.zeros(10))
        assert np.isfinite(val)

        # Indexing by ID / code
        f_by_id = suite["cec17_f01"]
        assert f_by_id is f0
        f_by_code = suite["f01"]
        assert f_by_code is f0

        # Iteration
        count = sum(1 for _ in suite)
        assert count == 29

    def test_get_suite_bbob(self):
        suite = get_suite("bbob", dimension=2, instance=1)
        assert len(suite) == 24
        assert suite.suite_id == "bbob_noiseless"
        assert suite.dimension == 2

        f1 = suite[0]
        assert f1.dimension == 2
        val = f1(np.zeros(2))
        assert np.isfinite(val)
        assert suite["bbob_f01"] is f1
        assert suite["f01"] is f1

    def test_get_suite_gnbg(self):
        suite = get_suite("gnbg", dimension=10)
        assert len(suite) == 24
        assert suite.dimension == 10
        f1 = suite[0]
        val = f1(np.zeros(10))
        assert np.isfinite(val)
        assert suite["gnbg_f01"] is f1

    def test_get_suite_unknown_raises(self):
        with pytest.raises(ValueError, match="No suite configuration found"):
            get_suite("nonexistent_suite_xyz")


class TestTopLevelPackageExports:
    """Test top-level pyMOFL exports."""

    def test_package_exports(self):
        assert hasattr(pyMOFL, "load")
        assert hasattr(pyMOFL, "get_suite")
        assert hasattr(pyMOFL, "BenchmarkSuite")
        assert pyMOFL.load is load
        assert pyMOFL.get_suite is get_suite
        assert pyMOFL.BenchmarkSuite is BenchmarkSuite
