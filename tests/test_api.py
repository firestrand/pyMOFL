"""Tests for the high-level pyMOFL API facade: load and get_suite."""

import numpy as np
import pytest

import pyMOFL
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.functions.benchmark.sphere import SphereFunction
from pyMOFL.functions.transformations.composed import ComposedFunction
from pyMOFL.loader import BenchmarkSuite, get_suite, load


@pytest.fixture
def canonical_suite_entries():
    """DATA-03: actual BBOB F1/F2/F3 instances and their existing identifiers."""
    return get_suite("bbob", dimension=2, instance=1)[:3]


def test_suite_numeric_strings_are_canonical_not_positions(canonical_suite_entries):
    first, second, third = canonical_suite_entries
    suite = BenchmarkSuite([third, first, second], "bbob")
    assert suite["1"] is first
    assert suite["01"] is first
    assert suite["f01"] is first
    assert suite[1] is first
    assert suite[-1] is second
    assert suite[:2] == [third, first]
    with pytest.raises(KeyError):
        suite["0"]
    with pytest.raises(KeyError):
        suite["cec17_f01"]  # A different full ID must not fall back to BBOB F1.


@pytest.mark.parametrize(
    "operation",
    [
        "append",
        "extend",
        "insert",
        "remove",
        "pop",
        "clear",
        "setitem",
        "setslice",
        "delitem",
        "delslice",
        "reverse",
        "sort",
        "iadd",
        "imul0",
        "imul2",
    ],
)
def test_suite_lookup_tracks_every_list_mutator(canonical_suite_entries, operation):
    first, second, third = canonical_suite_entries
    suite = BenchmarkSuite([first, second], "bbob")
    if operation == "append":
        suite.append(third)
    elif operation == "extend":
        suite.extend([third])
    elif operation == "insert":
        suite.insert(0, third)
    elif operation == "remove":
        suite.remove(first)
    elif operation == "pop":
        assert suite.pop() is second
    elif operation == "clear":
        suite.clear()
    elif operation == "setitem":
        suite[0] = third
    elif operation == "setslice":
        suite[:] = [third]
    elif operation == "delitem":
        del suite[0]
    elif operation == "delslice":
        del suite[:]
    elif operation == "reverse":
        suite.reverse()
    elif operation == "sort":
        suite.sort(key=lambda entry: entry.function_id, reverse=True)
    elif operation == "iadd":
        suite += [third]
    elif operation == "imul0":
        suite *= 0
    elif operation == "imul2":
        suite *= 2
    for number, entry in enumerate(canonical_suite_entries, start=1):
        count = sum(item is entry for item in suite)
        if count == 0:
            assert suite.get(str(number)) is None
            with pytest.raises(KeyError):
                suite[entry.function_id]
        elif count == 1:
            assert suite[str(number)] is entry
            assert suite[entry.function_id] is entry
        else:
            with pytest.raises(ValueError, match=r"[Aa]mbiguous"):
                suite.get(str(number))


def test_suite_lookup_tracks_metadata_and_rejects_ambiguity(canonical_suite_entries):
    first, second, third = canonical_suite_entries
    original_id = first.function_id
    suite = BenchmarkSuite([first, second], "bbob")
    first.function_id = third.function_id
    first.name = SphereFunction.__name__
    assert suite["3"] is first
    assert suite[" SPHEREFUNCTION "] is first
    assert suite.get(original_id) is None
    first.function_id = second.function_id
    with pytest.raises(ValueError, match=r"[Aa]mbiguous"):
        suite[second.function_id]
    second.name = first.name
    with pytest.raises(ValueError, match=r"[Aa]mbiguous"):
        suite[first.name]
    with pytest.raises(TypeError):
        suite[None]


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
