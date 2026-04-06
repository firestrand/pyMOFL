"""
Tests for Sub-Phase 6F: special/remaining benchmark functions and aliases.

Tests cover: ColaFunction, BiggsExp02-05, NewFunction01/02, NeedleEyeFunction,
MultiModalFunction, ZeroSumFunction, DecanomialFunction, and registry aliases.
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.biggs_exp import (
    BiggsExp02Function,
    BiggsExp03Function,
    BiggsExp04Function,
    BiggsExp05Function,
)
from pyMOFL.functions.benchmark.cola import ColaFunction
from pyMOFL.functions.benchmark.decanomial import DecanomialFunction
from pyMOFL.functions.benchmark.multi_modal_func import MultiModalFunction
from pyMOFL.functions.benchmark.needle_eye import NeedleEyeFunction
from pyMOFL.functions.benchmark.new_function import NewFunction01Function, NewFunction02Function
from pyMOFL.functions.benchmark.zero_sum import ZeroSumFunction
from pyMOFL.registry import get


# ---------------------------------------------------------------------------
# ColaFunction tests
# ---------------------------------------------------------------------------
class TestColaFunction:
    """Tests for the Cola function (dim=17, fixed)."""

    def test_initialization(self):
        f = ColaFunction()
        assert f.dimension == 17
        assert f.initialization_bounds.low.shape == (17,)
        assert f.operational_bounds.high[0] == 4.0

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="requires dimension=17"):
            ColaFunction(dimension=10)

    def test_global_minimum(self):
        f = ColaFunction()
        x_opt, f_opt = f.get_global_minimum()
        assert x_opt.shape == (17,)
        # The known approximate optimum from the literature is ~11.7464.
        # The returned point is a placeholder (zeros), not the true optimum.
        assert f_opt == pytest.approx(11.7464, abs=0.01)

    def test_evaluate_at_origin(self):
        f = ColaFunction()
        # At origin all cities collapse to (0,0), distances should be non-zero
        val = f.evaluate(np.zeros(17))
        assert val > 0.0

    def test_evaluate_batch(self):
        f = ColaFunction()
        X = np.random.default_rng(42).uniform(-4, 4, (5, 17))
        results = f.evaluate_batch(X)
        assert results.shape == (5,)
        for i in range(5):
            assert results[i] == pytest.approx(f.evaluate(X[i]), abs=1e-10)

    def test_non_negative(self):
        """Cola function is a sum of squares, so always non-negative."""
        f = ColaFunction()
        rng = np.random.default_rng(123)
        for _ in range(10):
            x = rng.uniform(-4, 4, 17)
            assert f.evaluate(x) >= 0.0


# ---------------------------------------------------------------------------
# BiggsExp02 tests
# ---------------------------------------------------------------------------
class TestBiggsExp02Function:
    """Tests for the Biggs EXP02 function (dim=2, fixed)."""

    def test_initialization(self):
        f = BiggsExp02Function()
        assert f.dimension == 2
        assert f.initialization_bounds.low[0] == 0.0
        assert f.operational_bounds.high[0] == 20.0

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="requires dimension=2"):
            BiggsExp02Function(dimension=3)

    def test_global_minimum(self):
        f = BiggsExp02Function()
        x_opt, f_opt = f.get_global_minimum()
        np.testing.assert_array_almost_equal(x_opt, [1.0, 10.0])
        assert f_opt == pytest.approx(0.0, abs=1e-12)
        assert f.evaluate(x_opt) == pytest.approx(0.0, abs=1e-12)

    def test_evaluate_batch(self):
        f = BiggsExp02Function()
        X = np.array([[1.0, 10.0], [2.0, 5.0], [0.5, 15.0]])
        results = f.evaluate_batch(X)
        assert results.shape == (3,)
        for i in range(3):
            assert results[i] == pytest.approx(f.evaluate(X[i]), abs=1e-10)

    def test_non_negative(self):
        """Sum of squares is always non-negative."""
        f = BiggsExp02Function()
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 20, (10, 2))
        results = f.evaluate_batch(X)
        assert np.all(results >= 0.0)


# ---------------------------------------------------------------------------
# BiggsExp03 tests
# ---------------------------------------------------------------------------
class TestBiggsExp03Function:
    """Tests for the Biggs EXP03 function (dim=3, fixed)."""

    def test_initialization(self):
        f = BiggsExp03Function()
        assert f.dimension == 3

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="requires dimension=3"):
            BiggsExp03Function(dimension=2)

    def test_global_minimum(self):
        f = BiggsExp03Function()
        x_opt, f_opt = f.get_global_minimum()
        np.testing.assert_array_almost_equal(x_opt, [1.0, 10.0, 5.0])
        assert f_opt == pytest.approx(0.0, abs=1e-12)
        assert f.evaluate(x_opt) == pytest.approx(0.0, abs=1e-10)

    def test_evaluate_batch(self):
        f = BiggsExp03Function()
        X = np.array([[1.0, 10.0, 5.0], [2.0, 5.0, 3.0]])
        results = f.evaluate_batch(X)
        assert results.shape == (2,)
        for i in range(2):
            assert results[i] == pytest.approx(f.evaluate(X[i]), abs=1e-10)


# ---------------------------------------------------------------------------
# BiggsExp04 tests
# ---------------------------------------------------------------------------
class TestBiggsExp04Function:
    """Tests for the Biggs EXP04 function (dim=4, fixed)."""

    def test_initialization(self):
        f = BiggsExp04Function()
        assert f.dimension == 4

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="requires dimension=4"):
            BiggsExp04Function(dimension=3)

    def test_global_minimum(self):
        f = BiggsExp04Function()
        x_opt, f_opt = f.get_global_minimum()
        np.testing.assert_array_almost_equal(x_opt, [1.0, 10.0, 1.0, 5.0])
        assert f_opt == pytest.approx(0.0, abs=1e-12)
        assert f.evaluate(x_opt) == pytest.approx(0.0, abs=1e-10)

    def test_evaluate_batch(self):
        f = BiggsExp04Function()
        X = np.array([[1.0, 10.0, 1.0, 5.0], [2.0, 5.0, 3.0, 7.0]])
        results = f.evaluate_batch(X)
        assert results.shape == (2,)
        for i in range(2):
            assert results[i] == pytest.approx(f.evaluate(X[i]), abs=1e-10)


# ---------------------------------------------------------------------------
# BiggsExp05 tests
# ---------------------------------------------------------------------------
class TestBiggsExp05Function:
    """Tests for the Biggs EXP05 function (dim=5, fixed)."""

    def test_initialization(self):
        f = BiggsExp05Function()
        assert f.dimension == 5

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="requires dimension=5"):
            BiggsExp05Function(dimension=4)

    def test_global_minimum(self):
        f = BiggsExp05Function()
        x_opt, f_opt = f.get_global_minimum()
        np.testing.assert_array_almost_equal(x_opt, [1.0, 10.0, 1.0, 5.0, 4.0])
        assert f_opt == pytest.approx(0.0, abs=1e-12)
        assert f.evaluate(x_opt) == pytest.approx(0.0, abs=1e-10)

    def test_evaluate_batch(self):
        f = BiggsExp05Function()
        X = np.array([[1.0, 10.0, 1.0, 5.0, 4.0], [2.0, 5.0, 3.0, 7.0, 1.0]])
        results = f.evaluate_batch(X)
        assert results.shape == (2,)
        for i in range(2):
            assert results[i] == pytest.approx(f.evaluate(X[i]), abs=1e-10)


# ---------------------------------------------------------------------------
# NewFunction01 tests
# ---------------------------------------------------------------------------
class TestNewFunction01Function:
    """Tests for New Function 01 (dim=2, fixed)."""

    def test_initialization(self):
        f = NewFunction01Function()
        assert f.dimension == 2
        assert f.initialization_bounds.low[0] == -10.0

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="requires dimension=2"):
            NewFunction01Function(dimension=3)

    def test_global_minimum(self):
        f = NewFunction01Function()
        x_opt, f_opt = f.get_global_minimum()
        assert x_opt.shape == (2,)
        assert f.evaluate(x_opt) == pytest.approx(f_opt, abs=1e-10)

    def test_evaluate_at_origin(self):
        f = NewFunction01Function()
        # cos(0) = 1, |1|^0.5 = 1, linear term = 0
        val = f.evaluate(np.array([0.0, 0.0]))
        assert val == pytest.approx(1.0, abs=1e-10)

    def test_evaluate_batch(self):
        f = NewFunction01Function()
        X = np.array([[0.0, 0.0], [1.0, 1.0], [-5.0, -5.0]])
        results = f.evaluate_batch(X)
        assert results.shape == (3,)
        for i in range(3):
            assert results[i] == pytest.approx(f.evaluate(X[i]), abs=1e-10)


# ---------------------------------------------------------------------------
# NewFunction02 tests
# ---------------------------------------------------------------------------
class TestNewFunction02Function:
    """Tests for New Function 02 (dim=2, fixed)."""

    def test_initialization(self):
        f = NewFunction02Function()
        assert f.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="requires dimension=2"):
            NewFunction02Function(dimension=4)

    def test_global_minimum(self):
        f = NewFunction02Function()
        x_opt, f_opt = f.get_global_minimum()
        assert x_opt.shape == (2,)
        assert f.evaluate(x_opt) == pytest.approx(f_opt, abs=1e-10)

    def test_evaluate_at_origin(self):
        f = NewFunction02Function()
        # sin(0) = 0, |0|^0.5 = 0, linear term = 0
        val = f.evaluate(np.array([0.0, 0.0]))
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_evaluate_batch(self):
        f = NewFunction02Function()
        X = np.array([[0.0, 0.0], [1.0, 1.0], [-5.0, -5.0]])
        results = f.evaluate_batch(X)
        assert results.shape == (3,)
        for i in range(3):
            assert results[i] == pytest.approx(f.evaluate(X[i]), abs=1e-10)


# ---------------------------------------------------------------------------
# NeedleEyeFunction tests
# ---------------------------------------------------------------------------
class TestNeedleEyeFunction:
    """Tests for the Needle Eye function (scalable)."""

    def test_initialization(self):
        f = NeedleEyeFunction(dimension=5)
        assert f.dimension == 5
        assert f.eye == 0.0001

    def test_custom_eye(self):
        f = NeedleEyeFunction(dimension=3, eye=0.01)
        assert f.eye == 0.01

    def test_global_minimum(self):
        f = NeedleEyeFunction(dimension=5)
        x_opt, f_opt = f.get_global_minimum()
        np.testing.assert_array_equal(x_opt, np.zeros(5))
        assert f_opt == 1.0
        assert f.evaluate(x_opt) == 1.0

    def test_inside_eye(self):
        """Points inside the eye should evaluate to 1.0."""
        f = NeedleEyeFunction(dimension=3, eye=0.1)
        x = np.array([0.01, -0.05, 0.09])
        assert f.evaluate(x) == 1.0

    def test_outside_eye(self):
        """Points outside the eye should evaluate to sum(100 + |xi|)."""
        f = NeedleEyeFunction(dimension=3, eye=0.1)
        x = np.array([0.5, 0.3, 0.2])
        expected = np.sum(100.0 + np.abs(x))
        assert f.evaluate(x) == pytest.approx(expected, abs=1e-10)

    def test_boundary_case(self):
        """Point exactly at the eye boundary (|xi| == eye) should be outside."""
        f = NeedleEyeFunction(dimension=2, eye=0.1)
        x = np.array([0.1, 0.0])
        # |0.1| is not < 0.1, so outside
        val = f.evaluate(x)
        assert val > 1.0

    def test_evaluate_batch(self):
        f = NeedleEyeFunction(dimension=3, eye=0.1)
        X = np.array(
            [
                [0.0, 0.0, 0.0],  # inside -> 1.0
                [0.01, 0.02, 0.03],  # inside -> 1.0
                [0.5, 0.5, 0.5],  # outside
            ]
        )
        results = f.evaluate_batch(X)
        assert results.shape == (3,)
        assert results[0] == 1.0
        assert results[1] == 1.0
        assert results[2] > 1.0
        for i in range(3):
            assert results[i] == pytest.approx(f.evaluate(X[i]), abs=1e-10)


# ---------------------------------------------------------------------------
# MultiModalFunction tests
# ---------------------------------------------------------------------------
class TestMultiModalFunction:
    """Tests for the Multi-Modal function (scalable)."""

    def test_initialization(self):
        f = MultiModalFunction(dimension=5)
        assert f.dimension == 5
        assert f.initialization_bounds.low[0] == -10.0

    def test_global_minimum(self):
        f = MultiModalFunction(dimension=5)
        x_opt, f_opt = f.get_global_minimum()
        np.testing.assert_array_equal(x_opt, np.zeros(5))
        assert f_opt == 0.0
        assert f.evaluate(x_opt) == 0.0

    def test_known_value(self):
        """Test with a known input: x = [1, 1, 1]."""
        f = MultiModalFunction(dimension=3)
        x = np.array([1.0, 1.0, 1.0])
        # sum(|x|) = 3, prod(|x|) = 1 -> f = 3
        assert f.evaluate(x) == pytest.approx(3.0, abs=1e-10)

    def test_known_value_2(self):
        """Test with x = [2, 3]."""
        f = MultiModalFunction(dimension=2)
        x = np.array([2.0, 3.0])
        # sum = 5, prod = 6 -> f = 30
        assert f.evaluate(x) == pytest.approx(30.0, abs=1e-10)

    def test_non_negative(self):
        """f(x) = sum(|x|) * prod(|x|) >= 0 always."""
        f = MultiModalFunction(dimension=4)
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.uniform(-10, 10, 4)
            assert f.evaluate(x) >= 0.0

    def test_evaluate_batch(self):
        f = MultiModalFunction(dimension=3)
        X = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 3.0, 4.0]])
        results = f.evaluate_batch(X)
        assert results.shape == (3,)
        for i in range(3):
            assert results[i] == pytest.approx(f.evaluate(X[i]), abs=1e-10)


# ---------------------------------------------------------------------------
# ZeroSumFunction tests
# ---------------------------------------------------------------------------
class TestZeroSumFunction:
    """Tests for the Zero Sum function (scalable)."""

    def test_initialization(self):
        f = ZeroSumFunction(dimension=5)
        assert f.dimension == 5
        assert f.initialization_bounds.low[0] == -10.0

    def test_global_minimum(self):
        f = ZeroSumFunction(dimension=5)
        x_opt, f_opt = f.get_global_minimum()
        np.testing.assert_array_equal(x_opt, np.zeros(5))
        assert f_opt == 0.0
        assert f.evaluate(x_opt) == 0.0

    def test_zero_sum_gives_zero(self):
        """Any point where sum(x) == 0 should give f = 0."""
        f = ZeroSumFunction(dimension=4)
        x = np.array([1.0, -1.0, 2.0, -2.0])
        assert f.evaluate(x) == 0.0

    def test_nonzero_sum_gives_penalty(self):
        """Points with sum != 0 should give f = 1 + sqrt(10000 * |sum|)."""
        f = ZeroSumFunction(dimension=3)
        x = np.array([1.0, 1.0, 1.0])
        s = 3.0
        expected = 1.0 + (10000.0 * s) ** 0.5
        assert f.evaluate(x) == pytest.approx(expected, abs=1e-10)

    def test_non_negative(self):
        """Zero Sum function should always be >= 0."""
        f = ZeroSumFunction(dimension=4)
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.uniform(-10, 10, 4)
            assert f.evaluate(x) >= 0.0

    def test_evaluate_batch(self):
        f = ZeroSumFunction(dimension=3)
        X = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        results = f.evaluate_batch(X)
        assert results.shape == (3,)
        assert results[0] == 0.0
        assert results[1] == 0.0
        assert results[2] > 0.0
        for i in range(3):
            assert results[i] == pytest.approx(f.evaluate(X[i]), abs=1e-10)


# ---------------------------------------------------------------------------
# DecanomialFunction tests
# ---------------------------------------------------------------------------
class TestDecanomialFunction:
    """Tests for the Decanomial function (dim=2, fixed)."""

    def test_initialization(self):
        f = DecanomialFunction()
        assert f.dimension == 2
        assert f.initialization_bounds.low[0] == -10.0

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="requires dimension=2"):
            DecanomialFunction(dimension=3)

    def test_global_minimum(self):
        f = DecanomialFunction()
        x_opt, f_opt = f.get_global_minimum()
        np.testing.assert_array_almost_equal(x_opt, [2.0, -3.0])
        assert f_opt == pytest.approx(0.0, abs=1e-8)
        assert f.evaluate(x_opt) == pytest.approx(0.0, abs=1e-8)

    def test_non_negative(self):
        """Decanomial is 0.001 * (|..| + |..|)^2 >= 0."""
        f = DecanomialFunction()
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.uniform(-10, 10, 2)
            assert f.evaluate(x) >= 0.0

    def test_evaluate_batch(self):
        f = DecanomialFunction()
        X = np.array([[2.0, -3.0], [0.0, 0.0], [1.0, 1.0]])
        results = f.evaluate_batch(X)
        assert results.shape == (3,)
        for i in range(3):
            assert results[i] == pytest.approx(f.evaluate(X[i]), abs=1e-10)


# ---------------------------------------------------------------------------
# Registry alias tests
# ---------------------------------------------------------------------------
class TestRegistryAliases:
    """Test that registry aliases resolve to the correct classes."""

    def test_cola_alias(self):
        cls = get("cola")
        assert cls is ColaFunction

    def test_biggs_exp02_alias(self):
        cls = get("biggs_exp02")
        assert cls is BiggsExp02Function

    def test_biggs_exp03_alias(self):
        cls = get("biggs_exp03")
        assert cls is BiggsExp03Function

    def test_biggs_exp04_alias(self):
        cls = get("biggs_exp04")
        assert cls is BiggsExp04Function

    def test_biggs_exp05_alias(self):
        cls = get("biggs_exp05")
        assert cls is BiggsExp05Function

    def test_new_function01_alias(self):
        cls = get("new_function01")
        assert cls is NewFunction01Function

    def test_new_function02_alias(self):
        cls = get("new_function02")
        assert cls is NewFunction02Function

    def test_needle_eye_alias(self):
        cls = get("needle_eye")
        assert cls is NeedleEyeFunction

    def test_multi_modal_alias(self):
        cls = get("multi_modal")
        assert cls is MultiModalFunction

    def test_zero_sum_alias(self):
        cls = get("zero_sum")
        assert cls is ZeroSumFunction

    def test_decanomial_alias(self):
        cls = get("decanomial")
        assert cls is DecanomialFunction

    def test_cigar_alias(self):
        """Test that 'cigar' alias resolves to BentCigarFunction."""
        from pyMOFL.functions.benchmark.bent_cigar import BentCigarFunction

        cls = get("cigar")
        assert cls is BentCigarFunction

    def test_yao_liu_04_alias(self):
        """Test that 'yao_liu_04' alias resolves to MaxAbsolute."""
        from pyMOFL.functions.benchmark.max_absolute import MaxAbsolute

        cls = get("yao_liu_04")
        assert cls is MaxAbsolute
