"""
Tests for classical fixed-dimension benchmark functions (2D-5D).

Following TDD approach with comprehensive test coverage for:
- Zirilli (2D)
- Zimmerman (2D)
- Box-Betts (3D)
- Gulf (3D)
- Helical Valley (3D)
- Kowalik (4D)
- Miele-Cantrell (4D)
- Corana (4D)
- De Villiers-Glasser 01 (4D)
- De Villiers-Glasser 02 (5D)
- Dolan (5D)
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.box_betts import BoxBettsFunction
from pyMOFL.functions.benchmark.corana import CoranaFunction
from pyMOFL.functions.benchmark.devillers_glasser import (
    DeVilliersGlasser01Function,
    DeVilliersGlasser02Function,
)
from pyMOFL.functions.benchmark.dolan import DolanFunction
from pyMOFL.functions.benchmark.gulf import GulfFunction
from pyMOFL.functions.benchmark.helical_valley import HelicalValleyFunction
from pyMOFL.functions.benchmark.kowalik import KowalikFunction
from pyMOFL.functions.benchmark.miele_cantrell import MieleCantrellFunction
from pyMOFL.functions.benchmark.zimmerman import ZimmermanFunction
from pyMOFL.functions.benchmark.zirilli import ZirilliFunction


class TestZirilliFunction:
    """Test Zirilli (Aluffi-Pentini) function (2D)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = ZirilliFunction()
        assert func.dimension == 2
        assert func.initialization_bounds.low.tolist() == [-10.0, -10.0]
        assert func.initialization_bounds.high.tolist() == [10.0, 10.0]

    def test_dimension_validation(self):
        """Test that non-2D initialization raises ValueError."""
        with pytest.raises(ValueError, match="requires dimension=2"):
            ZirilliFunction(dimension=3)

    def test_global_minimum(self):
        """Test global minimum evaluation."""
        func = ZirilliFunction()
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-10
        assert min_value < 0  # Known to be negative (~-0.3523)

    def test_evaluate_at_origin(self):
        """Test evaluation at origin."""
        func = ZirilliFunction()
        # f(0,0) = 0.25*0 - 0.5*0 + 0.1*0 + 0.5*0 = 0
        result = func.evaluate(np.array([0.0, 0.0]))
        assert abs(result) < 1e-10

    def test_evaluate_known_value(self):
        """Test evaluation at a known point."""
        func = ZirilliFunction()
        # f(1,1) = 0.25*1 - 0.5*1 + 0.1*1 + 0.5*1 = 0.25 - 0.5 + 0.1 + 0.5 = 0.35
        result = func.evaluate(np.array([1.0, 1.0]))
        expected = 0.25 * 1.0 - 0.5 * 1.0 + 0.1 * 1.0 + 0.5 * 1.0
        assert abs(result - expected) < 1e-10

    def test_evaluate_batch(self):
        """Test batch evaluation matches single evaluation."""
        func = ZirilliFunction()
        X = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_array_almost_equal(results, expected)


class TestZimmermanFunction:
    """Test Zimmerman function (2D)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = ZimmermanFunction()
        assert func.dimension == 2
        assert func.initialization_bounds.low.tolist() == [0.0, 0.0]
        assert func.initialization_bounds.high.tolist() == [100.0, 100.0]

    def test_dimension_validation(self):
        """Test that non-2D initialization raises ValueError."""
        with pytest.raises(ValueError, match="requires dimension=2"):
            ZimmermanFunction(dimension=3)

    def test_global_minimum(self):
        """Test global minimum at (7, 2)."""
        func = ZimmermanFunction()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_almost_equal(min_point, [7.0, 2.0])
        assert min_value == 0.0

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to 0 at (7, 2)."""
        func = ZimmermanFunction()
        result = func.evaluate(np.array([7.0, 2.0]))
        assert abs(result) < 1e-10

    def test_penalty_mechanism(self):
        """Test the penalty mechanism for constraint violations."""
        func = ZimmermanFunction()
        # At (50, 50): p1 = 9-50-50 = -91, p2 = 47^2+48^2-16 > 0 -> penalty,
        # p3 = 2500-14 > 0 -> penalty
        result = func.evaluate(np.array([50.0, 50.0]))
        assert result > 100.0  # Should have large penalty

    def test_evaluate_batch(self):
        """Test batch evaluation matches single evaluation."""
        func = ZimmermanFunction()
        X = np.array([[7.0, 2.0], [5.0, 3.0], [50.0, 50.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_array_almost_equal(results, expected)


class TestBoxBettsFunction:
    """Test Box-Betts function (3D)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = BoxBettsFunction()
        assert func.dimension == 3
        np.testing.assert_array_almost_equal(func.initialization_bounds.low, [0.9, 9.0, 0.9])
        np.testing.assert_array_almost_equal(func.initialization_bounds.high, [1.2, 11.2, 1.2])

    def test_dimension_validation(self):
        """Test that non-3D initialization raises ValueError."""
        with pytest.raises(ValueError, match="requires dimension=3"):
            BoxBettsFunction(dimension=2)

    def test_global_minimum(self):
        """Test global minimum at (1, 10, 1)."""
        func = BoxBettsFunction()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_almost_equal(min_point, [1.0, 10.0, 1.0])
        assert min_value == 0.0

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to 0 at (1, 10, 1)."""
        func = BoxBettsFunction()
        result = func.evaluate(np.array([1.0, 10.0, 1.0]))
        assert abs(result) < 1e-10

    def test_evaluate_nonzero(self):
        """Test function evaluates to non-zero away from minimum."""
        func = BoxBettsFunction()
        result = func.evaluate(np.array([1.1, 10.5, 1.1]))
        assert result > 0

    def test_evaluate_batch(self):
        """Test batch evaluation matches single evaluation."""
        func = BoxBettsFunction()
        X = np.array([[1.0, 10.0, 1.0], [1.1, 10.5, 1.1], [0.95, 9.5, 0.95]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_array_almost_equal(results, expected)


class TestGulfFunction:
    """Test Gulf research function (3D)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = GulfFunction()
        assert func.dimension == 3
        np.testing.assert_array_almost_equal(func.initialization_bounds.low, [0.1, 0.1, 0.1])
        np.testing.assert_array_almost_equal(func.initialization_bounds.high, [100.0, 100.0, 100.0])

    def test_dimension_validation(self):
        """Test that non-3D initialization raises ValueError."""
        with pytest.raises(ValueError, match="requires dimension=3"):
            GulfFunction(dimension=4)

    def test_global_minimum(self):
        """Test global minimum at (50, 25, 1.5)."""
        func = GulfFunction()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_almost_equal(min_point, [50.0, 25.0, 1.5])
        assert min_value == 0.0

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to 0 at (50, 25, 1.5)."""
        func = GulfFunction()
        result = func.evaluate(np.array([50.0, 25.0, 1.5]))
        assert abs(result) < 1e-10

    def test_evaluate_nonzero(self):
        """Test function evaluates to non-zero away from minimum."""
        func = GulfFunction()
        result = func.evaluate(np.array([45.0, 20.0, 1.0]))
        assert result > 0

    def test_evaluate_batch(self):
        """Test batch evaluation matches single evaluation."""
        func = GulfFunction()
        X = np.array([[50.0, 25.0, 1.5], [45.0, 20.0, 1.0], [55.0, 30.0, 2.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_array_almost_equal(results, expected)


class TestHelicalValleyFunction:
    """Test Helical Valley (Fletcher-Powell) function (3D)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = HelicalValleyFunction()
        assert func.dimension == 3
        assert func.initialization_bounds.low.tolist() == [-10.0, -10.0, -10.0]
        assert func.initialization_bounds.high.tolist() == [10.0, 10.0, 10.0]

    def test_dimension_validation(self):
        """Test that non-3D initialization raises ValueError."""
        with pytest.raises(ValueError, match="requires dimension=3"):
            HelicalValleyFunction(dimension=2)

    def test_global_minimum(self):
        """Test global minimum at (1, 0, 0)."""
        func = HelicalValleyFunction()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_almost_equal(min_point, [1.0, 0.0, 0.0])
        assert min_value == 0.0

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to 0 at (1, 0, 0)."""
        func = HelicalValleyFunction()
        result = func.evaluate(np.array([1.0, 0.0, 0.0]))
        assert abs(result) < 1e-10

    def test_theta_computation(self):
        """Test that theta is correctly computed using atan2."""
        func = HelicalValleyFunction()
        # At (1, 0): theta = atan2(0, 1)/(2*pi) = 0
        theta = func._theta(1.0, 0.0)
        assert abs(theta) < 1e-10
        # At (0, 1): theta = atan2(1, 0)/(2*pi) = 0.25
        theta = func._theta(0.0, 1.0)
        assert abs(theta - 0.25) < 1e-10

    def test_evaluate_batch(self):
        """Test batch evaluation matches single evaluation."""
        func = HelicalValleyFunction()
        X = np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 0.5], [-1.0, -1.0, 1.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_array_almost_equal(results, expected)


class TestKowalikFunction:
    """Test Kowalik function (4D)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = KowalikFunction()
        assert func.dimension == 4
        assert func.initialization_bounds.low.tolist() == [-5.0, -5.0, -5.0, -5.0]
        assert func.initialization_bounds.high.tolist() == [5.0, 5.0, 5.0, 5.0]

    def test_dimension_validation(self):
        """Test that non-4D initialization raises ValueError."""
        with pytest.raises(ValueError, match="requires dimension=4"):
            KowalikFunction(dimension=3)

    def test_global_minimum(self):
        """Test global minimum near (0.1928, 0.1909, 0.1231, 0.1358)."""
        func = KowalikFunction()
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-10
        assert min_value < 0.001  # Known to be ~3.075e-4

    def test_evaluate_positive(self):
        """Test function is always non-negative (sum of squares)."""
        func = KowalikFunction()
        result = func.evaluate(np.array([0.5, 0.5, 0.5, 0.5]))
        assert result >= 0

    def test_evaluate_batch(self):
        """Test batch evaluation matches single evaluation."""
        func = KowalikFunction()
        X = np.array(
            [
                [0.1928, 0.1909, 0.1231, 0.1358],
                [0.5, 0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_array_almost_equal(results, expected)


class TestMieleCantrellFunction:
    """Test Miele-Cantrell function (4D)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = MieleCantrellFunction()
        assert func.dimension == 4
        assert func.initialization_bounds.low.tolist() == [-1.0, -1.0, -1.0, -1.0]
        assert func.initialization_bounds.high.tolist() == [1.0, 1.0, 1.0, 1.0]

    def test_dimension_validation(self):
        """Test that non-4D initialization raises ValueError."""
        with pytest.raises(ValueError, match="requires dimension=4"):
            MieleCantrellFunction(dimension=5)

    def test_global_minimum(self):
        """Test global minimum at (0, 1, 1, 1)."""
        func = MieleCantrellFunction()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_almost_equal(min_point, [0.0, 1.0, 1.0, 1.0])
        assert min_value == 0.0

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to 0 at (0, 1, 1, 1)."""
        func = MieleCantrellFunction()
        result = func.evaluate(np.array([0.0, 1.0, 1.0, 1.0]))
        # exp(0) = 1, so (1 - 1)^4 = 0, (1-1)^6 = 0, tan(0)^4 = 0, 0^8 = 0
        assert abs(result) < 1e-10

    def test_evaluate_nonzero(self):
        """Test function evaluates to non-zero away from minimum."""
        func = MieleCantrellFunction()
        result = func.evaluate(np.array([0.5, 0.5, 0.5, 0.5]))
        assert result > 0

    def test_evaluate_batch(self):
        """Test batch evaluation matches single evaluation."""
        func = MieleCantrellFunction()
        X = np.array(
            [
                [0.0, 1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5, 0.5],
                [-0.5, 0.8, 0.9, 0.9],
            ]
        )
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_array_almost_equal(results, expected)


class TestCoranaFunction:
    """Test Corana function (4D)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = CoranaFunction()
        assert func.dimension == 4
        assert func.initialization_bounds.low.tolist() == [-5.0, -5.0, -5.0, -5.0]
        assert func.initialization_bounds.high.tolist() == [5.0, 5.0, 5.0, 5.0]

    def test_dimension_validation(self):
        """Test that non-4D initialization raises ValueError."""
        with pytest.raises(ValueError, match="requires dimension=4"):
            CoranaFunction(dimension=3)

    def test_global_minimum(self):
        """Test global minimum at origin."""
        func = CoranaFunction()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_almost_equal(min_point, [0.0, 0.0, 0.0, 0.0])
        assert min_value == 0.0

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to 0 at origin."""
        func = CoranaFunction()
        result = func.evaluate(np.array([0.0, 0.0, 0.0, 0.0]))
        assert abs(result) < 1e-10

    def test_piecewise_behavior(self):
        """Test piecewise nature - far from quantized grid should use x^2 terms."""
        func = CoranaFunction()
        # At x = [0.11, 0, 0, 0]: z1 = floor(0.55 + 0.49999)*sign(0.11)*0.2 = 0.2
        # |0.11 - 0.2| = 0.09 >= 0.05, so f1 = d1 * 0.11^2 = 1 * 0.0121
        result = func.evaluate(np.array([0.11, 0.0, 0.0, 0.0]))
        assert result > 0

    def test_evaluate_batch(self):
        """Test batch evaluation matches single evaluation."""
        func = CoranaFunction()
        X = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.11, 0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        )
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_array_almost_equal(results, expected)


class TestDeVilliersGlasser01Function:
    """Test De Villiers-Glasser 01 function (4D)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = DeVilliersGlasser01Function()
        assert func.dimension == 4
        np.testing.assert_array_almost_equal(func.initialization_bounds.low, [1.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(func.initialization_bounds.high, [60.0, 2.0, 5.0, 2.0])

    def test_dimension_validation(self):
        """Test that non-4D initialization raises ValueError."""
        with pytest.raises(ValueError, match="requires dimension=4"):
            DeVilliersGlasser01Function(dimension=3)

    def test_global_minimum(self):
        """Test global minimum at the known parameter values."""
        func = DeVilliersGlasser01Function()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_almost_equal(min_point, [60.137, 1.371, 3.112, 1.761])
        assert min_value == 0.0

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to 0 at the true parameters."""
        func = DeVilliersGlasser01Function()
        result = func.evaluate(np.array([60.137, 1.371, 3.112, 1.761]))
        assert abs(result) < 1e-10

    def test_evaluate_nonzero(self):
        """Test function evaluates to non-zero away from minimum."""
        func = DeVilliersGlasser01Function()
        result = func.evaluate(np.array([50.0, 1.3, 3.0, 1.5]))
        assert result > 0

    def test_evaluate_batch(self):
        """Test batch evaluation matches single evaluation."""
        func = DeVilliersGlasser01Function()
        X = np.array(
            [
                [60.137, 1.371, 3.112, 1.761],
                [50.0, 1.3, 3.0, 1.5],
                [30.0, 1.5, 2.0, 1.0],
            ]
        )
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_array_almost_equal(results, expected)


class TestDeVilliersGlasser02Function:
    """Test De Villiers-Glasser 02 function (5D)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = DeVilliersGlasser02Function()
        assert func.dimension == 5
        np.testing.assert_array_almost_equal(
            func.initialization_bounds.low, [1.0, 1.0, 1.0, 0.0, 0.0]
        )
        np.testing.assert_array_almost_equal(
            func.initialization_bounds.high, [60.0, 2.0, 5.0, 2.0, 2.0]
        )

    def test_dimension_validation(self):
        """Test that non-5D initialization raises ValueError."""
        with pytest.raises(ValueError, match="requires dimension=5"):
            DeVilliersGlasser02Function(dimension=4)

    def test_global_minimum(self):
        """Test global minimum at the known parameter values."""
        func = DeVilliersGlasser02Function()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_almost_equal(min_point, [53.81, 1.27, 3.012, 2.13, 0.507])
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-10

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to approximately 0 at the true parameters."""
        func = DeVilliersGlasser02Function()
        result = func.evaluate(np.array([53.81, 1.27, 3.012, 2.13, 0.507]))
        assert abs(result) < 1e-10

    def test_evaluate_nonzero(self):
        """Test function evaluates to non-zero away from minimum."""
        func = DeVilliersGlasser02Function()
        result = func.evaluate(np.array([40.0, 1.5, 2.0, 1.0, 0.3]))
        assert result > 0

    def test_evaluate_batch(self):
        """Test batch evaluation matches single evaluation."""
        func = DeVilliersGlasser02Function()
        X = np.array(
            [
                [53.81, 1.27, 3.012, 2.13, 0.507],
                [40.0, 1.5, 2.0, 1.0, 0.3],
                [20.0, 1.1, 4.0, 1.5, 1.0],
            ]
        )
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_array_almost_equal(results, expected)


class TestDolanFunction:
    """Test Dolan function (5D)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = DolanFunction()
        assert func.dimension == 5
        assert func.initialization_bounds.low.tolist() == [-100.0] * 5
        assert func.initialization_bounds.high.tolist() == [100.0] * 5

    def test_dimension_validation(self):
        """Test that non-5D initialization raises ValueError."""
        with pytest.raises(ValueError, match="requires dimension=5"):
            DolanFunction(dimension=4)

    def test_global_minimum(self):
        """Test global minimum near the known point."""
        func = DolanFunction()
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-10
        # The minimum should be approximately 0
        assert min_value < 1.0

    def test_evaluate_nonnegative(self):
        """Test function is always non-negative (absolute value)."""
        func = DolanFunction()
        rng = np.random.default_rng(42)
        for _ in range(10):
            x = rng.uniform(-10, 10, size=5)
            result = func.evaluate(x)
            assert result >= 0

    def test_evaluate_batch(self):
        """Test batch evaluation matches single evaluation."""
        func = DolanFunction()
        X = np.array(
            [
                [8.39, 4.81, 7.35, 68.88, 3.85],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
            ]
        )
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_array_almost_equal(results, expected)


class TestClassicalFixed3D5DIntegration:
    """Integration tests for all fixed 3D-5D functions."""

    @pytest.mark.parametrize(
        "func_class,expected_dim",
        [
            (ZirilliFunction, 2),
            (ZimmermanFunction, 2),
            (BoxBettsFunction, 3),
            (GulfFunction, 3),
            (HelicalValleyFunction, 3),
            (KowalikFunction, 4),
            (MieleCantrellFunction, 4),
            (CoranaFunction, 4),
            (DeVilliersGlasser01Function, 4),
            (DeVilliersGlasser02Function, 5),
            (DolanFunction, 5),
        ],
    )
    def test_all_functions_instantiate(self, func_class, expected_dim):
        """Test all functions can be instantiated with correct dimension."""
        func = func_class()
        assert hasattr(func, "evaluate")
        assert hasattr(func, "evaluate_batch")
        assert hasattr(func, "get_global_minimum")
        assert func.dimension == expected_dim

    @pytest.mark.parametrize(
        "func_class,expected_dim",
        [
            (ZirilliFunction, 2),
            (ZimmermanFunction, 2),
            (BoxBettsFunction, 3),
            (GulfFunction, 3),
            (HelicalValleyFunction, 3),
            (KowalikFunction, 4),
            (MieleCantrellFunction, 4),
            (CoranaFunction, 4),
            (DeVilliersGlasser01Function, 4),
            (DeVilliersGlasser02Function, 5),
            (DolanFunction, 5),
        ],
    )
    def test_global_minimum_consistency(self, func_class, expected_dim):
        """Test global minimum methods are consistent."""
        func = func_class()
        min_point, min_value = func.get_global_minimum()
        evaluated_value = func.evaluate(min_point)
        assert abs(evaluated_value - min_value) < 1e-6, (
            f"Global minimum inconsistent for {func_class.__name__}: "
            f"reported={min_value}, evaluated={evaluated_value}"
        )

    @pytest.mark.parametrize(
        "func_class,expected_dim",
        [
            (ZirilliFunction, 2),
            (ZimmermanFunction, 2),
            (BoxBettsFunction, 3),
            (GulfFunction, 3),
            (HelicalValleyFunction, 3),
            (KowalikFunction, 4),
            (MieleCantrellFunction, 4),
            (CoranaFunction, 4),
            (DeVilliersGlasser01Function, 4),
            (DeVilliersGlasser02Function, 5),
            (DolanFunction, 5),
        ],
    )
    def test_bounds_consistency(self, func_class, expected_dim):
        """Test bounds have correct dimensionality."""
        func = func_class()
        assert len(func.initialization_bounds.low) == expected_dim
        assert len(func.initialization_bounds.high) == expected_dim

    @pytest.mark.parametrize(
        "func_class,expected_dim",
        [
            (ZirilliFunction, 2),
            (ZimmermanFunction, 2),
            (BoxBettsFunction, 3),
            (GulfFunction, 3),
            (HelicalValleyFunction, 3),
            (KowalikFunction, 4),
            (MieleCantrellFunction, 4),
            (CoranaFunction, 4),
            (DeVilliersGlasser01Function, 4),
            (DeVilliersGlasser02Function, 5),
            (DolanFunction, 5),
        ],
    )
    def test_input_validation(self, func_class, expected_dim):
        """Test input validation rejects wrong dimensions."""
        func = func_class()
        wrong_dim = expected_dim + 1
        with pytest.raises(ValueError):
            func.evaluate(np.ones(wrong_dim))

    def test_registry_integration(self):
        """Test functions are properly registered."""
        from pyMOFL.registry import get

        assert isinstance(get("zirilli")(), ZirilliFunction)
        assert isinstance(get("zimmerman")(), ZimmermanFunction)
        assert isinstance(get("box_betts")(), BoxBettsFunction)
        assert isinstance(get("gulf")(), GulfFunction)
        assert isinstance(get("helical_valley")(), HelicalValleyFunction)
        assert isinstance(get("kowalik")(), KowalikFunction)
        assert isinstance(get("miele_cantrell")(), MieleCantrellFunction)
        assert isinstance(get("corana")(), CoranaFunction)
        assert isinstance(get("devillers_glasser01")(), DeVilliersGlasser01Function)
        assert isinstance(get("devillers_glasser02")(), DeVilliersGlasser02Function)
        assert isinstance(get("dolan")(), DolanFunction)
