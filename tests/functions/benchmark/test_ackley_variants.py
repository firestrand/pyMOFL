"""
Tests for Ackley 2, 3, and 4 function variants.

References:
    Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions
    for global optimization problems". arXiv:1308.4008
    BenchmarkFcns: https://benchmarkfcns.info/
"""

import numpy as np
import pytest

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.functions.benchmark.ackley import (
    Ackley2Function,
    Ackley3Function,
    Ackley4Function,
)


# ---------------------------------------------------------------------------
# Ackley 2: f(x,y) = -200 * exp(-0.02 * sqrt(x^2 + y^2))
# ---------------------------------------------------------------------------
class TestAckley2Function:
    """Tests for Ackley 2 function."""

    def test_initialization_default(self):
        func = Ackley2Function()
        assert func.dimension == 2
        np.testing.assert_allclose(func.operational_bounds.low, [-32, -32])
        np.testing.assert_allclose(func.operational_bounds.high, [32, 32])

    def test_dimension_must_be_2(self):
        with pytest.raises(ValueError, match="dimension=2"):
            Ackley2Function(dimension=3)

    def test_global_minimum(self):
        func = Ackley2Function()
        x_opt = np.array([0.0, 0.0])
        assert np.isclose(func.evaluate(x_opt), -200.0, atol=1e-12)

    def test_get_global_minimum(self):
        func = Ackley2Function()
        point, value = func.get_global_minimum()
        np.testing.assert_array_equal(point, [0.0, 0.0])
        assert value == -200.0

    def test_known_values(self):
        """Test against hand-computed reference values."""
        func = Ackley2Function()
        # f(1,1) = -200*exp(-0.02*sqrt(2))
        expected = -200.0 * np.exp(-0.02 * np.sqrt(2.0))
        assert np.isclose(func.evaluate(np.array([1.0, 1.0])), expected, atol=1e-10)

        # f(5,5) = -200*exp(-0.02*sqrt(50))
        expected2 = -200.0 * np.exp(-0.02 * np.sqrt(50.0))
        assert np.isclose(func.evaluate(np.array([5.0, 5.0])), expected2, atol=1e-10)

    def test_symmetry(self):
        """f(x,y) = f(-x,-y) = f(-x,y) = f(x,-y) due to squared terms."""
        func = Ackley2Function()
        x = np.array([3.0, 7.0])
        val = func.evaluate(x)
        assert np.isclose(func.evaluate(-x), val)
        assert np.isclose(func.evaluate(np.array([-3.0, 7.0])), val)
        assert np.isclose(func.evaluate(np.array([3.0, -7.0])), val)

    def test_monotonic_away_from_origin(self):
        """Value increases (toward 0) as we move away from origin."""
        func = Ackley2Function()
        v_close = func.evaluate(np.array([1.0, 0.0]))
        v_far = func.evaluate(np.array([10.0, 0.0]))
        assert v_close < v_far  # closer to -200 is smaller

    def test_batch_evaluation(self):
        func = Ackley2Function()
        X = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, -3.0]])
        batch = func.evaluate_batch(X)
        individual = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch, individual, atol=1e-12)

    def test_batch_shape(self):
        func = Ackley2Function()
        X = np.array([[0.0, 0.0], [1.0, 2.0]])
        assert func.evaluate_batch(X).shape == (2,)

    def test_dimension_validation(self):
        func = Ackley2Function()
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0]))

    def test_custom_bounds(self):
        bounds = Bounds(
            low=np.array([-10, -10]),
            high=np.array([10, 10]),
            mode=BoundModeEnum.OPERATIONAL,
        )
        func = Ackley2Function(initialization_bounds=bounds, operational_bounds=bounds)
        np.testing.assert_allclose(func.operational_bounds.low, [-10, -10])


# ---------------------------------------------------------------------------
# Ackley 3: f(x,y) = -200*exp(-0.02*sqrt(x^2+y^2)) + 5*exp(cos(3x)+sin(3y))
# ---------------------------------------------------------------------------
class TestAckley3Function:
    """Tests for Ackley 3 function."""

    def test_initialization_default(self):
        func = Ackley3Function()
        assert func.dimension == 2
        np.testing.assert_allclose(func.operational_bounds.low, [-32, -32])
        np.testing.assert_allclose(func.operational_bounds.high, [32, 32])

    def test_dimension_must_be_2(self):
        with pytest.raises(ValueError, match="dimension=2"):
            Ackley3Function(dimension=5)

    def test_value_at_origin(self):
        """f(0,0) = -200*exp(0) + 5*exp(cos(0)+sin(0)) = -200 + 5*exp(1)."""
        func = Ackley3Function()
        expected = -200.0 + 5.0 * np.exp(1.0)
        assert np.isclose(func.evaluate(np.array([0.0, 0.0])), expected, atol=1e-10)

    def test_global_minimum_region(self):
        """Global min is near (±0.6826, -0.3608) ≈ -195.6290."""
        func = Ackley3Function()
        val = func.evaluate(np.array([0.682584587365898, -0.36075325513719]))
        assert np.isclose(val, -195.62902823841935, atol=1e-6)
        # Negative x1 gives same value (symmetric in first exponential term)
        val_neg = func.evaluate(np.array([-0.682584587365898, -0.36075325513719]))
        assert np.isclose(val_neg, -195.62902823841935, atol=1e-6)

    def test_get_global_minimum(self):
        func = Ackley3Function()
        point, value = func.get_global_minimum()
        # Verify the reported minimum is actually achieved
        actual = func.evaluate(point)
        assert np.isclose(actual, value, atol=1e-6)
        # It should be near the known value
        assert value < -195.0

    def test_not_symmetric_in_y(self):
        """Ackley 3 is NOT symmetric in y due to sin(3y) term."""
        func = Ackley3Function()
        v1 = func.evaluate(np.array([1.0, 2.0]))
        v2 = func.evaluate(np.array([1.0, -2.0]))
        assert not np.isclose(v1, v2)

    def test_known_value(self):
        """f(0, -0.4) hand-computed."""
        func = Ackley3Function()
        x, y = 0.0, -0.4
        expected = -200.0 * np.exp(-0.02 * np.sqrt(x**2 + y**2)) + 5.0 * np.exp(
            np.cos(3 * x) + np.sin(3 * y)
        )
        assert np.isclose(func.evaluate(np.array([x, y])), expected, atol=1e-12)

    def test_batch_evaluation(self):
        func = Ackley3Function()
        X = np.array([[0.0, 0.0], [1.0, -0.5], [0.68, -0.36]])
        batch = func.evaluate_batch(X)
        individual = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch, individual, atol=1e-12)

    def test_batch_shape(self):
        func = Ackley3Function()
        X = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
        assert func.evaluate_batch(X).shape == (3,)

    def test_dimension_validation(self):
        func = Ackley3Function()
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0]))


# ---------------------------------------------------------------------------
# Ackley 4 (Modified Ackley):
#   f(x) = sum_{i=0}^{D-2} [exp(-0.2)*sqrt(x_i^2 + x_{i+1}^2)
#                            + 3*(cos(2*x_i) + sin(2*x_{i+1}))]
# ---------------------------------------------------------------------------
class TestAckley4Function:
    """Tests for Ackley 4 (Modified Ackley) function."""

    def test_initialization(self):
        func = Ackley4Function(dimension=2)
        assert func.dimension == 2
        np.testing.assert_allclose(func.operational_bounds.low, [-35, -35])
        np.testing.assert_allclose(func.operational_bounds.high, [35, 35])

    def test_scalable(self):
        """Ackley 4 is scalable — works with any dimension >= 2."""
        for d in [2, 5, 10, 50]:
            func = Ackley4Function(dimension=d)
            assert func.dimension == d
            val = func.evaluate(np.zeros(d))
            assert np.isfinite(val)

    def test_value_at_origin_d2(self):
        """f(0,0) = exp(-0.2)*sqrt(0) + 3*(cos(0)+sin(0)) = 3*(1+0) = 3.0."""
        func = Ackley4Function(dimension=2)
        assert np.isclose(func.evaluate(np.zeros(2)), 3.0, atol=1e-12)

    def test_value_at_origin_d5(self):
        """f(0,...,0) with D=5 has 4 terms, each = 3.0, total = 12.0."""
        func = Ackley4Function(dimension=5)
        assert np.isclose(func.evaluate(np.zeros(5)), 12.0, atol=1e-12)

    def test_known_values_d2(self):
        """Test specific known values for D=2."""
        func = Ackley4Function(dimension=2)
        # f(-1.51, -0.755) ≈ -4.5901
        val = func.evaluate(np.array([-1.51, -0.755]))
        assert np.isclose(val, -4.590100665150724, atol=1e-6)

        # Symmetric in first variable
        val2 = func.evaluate(np.array([1.51, -0.755]))
        # NOT symmetric due to cos(2*x_i) + sin(2*x_{i+1}) coupling
        # Actually cos is even so cos(2*1.51) = cos(2*(-1.51)), but sin(2*x2) stays same
        # So f(1.51, -0.755) = f(-1.51, -0.755) due to cos being even
        assert np.isclose(val, val2, atol=1e-12)

    def test_known_values_jamil_yang(self):
        """Values from Jamil & Yang 2013."""
        func = Ackley4Function(dimension=2)
        val = func.evaluate(np.array([-1.479252, -0.739807]))
        assert val < -4.5  # Close to global min

    def test_global_min_d2(self):
        """Global min for D=2 is approximately -4.59."""
        func = Ackley4Function(dimension=2)
        point, value = func.get_global_minimum()
        actual = func.evaluate(point)
        # The reported minimum should match evaluation
        assert np.isclose(actual, value, atol=1e-6)
        # Should be near -4.59
        assert value < -4.5

    def test_dimension_validation(self):
        func = Ackley4Function(dimension=3)
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0]))

    def test_batch_evaluation(self):
        func = Ackley4Function(dimension=2)
        X = np.array([[0.0, 0.0], [-1.51, -0.755], [1.0, 1.0]])
        batch = func.evaluate_batch(X)
        individual = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch, individual, atol=1e-12)

    def test_batch_evaluation_higher_dim(self):
        func = Ackley4Function(dimension=5)
        X = np.random.default_rng(42).uniform(-35, 35, size=(10, 5))
        batch = func.evaluate_batch(X)
        individual = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch, individual, atol=1e-10)

    def test_batch_shape(self):
        func = Ackley4Function(dimension=3)
        X = np.zeros((4, 3))
        assert func.evaluate_batch(X).shape == (4,)

    def test_custom_bounds(self):
        bounds = Bounds(
            low=np.full(3, -10),
            high=np.full(3, 10),
            mode=BoundModeEnum.OPERATIONAL,
        )
        func = Ackley4Function(dimension=3, initialization_bounds=bounds, operational_bounds=bounds)
        np.testing.assert_allclose(func.operational_bounds.low, [-10, -10, -10])

    def test_multimodal(self):
        """Ackley 4 is multimodal — multiple distinct local minima exist."""
        func = Ackley4Function(dimension=2)
        # Sample several points — should find different local basins
        rng = np.random.default_rng(123)
        values = [func.evaluate(rng.uniform(-35, 35, 2)) for _ in range(100)]
        # Not all values should be the same (multimodal)
        assert max(values) - min(values) > 1.0
