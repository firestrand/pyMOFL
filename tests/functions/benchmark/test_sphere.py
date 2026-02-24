"""
Tests for the Sphere function (refactored for new bounds logic).
"""

import numpy as np
import pytest

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.functions import SphereFunction


class TestSphereFunction:
    """Tests for the Sphere function."""

    def test_initialization_defaults(self):
        """Test initialization with default bounds."""
        func = SphereFunction(dimension=2)
        assert func.dimension == 2
        assert isinstance(func.initialization_bounds, Bounds)
        assert isinstance(func.operational_bounds, Bounds)
        np.testing.assert_allclose(func.initialization_bounds.low, [-100, -100])
        np.testing.assert_allclose(func.initialization_bounds.high, [100, 100])
        np.testing.assert_allclose(func.operational_bounds.low, [-100, -100])
        np.testing.assert_allclose(func.operational_bounds.high, [100, 100])

    def test_initialization_custom_bounds(self):
        """Test initialization with custom bounds."""
        init_bounds = Bounds(
            low=np.array([-10, -5]), high=np.array([10, 5]), mode=BoundModeEnum.INITIALIZATION
        )
        op_bounds = Bounds(
            low=np.array([-1, -2]), high=np.array([1, 2]), mode=BoundModeEnum.OPERATIONAL
        )
        func = SphereFunction(
            dimension=2, initialization_bounds=init_bounds, operational_bounds=op_bounds
        )
        np.testing.assert_allclose(func.initialization_bounds.low, [-10, -5])
        np.testing.assert_allclose(func.initialization_bounds.high, [10, 5])
        np.testing.assert_allclose(func.operational_bounds.low, [-1, -2])
        np.testing.assert_allclose(func.operational_bounds.high, [1, 2])

    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        func = SphereFunction(dimension=2)

        # Test with batch of vectors
        X = np.array([[0, 0], [1, 1], [2, 3]])
        expected = np.array([0, 2, 13])
        np.testing.assert_allclose(func.evaluate_batch(X), expected)

    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        func = SphereFunction(dimension=2)

        # Test with incorrect dimension
        with pytest.raises(ValueError):
            func.evaluate(np.array([1, 2, 3]))

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_global_minimum(self):
        """Test the global minimum function."""
        for dim in [2, 5, 10]:
            # Get the global minimum
            point, value = SphereFunction.get_global_minimum(dim)

            # Check the point shape and value
            assert point.shape == (dim,)
            assert np.all(point == 0)
            assert value == 0.0

            # Verify that evaluating at the global minimum gives the expected value
            func = SphereFunction(dimension=dim)
            assert np.isclose(func.evaluate(point), value)

            # Check with bias
            # bias_value = 3.0
            # biased_func = BiasWrapper(inner_function=func, bias=bias_value)
            # assert np.isclose(biased_func.evaluate(point), value + bias_value)
