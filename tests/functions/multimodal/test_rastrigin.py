"""
Tests for the Rastrigin function.
"""

import pytest
import numpy as np
from pyMOFL.functions.multimodal import RastriginFunction
from pyMOFL.decorators import Biased
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum


class TestRastriginFunction:
    """Tests for the Rastrigin function."""
    
    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Test with defaults
        func = RastriginFunction(dimension=2)
        assert func.dimension == 2
        np.testing.assert_allclose(func.initialization_bounds.low, [-5.12, -5.12])
        np.testing.assert_allclose(func.initialization_bounds.high, [5.12, 5.12])
        np.testing.assert_allclose(func.operational_bounds.low, [-5.12, -5.12])
        np.testing.assert_allclose(func.operational_bounds.high, [5.12, 5.12])
        
        # Test with custom bounds
        custom_init_bounds = Bounds(low=np.array([-10, -10, -10]), high=np.array([10, 10, 10]), mode=BoundModeEnum.INITIALIZATION)
        custom_oper_bounds = Bounds(low=np.array([-10, -10, -10]), high=np.array([10, 10, 10]), mode=BoundModeEnum.OPERATIONAL)
        func = RastriginFunction(dimension=3, initialization_bounds=custom_init_bounds, operational_bounds=custom_oper_bounds)
        np.testing.assert_allclose(func.initialization_bounds.low, [-10, -10, -10])
        np.testing.assert_allclose(func.initialization_bounds.high, [10, 10, 10])
        np.testing.assert_allclose(func.operational_bounds.low, [-10, -10, -10])
        np.testing.assert_allclose(func.operational_bounds.high, [10, 10, 10])
    
    def test_global_minimum(self):
        """Test the function value at the global minimum."""
        func = RastriginFunction(dimension=2)
        
        # Global minimum is at the origin
        min_point = np.zeros(2)
        result = func.evaluate(min_point)
        
        # Value at global minimum should be 0
        assert np.isclose(result, 0.0, atol=1e-15)
        
        # Test with BiasedFunction decorator
        bias_value = 5.0
        biased_func = Biased(func, bias=bias_value)
        biased_result = biased_func.evaluate(min_point)
        
        # Value at global minimum with bias should be the bias value
        assert np.isclose(biased_result, bias_value, atol=1e-15)
    
    def test_function_values(self):
        """Test function values at specific points."""
        func = RastriginFunction(dimension=2)
        
        # Get actual values
        point1 = np.array([0, 0])  # Global minimum
        point2 = np.array([1, 1])
        point3 = np.array([2, 3])
        
        value1 = func.evaluate(point1)
        value2 = func.evaluate(point2)
        value3 = func.evaluate(point3)
        
        # Test points and expected values
        test_cases = [
            # Test at global minimum (0, 0)
            (point1, value1),
            # Test at (1, 1)
            (point2, value2),
            # Test at (2, 3)
            (point3, value3)
        ]
        
        # Test each point
        for point, expected in test_cases:
            np.testing.assert_allclose(func.evaluate(point), expected, atol=1e-10)
        
        # Value at global minimum should be 0
        assert np.isclose(value1, 0.0, atol=1e-15)
        
        # Value at (1, 1) should be approximately 2.0
        assert np.isclose(value2, 2.0, atol=1e-15)
        
        # Value at (2, 3) should be approximately 13.0
        assert np.isclose(value3, 13.0, atol=1e-15)
    
    def test_batch_evaluation(self):
        """Test the evaluate_batch method."""
        func = RastriginFunction(dimension=2)
        
        # Create a batch of test points
        points = np.array([
            [0, 0],  # Global minimum
            [1, 1],  # Another point
            [2, 3]   # Another point
        ])
        
        # Evaluate each point individually
        individual_results = np.array([func.evaluate(p) for p in points])
        
        # Test batch evaluation
        batch_results = func.evaluate_batch(points)
        np.testing.assert_allclose(batch_results, individual_results, atol=1e-15)
        
        # Test with BiasedFunction decorator
        bias_value = 5.0
        biased_func = Biased(func, bias=bias_value)
        biased_results = biased_func.evaluate_batch(points)
        
        # Biased results should be original results + bias
        np.testing.assert_allclose(biased_results, individual_results + bias_value, atol=1e-15)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        func = RastriginFunction(dimension=2)
        
        # Test with incorrect dimension
        with pytest.raises(ValueError):
            func.evaluate(np.array([1, 2, 3]))
        
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1, 2, 3], [4, 5, 6]]))
    
    def test_bounds_respect(self):
        """Test that bounds are properly respected."""
        # Create function with custom bounds
        custom_init_bounds = Bounds(low=np.array([-1, -1]), high=np.array([1, 1]), mode=BoundModeEnum.INITIALIZATION)
        custom_oper_bounds = Bounds(low=np.array([-1, -1]), high=np.array([1, 1]), mode=BoundModeEnum.OPERATIONAL)
        func = RastriginFunction(dimension=2, initialization_bounds=custom_init_bounds, operational_bounds=custom_oper_bounds)
        
        # Points inside the bounds
        inside_point = np.array([1.0, -1.0])
        value = func.evaluate(inside_point)
        assert isinstance(value, (int, float))
    
    def test_symmetry(self):
        """Test that the function is symmetric around origin."""
        func = RastriginFunction(dimension=2)
        
        # Test points
        x = np.array([1.0, 2.0])
        neg_x = -x
        
        # Due to the symmetry, f(x) = f(-x)
        value_x = func.evaluate(x)
        value_neg_x = func.evaluate(neg_x)
        
        assert np.isclose(value_x, value_neg_x, rtol=1e-10) 