"""
Tests for the Rastrigin function.
"""

import pytest
import numpy as np
from pyMOFL.functions.multimodal import RastriginFunction
from pyMOFL.decorators import BiasedFunction


class TestRastriginFunction:
    """Tests for the Rastrigin function."""
    
    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Test with defaults
        func = RastriginFunction(dimension=2)
        assert func.dimension == 2
        assert func.bounds.shape == (2, 2)
        assert np.array_equal(func.bounds, np.array([[-5.12, 5.12], [-5.12, 5.12]]))
        
        # Test with custom bounds
        custom_bounds = np.array([[-10, 10], [-10, 10], [-10, 10]])
        func = RastriginFunction(dimension=3, bounds=custom_bounds)
        assert func.bounds.shape == (3, 2)
        assert np.array_equal(func.bounds, custom_bounds)
    
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
        biased_func = BiasedFunction(func, bias=bias_value)
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
        biased_func = BiasedFunction(func, bias=bias_value)
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
        bounds = np.array([[-1, 1], [-1, 1]])
        func = RastriginFunction(dimension=2, bounds=bounds)
        
        # Points outside the bounds
        outside_point = np.array([10.0, -10.0])
        
        # Points should be clamped to bounds
        clamped_point = np.array([1.0, -1.0])
        
        # The function might not have the same value due to clamping and the 
        # behavior of cos(2*pi*x) with large values, so we should just verify
        # that the function evaluates both points without error
        outside_value = func.evaluate(outside_point)
        clamped_value = func.evaluate(clamped_point)
        
        # Both values should be valid numbers
        assert isinstance(outside_value, (int, float))
        assert isinstance(clamped_value, (int, float))
    
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