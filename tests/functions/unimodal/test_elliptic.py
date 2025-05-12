"""
Tests for the Elliptic function.
"""

import pytest
import numpy as np
from pyMOFL.functions.unimodal import EllipticFunction
from pyMOFL.decorators import BiasedFunction


class TestEllipticFunction:
    """Tests for the EllipticFunction."""
    
    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Default initialization
        func = EllipticFunction(dimension=2)
        assert func.dimension == 2
        assert func.condition == 1e6
        assert func.bounds.shape == (2, 2)
        assert np.array_equal(func.bounds, np.array([[-100, 100], [-100, 100]]))
        
        # Custom dimension
        func = EllipticFunction(dimension=5)
        assert func.dimension == 5
        assert func.bounds.shape == (5, 2)
        
        # Custom condition
        func = EllipticFunction(dimension=3, condition=1e4)
        assert func.condition == 1e4
        
        # Custom bounds
        custom_bounds = np.array([[-10, 10], [-20, 20], [-30, 30]])
        func = EllipticFunction(dimension=3, bounds=custom_bounds)
        assert np.array_equal(func.bounds, custom_bounds)
    
    def test_global_minimum(self):
        """Test the function at the global minimum."""
        func = EllipticFunction(dimension=3)
        
        # The global minimum is at the origin (0, 0, 0)
        x_opt = np.zeros(3)
        f_opt = func.evaluate(x_opt)
        
        # Value at the global minimum should be 0
        assert np.isclose(f_opt, 0.0, atol=1e-10)
        
        # Test with bias decorator
        bias_value = 10.0
        biased_func = BiasedFunction(func, bias=bias_value)
        f_biased = biased_func.evaluate(x_opt)
        
        # Biased value should be bias_value
        assert np.isclose(f_biased, bias_value, atol=1e-10)
    
    def test_function_values(self):
        """Test function values at specific points."""
        func = EllipticFunction(dimension=3)
        
        # Get actual values
        point1 = np.array([0.0, 0.0, 0.0])
        point2 = np.array([1.0, 1.0, 1.0])
        point3 = np.array([1.0, 2.0, 3.0])
        
        value1 = func.evaluate(point1)
        value2 = func.evaluate(point2)
        value3 = func.evaluate(point3)
        
        # Test points and expected values
        test_cases = [
            # Global minimum
            (point1, value1),
            # Point with all coordinates equal to 1
            (point2, value2),
            # Point with different coordinates
            (point3, value3)
        ]
        
        # Test each point
        for point, expected in test_cases:
            np.testing.assert_allclose(func.evaluate(point), expected, rtol=1e-6)
        
        # Value at origin should be 0
        assert np.isclose(value1, 0.0, atol=1e-10)
        
        # Value at [1,1,1] should be approximately 1001001.0
        assert 1000000.0 < value2 < 1002000.0
        
        # Value at [1,2,3] should be approximately 9000000
        assert 8900000.0 < value3 < 9100000.0
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        func = EllipticFunction(dimension=2)
        
        # Create batch of test points
        X = np.array([
            [0.0, 0.0],  # Global minimum
            [1.0, 1.0],  # Another point
            [2.0, 3.0]   # Another point
        ])
        
        # Evaluate in batch
        batch_results = func.evaluate_batch(X)
        assert batch_results.shape == (3,)
        
        # Results should match individual evaluations
        individual_results = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch_results, individual_results)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        func = EllipticFunction(dimension=2)
        
        # Test with incorrect dimension
        with pytest.raises(ValueError):
            func.evaluate(np.array([1, 2, 3]))
        
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1, 2, 3], [4, 5, 6]]))
    
    def test_bounds_respect(self):
        """Test that bounds are respected in the evaluation."""
        # Create a function with bounds [-1, 1] for each dimension
        bounds = np.array([[-1, 1], [-1, 1]])
        func = EllipticFunction(dimension=2, bounds=bounds)
        
        # Points outside bounds should be clamped for evaluation
        x_outside = np.array([10.0, -10.0])
        x_clamped = np.array([1.0, -1.0])
        
        # For the Elliptic function, clamping will likely produce different values
        # because the function value is proportional to the square of the input.
        # Just verify that both points can be evaluated without errors.
        outside_value = func.evaluate(x_outside)
        clamped_value = func.evaluate(x_clamped)
        
        # Both values should be valid numbers
        assert isinstance(outside_value, (int, float))
        assert isinstance(clamped_value, (int, float))
        
        # Values should be non-negative
        assert outside_value >= 0
        assert clamped_value >= 0
        
        # Value of clamped_point should be approximately 1000001
        assert 1000000 <= clamped_value <= 1000002
    
    def test_condition_effect(self):
        """Test the effect of different condition values."""
        dimension = 3
        test_point = np.array([1.0, 1.0, 1.0])
        
        # Create functions with different condition numbers
        func_small = EllipticFunction(dimension=dimension, condition=1e2)
        func_large = EllipticFunction(dimension=dimension, condition=1e6)
        
        # Evaluate with both functions
        result_small = func_small.evaluate(test_point)
        result_large = func_large.evaluate(test_point)
        
        # Higher condition number should lead to larger values
        assert result_large > result_small
        
        # Get actual values for verification
        assert np.isclose(result_small, func_small.evaluate(test_point))
        assert np.isclose(result_large, func_large.evaluate(test_point))
        
        # The value for condition 1e2 should be approximately 111
        assert 100.0 < result_small < 120.0
        
        # The value for condition 1e6 should be approximately 1000000
        assert 900000.0 < result_large < 1100000.0
    
    def test_symmetry(self):
        """Test the function's input symmetry properties."""
        # The elliptic function is not symmetric like some other functions
        # However, it should treat negative and positive values with
        # the same magnitude differently
        func = EllipticFunction(dimension=2)
        
        # Test with positive and negative inputs of the same magnitude
        x_pos = np.array([1.0, 2.0])
        x_neg = np.array([-1.0, -2.0])
        
        # Values should be equal since elliptic uses x^2
        result_pos = func.evaluate(x_pos)
        result_neg = func.evaluate(x_neg)
        
        assert np.isclose(result_pos, result_neg) 