"""
Tests for the Compression Spring function.
"""

import pytest
import numpy as np
from pyMOFL.functions.hybrid import CompressionSpringFunction
from pyMOFL.decorators import BiasedFunction


class TestCompressionSpringFunction:
    """Tests for the CompressionSpringFunction."""
    
    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Default initialization
        func = CompressionSpringFunction()
        assert func.dimension == 3
        assert func._BIG == 1.0e5
        assert func._scaling == 0.4234
        
        # Check default bounds
        expected_bounds = np.array([
            [0.05, 2.00],  # d: wire diameter
            [0.25, 1.30],  # D: mean coil diameter
            [2, 15]        # N: active coil count
        ])
        assert np.array_equal(func.bounds, expected_bounds)
        
        # Custom bounds
        custom_bounds = np.array([
            [0.1, 1.5],   # d
            [0.3, 1.0],   # D
            [3, 10]       # N
        ])
        func = CompressionSpringFunction(bounds=custom_bounds)
        assert np.array_equal(func.bounds, custom_bounds)
    
    def test_evaluate_global_minimum(self):
        """Test the function value at the known global minimum."""
        func = CompressionSpringFunction()
        
        # Known global minimum
        x_opt = np.array([0.05150, 0.35166, 11])
        
        # Function value at global minimum should be approximately 0.012665
        result = func.evaluate(x_opt)
        expected_value = 0.012665
        assert np.isclose(result, expected_value, atol=1e-5)
        
        # Test with bias decorator
        bias_value = 1.0
        biased_func = BiasedFunction(func, bias=bias_value)
        biased_result = biased_func.evaluate(x_opt)
        assert np.isclose(biased_result, result + bias_value, atol=1e-5)
    
    def test_integer_rounding(self):
        """Test that the function properly rounds the third coordinate to an integer."""
        func = CompressionSpringFunction()
        
        # Test points with non-integer N that should be rounded
        x1 = np.array([0.05150, 0.35166, 10.7])
        x2 = np.array([0.05150, 0.35166, 10.2])
        
        # Create points with explicitly rounded values
        x1_rounded = np.array([0.05150, 0.35166, 11.0])
        x2_rounded = np.array([0.05150, 0.35166, 10.0])
        
        # The points with same rounded values should produce the same results
        assert np.isclose(func.evaluate(x1), func.evaluate(x1_rounded), atol=1e-10)
        assert np.isclose(func.evaluate(x2), func.evaluate(x2_rounded), atol=1e-10)
        
        # Values should differ when rounded to different integers
        assert func.evaluate(x1) != func.evaluate(x2)
    
    def test_constraint_violations(self):
        """Test that constraint violations are properly penalized."""
        func = CompressionSpringFunction()
        
        # Generate a known infeasible point by using extreme values
        # that violate at least one constraint
        infeasible_point = np.array([0.05, 1.30, 2])  # Extreme values
        
        # Evaluate the infeasible point
        result = func.evaluate(infeasible_point)
        
        # The result should be much higher than the global minimum due to penalty
        assert result > 1.0  # Global minimum is around 0.012665
        
        # The value should include a penalty term
        # Calculate the weight alone for this point
        d, D, N = 0.05, 1.30, 2
        weight = func._scaling * np.pi**2 * D * d**2 * (N + 2) * 0.25
        assert result > weight  # The result should include penalty on top of weight
    
    def test_bounds_clipping(self):
        """Test that input values are properly clipped to the bounds."""
        func = CompressionSpringFunction()
        
        # Test point outside the bounds
        out_of_bounds = np.array([0.01, 2.0, 20])
        
        # Expected clipped values
        clipped = np.array([0.05, 1.30, 15])
        
        # Both should give the same result after clipping
        result_out_of_bounds = func.evaluate(out_of_bounds)
        result_clipped = func.evaluate(clipped)
        
        assert np.isclose(result_out_of_bounds, result_clipped, atol=1e-10)
    
    def test_evaluate_batch(self):
        """Test the batch evaluation method."""
        func = CompressionSpringFunction()
        
        # Create a batch of test points
        points = np.array([
            [0.05150, 0.35166, 11],  # Global minimum
            [0.06, 0.40, 10],        # Another point
            [0.07, 0.45, 9],         # Another point
            [0.05, 1.30, 2]          # Infeasible point
        ])
        
        # Calculate expected values individually
        expected = np.array([
            func.evaluate(points[0]),
            func.evaluate(points[1]),
            func.evaluate(points[2]),
            func.evaluate(points[3])
        ])
        
        # Evaluate the batch
        batch_results = func.evaluate_batch(points)
        
        # Check that the batch has the right shape
        assert batch_results.shape == (4,)
        
        # Check that batch results match individual evaluations
        np.testing.assert_allclose(batch_results, expected, atol=1e-10)
        
        # Test with bias decorator
        bias_value = 2.0
        biased_func = BiasedFunction(func, bias=bias_value)
        biased_results = biased_func.evaluate_batch(points)
        np.testing.assert_allclose(biased_results, batch_results + bias_value, atol=1e-10)
    
    def test_dimension_validation(self):
        """Test that the function correctly validates input dimensions."""
        func = CompressionSpringFunction()
        
        # Test with incorrect dimensions
        with pytest.raises(ValueError):
            func.evaluate(np.array([0.05, 0.35]))  # Too few dimensions
        
        with pytest.raises(ValueError):
            func.evaluate(np.array([0.05, 0.35, 11, 1.0]))  # Too many dimensions
        
        # Test batch with incorrect dimensions
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[0.05, 0.35], [0.06, 0.40]]))  # Too few columns
        
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[0.05, 0.35, 11, 1.0], [0.06, 0.40, 10, 2.0]]))  # Too many columns
    
    def test_non_negativity(self):
        """Test that function values are non-negative for feasible points."""
        func = CompressionSpringFunction()
        
        # Generate random points within the bounds
        rng = np.random.default_rng(42)
        
        # Generate 50 random points
        d = rng.uniform(0.05, 2.00, size=50)
        D = rng.uniform(0.25, 1.30, size=50)
        N = rng.integers(2, 15, size=50)
        
        points = np.column_stack((d, D, N))
        
        # Evaluate all points
        results = func.evaluate_batch(points)
        
        # All function values should be non-negative
        assert np.all(results >= 0) 