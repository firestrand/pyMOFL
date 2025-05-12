"""
Tests for the Tripod function.
"""

import pytest
import numpy as np
from pyMOFL.functions.multimodal import TripodFunction
from pyMOFL.decorators import BiasedFunction


class TestTripodFunction:
    """Tests for the Tripod function."""
    
    def test_initialization(self):
        """Test initialization with default and custom bounds."""
        # Test with default bounds
        func = TripodFunction()
        assert func.dimension == 2
        assert func.bounds.shape == (2, 2)
        assert np.array_equal(func.bounds, np.array([[-100, 100], [-100, 100]]))
        
        # Test with custom bounds
        custom_bounds = np.array([[-10, 10], [-5, 5]])
        func = TripodFunction(bounds=custom_bounds)
        assert func.dimension == 2
        assert func.bounds.shape == (2, 2)
        assert np.array_equal(func.bounds, custom_bounds)
    
    def test_global_minimum(self):
        """Test the function at its global minimum."""
        func = TripodFunction()
        
        # Get global minimum
        min_point, min_value = TripodFunction.get_global_minimum()
        
        # Verify global minimum point and value
        np.testing.assert_allclose(min_point, np.array([0, -50]))
        assert min_value == 0.0
        
        # Test evaluation at global minimum
        np.testing.assert_allclose(func.evaluate(min_point), min_value, atol=1e-10)
        
        # Test with bias
        bias_value = 10.0
        biased_func = BiasedFunction(func, bias=bias_value)
        np.testing.assert_allclose(biased_func.evaluate(min_point), min_value + bias_value, atol=1e-10)
    
    def test_function_values(self):
        """Test function values at specific points in different quadrants."""
        func = TripodFunction()
        
        # Test points and expected values
        test_cases = [
            # Global minimum
            (np.array([0, -50]), 0.0),
            
            # Quadrant I: x1 >= 0, x2 >= 0
            (np.array([5, 10]), 87.0),
            
            # Quadrant II: x1 < 0, x2 >= 0
            (np.array([-5, 10]), 86.0),
            
            # Quadrant III: x1 < 0, x2 < 0
            (np.array([-5, -10]), 45.0),
            
            # Quadrant IV: x1 >= 0, x2 < 0
            (np.array([5, -10]), 45.0)
        ]
        
        # Test each point
        for point, expected in test_cases:
            np.testing.assert_allclose(func.evaluate(point), expected, atol=1e-14)
        
        # Test with bias decorator
        bias_value = 15.0
        biased_func = BiasedFunction(func, bias=bias_value)
        for point, expected in test_cases:
            np.testing.assert_allclose(biased_func.evaluate(point), expected + bias_value, atol=1e-14)
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        func = TripodFunction()
        
        # Create a batch of test points from different quadrants
        points = np.array([
            [0, -50],  # global minimum
            [5, 10],   # quadrant I
            [-5, 10],  # quadrant II
            [-5, -10], # quadrant III
            [5, -10]   # quadrant IV
        ])
        
        # Get expected values by individual evaluation
        expected = np.array([
            func.evaluate(points[0]),
            func.evaluate(points[1]),
            func.evaluate(points[2]),
            func.evaluate(points[3]),
            func.evaluate(points[4])
        ])
        
        # Test batch evaluation
        np.testing.assert_allclose(func.evaluate_batch(points), expected)
        
        # Test with bias decorator
        bias_value = 7.5
        biased_func = BiasedFunction(func, bias=bias_value)
        biased_results = biased_func.evaluate_batch(points)
        np.testing.assert_allclose(biased_results, expected + bias_value)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        func = TripodFunction()
        
        # Test with incorrect dimension
        with pytest.raises(ValueError):
            func.evaluate(np.array([1, 2, 3]))
        
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1, 2, 3], [4, 5, 6]]))
            
        # Test global_minimum with wrong dimension
        with pytest.raises(ValueError):
            TripodFunction.get_global_minimum(dimension=3)
    
    def test_non_negativity(self):
        """Test that function values are non-negative for points in the domain."""
        func = TripodFunction()
        
        # Generate random points in the domain
        rng = np.random.default_rng(42)
        points = rng.uniform(-100, 100, size=(100, 2))
        
        # Evaluate function at these points
        values = func.evaluate_batch(points)
        
        # Check that all values are non-negative
        assert np.all(values >= 0)
    
    def test_bounds_respect(self):
        """Test that bounds are properly respected."""
        # Create function with specific bounds
        bounds = np.array([[-50, 50], [-50, 50]])
        func = TripodFunction(bounds=bounds)
        
        # Points at the bounds
        edge_points = [
            np.array([-50, 0]),
            np.array([50, 0]),
            np.array([0, -50]),
            np.array([0, 50])
        ]
        
        # All points should be valid and evaluate without error
        for point in edge_points:
            # This should execute without error
            value = func.evaluate(point)
            assert isinstance(value, (int, float))
            assert value >= 0 