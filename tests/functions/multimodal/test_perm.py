"""
Tests for the Perm function.
"""

import pytest
import numpy as np
from pyMOFL.functions.multimodal import PermFunction
from pyMOFL.decorators import BiasedFunction


class TestPermFunction:
    """Tests for the PermFunction."""
    
    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Default initialization
        func = PermFunction()
        assert func.dimension == 5
        assert func.beta == 0.5
        assert np.array_equal(func.bounds, np.array([[-5, 5]] * 5))
        
        # Custom beta
        func = PermFunction(beta=0.75)
        assert func.beta == 0.75
        
        # Custom bounds
        custom_bounds = np.array([[-10, 10]] * 5)
        func = PermFunction(bounds=custom_bounds)
        assert np.array_equal(func.bounds, custom_bounds)
    
    def test_evaluate_global_minimum(self):
        """Test the function value at the global minimum."""
        func = PermFunction()
        x_opt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # The function value at the global minimum should be zero
        assert func.evaluate(x_opt) == pytest.approx(0.0, abs=1e-10)
        
        # Check that the global minimum function returns the correct values
        global_min_point, global_min_value = PermFunction.get_global_minimum()
        np.testing.assert_allclose(global_min_point, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert global_min_value == 0.0
        
        # Test with a bias using decorator
        bias_value = 0.1
        biased_func = BiasedFunction(func, bias=bias_value)
        assert biased_func.evaluate(x_opt) == pytest.approx(bias_value, abs=1e-10)
    
    def test_integer_rounding(self):
        """Test that the function properly rounds inputs to integers."""
        func = PermFunction()
        
        # Test point that gets rounded to the global minimum
        x_almost = np.array([1.1, 1.9, 3.2, 4.1, 4.7])
        x_rounded = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # The nearly-optimal point should give the same result as the optimal point
        assert func.evaluate(x_almost) == pytest.approx(func.evaluate(x_rounded), abs=1e-10)
    
    def test_non_optimal_points(self):
        """Test the function value at non-optimal points."""
        func = PermFunction()
        
        # Test a few non-optimal points
        x1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        assert func.evaluate(x1) > 0.0
        
        x2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Reversed optimal point
        assert func.evaluate(x2) > 0.0
        
        x3 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        assert func.evaluate(x3) > 0.0
        
        # Test with bias using decorator
        bias_value = 5.0
        biased_func = BiasedFunction(func, bias=bias_value)
        assert biased_func.evaluate(x1) == pytest.approx(func.evaluate(x1) + bias_value)
        assert biased_func.evaluate(x2) == pytest.approx(func.evaluate(x2) + bias_value)
        assert biased_func.evaluate(x3) == pytest.approx(func.evaluate(x3) + bias_value)
    
    def test_evaluate_batch(self):
        """Test the batch evaluation method."""
        func = PermFunction()
        
        # Create a batch of test points
        batch_size = 10
        rng = np.random.default_rng(42)
        X = rng.uniform(-5, 5, size=(batch_size, 5))
        
        # Include the global minimum in the batch
        X[0] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Evaluate the batch
        results = func.evaluate_batch(X)
        
        # Check results
        assert results.shape == (batch_size,)
        assert results[0] == pytest.approx(0.0, abs=1e-10)
        
        # Compare with individual evaluations
        for i in range(batch_size):
            assert results[i] == pytest.approx(func.evaluate(X[i]), abs=1e-10)
            
        # Test with bias using decorator
        bias_value = 2.5
        biased_func = BiasedFunction(func, bias=bias_value)
        biased_results = biased_func.evaluate_batch(X)
        np.testing.assert_allclose(biased_results, results + bias_value)
    
    def test_dimension_validation(self):
        """Test that the function correctly validates the input dimension."""
        func = PermFunction()
        
        # Test with incorrect input dimensions
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0]))
        
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            
        # Test get_global_minimum with incorrect dimension
        with pytest.raises(ValueError):
            PermFunction.get_global_minimum(dimension=3)
    
    def test_non_negativity(self):
        """Test that the function value is always non-negative."""
        func = PermFunction()
        
        # Generate random test points
        rng = np.random.default_rng(42)
        points = rng.uniform(-5, 5, size=(100, 5))
        
        # Evaluate the function at all points
        values = func.evaluate_batch(points)
        
        # Check that all values are non-negative
        assert np.all(values >= 0) 