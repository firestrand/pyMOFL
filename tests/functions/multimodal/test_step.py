"""
Tests for the Step function.
"""

import pytest
import numpy as np
from pyMOFL.functions.multimodal import StepFunction
from pyMOFL.decorators import BiasedFunction


class TestStepFunction:
    """Tests for the Step function."""
    
    def test_initialization(self):
        """Test initialization with default and custom bounds."""
        # Test with defaults
        func = StepFunction()
        assert func.dimension == 10
        assert func.bounds.shape == (10, 2)
        assert np.array_equal(func.bounds, np.array([[-100, 100]] * 10))
        
        # Test with BiasedFunction decorator
        bias_value = 30.0
        biased_func = BiasedFunction(func, bias=bias_value)
        assert biased_func.bias == 30.0
        
        # Test with custom bounds
        custom_bounds = np.array([[-10, 10]] * 10)
        func = StepFunction(bounds=custom_bounds)
        assert func.bounds.shape == (10, 2)
        assert np.array_equal(func.bounds, custom_bounds)
    
    def test_evaluate_global_minimum(self):
        """Test the evaluate method at global minimum."""
        # Create base function
        func = StepFunction()
        
        # Create biased function with decorator
        bias_value = 30.0
        biased_func = BiasedFunction(func, bias=bias_value)
        
        # Test at origin (within optimal region [-0.5, 0.5))
        result = biased_func.evaluate(np.zeros(10))
        assert result == 30.0
        
        # Test at another point within optimal region
        x = np.array([-0.49, -0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.49, -0.25])
        result = biased_func.evaluate(x)
        assert result == 30.0
        
        # Test at boundary of optimal region (-0.5 is not in the region)
        x = np.zeros(10)
        x[0] = -0.5
        result = biased_func.evaluate(x)
        assert result == 30.0  # 0^2 + 0^2 + ... + 0^2 + bias
    
    def test_evaluate_stepped_values(self):
        """Test that floor(x + 0.5) works correctly."""
        # Create base function
        func = StepFunction()
        
        # Create biased function with decorator
        bias_value = 30.0
        biased_func = BiasedFunction(func, bias=bias_value)
        
        # Test with various inputs to check the step function behavior
        test_cases = [
            # x value, expected stepped value
            (-1.7, -2.0),
            (-1.5, -1.0),
            (-1.3, -1.0),
            (-0.7, -1.0),
            (-0.5, 0.0),
            (-0.3, 0.0),
            (0.0, 0.0),
            (0.3, 0.0),
            (0.5, 1.0),
            (0.7, 1.0),
            (1.3, 1.0),
            (1.5, 2.0),
            (1.7, 2.0)
        ]
        
        for x_val, expected_step in test_cases:
            # Create a point with one non-zero coordinate
            x = np.zeros(10)
            x[0] = x_val
            
            # Expected result is the square of the stepped value plus bias
            expected = expected_step**2 + bias_value
            
            # Test
            result = biased_func.evaluate(x)
            assert result == expected, f"Failed for x={x_val}, expected={expected}, got={result}"
    
    def test_evaluate_multiple_steps(self):
        """Test with multiple non-zero coordinates."""
        # Create base function
        func = StepFunction()
        
        # Create biased function with decorator
        bias_value = 30.0
        biased_func = BiasedFunction(func, bias=bias_value)
        
        # Create a test case with multiple stepped values
        x = np.array([1.7, -0.3, 0.6, -1.7, 0.0, 2.3, -2.8, 3.5, -3.2, 4.1])
        
        # Calculate expected stepped values
        stepped = np.floor(x + 0.5)  # Should be [2, 0, 1, -2, 0, 2, -3, 4, -3, 4]
        expected = np.sum(stepped**2) + bias_value
        
        # Test
        result = biased_func.evaluate(x)
        assert result == expected
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        # Create base function
        func = StepFunction()
        
        # Create biased function with decorator
        bias_value = 30.0
        biased_func = BiasedFunction(func, bias=bias_value)
        
        # Create a batch of test points
        batch = np.array([
            np.zeros(10),  # Global minimum
            np.ones(10),   # All 1's --> stepped to 1's
            np.array([1.7, -0.3, 0.6, -1.7, 0.0, 2.3, -2.8, 3.5, -3.2, 4.1])
        ])
        
        # Calculate expected values
        expected = np.array([
            30.0,  # Global minimum: 0^2 + ... + 0^2 + bias
            40.0,  # All 1's: 10*1^2 + bias
            np.sum(np.floor(batch[2] + 0.5)**2) + bias_value  # Complex case
        ])
        
        # Test
        results = biased_func.evaluate_batch(batch)
        np.testing.assert_allclose(results, expected)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        func = StepFunction()
        
        # Test with incorrect dimension
        with pytest.raises(ValueError):
            func.evaluate(np.array([1, 2, 3]))
        
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1, 2, 3], [4, 5, 6]]))
    
    def test_non_negativity(self):
        """Test that function values are non-negative for base function and >= bias for biased function."""
        # Create base function
        func = StepFunction()
        
        # Create biased function with decorator
        bias_value = 30.0
        biased_func = BiasedFunction(func, bias=bias_value)
        
        # Generate random points in the domain
        rng = np.random.default_rng(42)
        points = rng.uniform(-100, 100, size=(100, 10))
        
        # Evaluate base function at these points
        base_values = func.evaluate_batch(points)
        
        # Evaluate biased function at these points
        biased_values = biased_func.evaluate_batch(points)
        
        # Check that all base values are >= 0
        assert (base_values >= 0).all()
        
        # Check that all biased values are >= bias
        assert (biased_values >= bias_value).all()
        
        # Check that biased values = base values + bias
        np.testing.assert_allclose(biased_values, base_values + bias_value) 