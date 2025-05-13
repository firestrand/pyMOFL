"""
Tests for the ScaledFunction decorator.
"""

import pytest
import numpy as np
from pyMOFL.functions.unimodal import SphereFunction
from pyMOFL.decorators import ScaledFunction


class TestScaledFunction:
    """Tests for the ScaledFunction decorator."""
    
    def test_initialization(self):
        """Test initialization with scalar and vector lambda coefficients."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Test with scalar lambda
        lambda_scalar = 2.0
        scaled_func = ScaledFunction(base_func, lambda_scalar)
        assert scaled_func.dimension == 2
        assert np.array_equal(scaled_func.lambda_coef, np.array([2.0, 2.0]))
        
        # Test bounds scaling with positive lambda
        # SphereFunction uses the default bounds [-100, 100]
        expected_bounds = np.array([[-200, 200], [-200, 200]])
        assert np.allclose(scaled_func.bounds, expected_bounds)
        
        # Test with vector lambda
        lambda_vector = np.array([2.0, 3.0])
        scaled_func = ScaledFunction(base_func, lambda_vector)
        assert scaled_func.dimension == 2
        assert np.array_equal(scaled_func.lambda_coef, lambda_vector)
        
        # Test bounds scaling with mixed lambdas
        expected_bounds = np.array([[-200, 200], [-300, 300]])
        assert np.allclose(scaled_func.bounds, expected_bounds)
        
        # Test with negative lambda
        lambda_negative = np.array([-2.0, 3.0])
        scaled_func = ScaledFunction(base_func, lambda_negative)
        # For negative lambda, bounds are flipped: lower_bound = upper_bound * lambda, upper_bound = lower_bound * lambda
        expected_bounds = np.array([[-200, 200], [-300, 300]])  # Signs flipped for negative lambda index
        assert np.allclose(scaled_func.bounds, expected_bounds)
    
    def test_lambda_validation(self):
        """Test that lambda coefficients are validated correctly."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Test with incorrect lambda dimension
        invalid_lambda = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            ScaledFunction(base_func, invalid_lambda)
    
    def test_evaluate(self):
        """Test the evaluate method."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Create a scaled function with scalar lambda
        lambda_scalar = 2.0
        scaled_func = ScaledFunction(base_func, lambda_scalar)
        
        # Test at origin (optimum is invariant under scaling)
        assert scaled_func.evaluate(np.array([0.0, 0.0])) == 0
        
        # Test at arbitrary point
        # f(2,4) = base_func(2/2, 4/2) = base_func(1, 2) = 1^2 + 2^2 = 5
        assert scaled_func.evaluate(np.array([2.0, 4.0])) == 5
        
        # Create a scaled function with vector lambda
        lambda_vector = np.array([2.0, 4.0])
        scaled_func = ScaledFunction(base_func, lambda_vector)
        
        # Test at origin
        assert scaled_func.evaluate(np.array([0.0, 0.0])) == 0
        
        # Test at arbitrary point
        # f(2,4) = base_func(2/2, 4/4) = base_func(1, 1) = 1^2 + 1^2 = 2
        assert scaled_func.evaluate(np.array([2.0, 4.0])) == 2
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Create a scaled function
        lambda_vector = np.array([2.0, 4.0])
        scaled_func = ScaledFunction(base_func, lambda_vector)
        
        # Test with batch of vectors
        X = np.array([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]])
        expected = np.array([0, 2, 8])
        assert np.array_equal(scaled_func.evaluate_batch(X), expected)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Create a scaled function
        lambda_scalar = 2.0
        scaled_func = ScaledFunction(base_func, lambda_scalar)
        
        # Test with incorrect dimension
        with pytest.raises(ValueError):
            scaled_func.evaluate(np.array([1.0, 2.0, 3.0]))
        
        with pytest.raises(ValueError):
            scaled_func.evaluate_batch(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    
    def test_composition_with_shift(self):
        """Test that scaling can be composed with shifting."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Create a shifted function first
        from pyMOFL.decorators import ShiftedFunction
        shift = np.array([1.0, 2.0])
        shifted_func = ShiftedFunction(base_func, shift)
        
        # Apply scaling on top of shifting
        lambda_scalar = 2.0
        scaled_shifted_func = ScaledFunction(shifted_func, lambda_scalar)
        
        # Test at scaled and shifted optimum
        # The original optimum at shift=[1,2] should now be at [2,4]
        assert scaled_shifted_func.evaluate(np.array([2.0, 4.0])) == 0
        
        # Test at origin
        # f(0,0) = shifted_func(0/2, 0/2) = shifted_func(0, 0)
        #        = base_func(0-1, 0-2) = base_func(-1, -2)
        #        = (-1)^2 + (-2)^2 = 5
        assert scaled_shifted_func.evaluate(np.array([0.0, 0.0])) == 5 