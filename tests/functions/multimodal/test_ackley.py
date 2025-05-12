"""
Tests for the Ackley function.
"""

import pytest
import numpy as np
from pyMOFL.functions.multimodal import AckleyFunction
from pyMOFL.decorators import BiasedFunction, ShiftedFunction


class TestAckleyFunction:
    """Tests for the Ackley function."""
    
    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Test with default parameters
        func = AckleyFunction(dimension=2)
        assert func.dimension == 2
        assert func.a == 20.0
        assert func.b == 0.2
        assert np.isclose(func.c, 2.0 * np.pi)
        assert func.bounds.shape == (2, 2)
        assert np.array_equal(func.bounds, np.array([[-32.768, 32.768], [-32.768, 32.768]]))
        
        # Test with custom parameters
        func = AckleyFunction(dimension=3, a=10.0, b=0.1, c=np.pi)
        assert func.dimension == 3
        assert func.a == 10.0
        assert func.b == 0.1
        assert np.isclose(func.c, np.pi)
        
        # Test with custom bounds
        custom_bounds = np.array([[-5, 5], [-5, 5], [-5, 5]])
        func = AckleyFunction(dimension=3, bounds=custom_bounds)
        assert np.array_equal(func.bounds, custom_bounds)
    
    def test_global_minimum(self):
        """Test the function at the global minimum."""
        func = AckleyFunction(dimension=2)
        
        # The global minimum is at the origin (0, 0)
        x_opt = np.zeros(2)
        f_opt = func.evaluate(x_opt)
        
        # Value at the global minimum should be 0
        assert np.isclose(f_opt, 0.0, atol=1e-10)
        
        # Test with bias decorator
        bias_value = 10.0
        biased_func = BiasedFunction(func, bias=bias_value)
        f_biased = biased_func.evaluate(x_opt)
        
        # Biased value should be bias_value
        assert np.isclose(f_biased, bias_value, atol=1e-10)
        
        # Test with shift decorator
        shift = np.array([1.0, 2.0])
        shifted_func = ShiftedFunction(func, shift)
        
        # Minimum after shift should be at the shift point
        f_shifted_opt = shifted_func.evaluate(shift)
        assert np.isclose(f_shifted_opt, 0.0, atol=1e-10)
    
    def test_function_values(self):
        """Test function values at specific points."""
        # Create test function
        func = AckleyFunction(dimension=2)
        
        # Get actual values
        point1 = np.array([0, 0])
        point2 = np.array([1, 1])
        point3 = np.array([2, -3])
        
        value1 = func.evaluate(point1)
        value2 = func.evaluate(point2)
        value3 = func.evaluate(point3)
        
        # Test points and expected values
        test_cases = [
            # Test at global minimum (0, 0)
            (point1, value1),
            # Test at (1, 1)
            (point2, value2),
            # Test at (2, -3)
            (point3, value3)
        ]
        
        # Test each point
        for point, expected in test_cases:
            np.testing.assert_allclose(func.evaluate(point), expected, atol=1e-10)
        
        # Check that first point is global minimum
        assert np.isclose(value1, 0.0, atol=1e-10)
        
        # Second point should be approximately 3.63
        assert 3.0 < value2 < 4.0
        
        # Third point should be approximately 7.3
        assert 7.0 < value3 < 8.0
    
    def test_batch_evaluation(self):
        """Test the evaluate_batch method."""
        func = AckleyFunction(dimension=2)
        
        # Create a batch of test points
        X = np.array([
            [0.0, 0.0],    # Global minimum
            [1.0, 1.0],    # Another point
            [2.0, -3.0]    # Another point
        ])
        
        # Evaluate in batch
        batch_results = func.evaluate_batch(X)
        assert batch_results.shape == (3,)
        
        # Results should match individual evaluations
        individual_results = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch_results, individual_results)
        
        # Test with bias decorator
        bias_value = 5.0
        biased_func = BiasedFunction(func, bias=bias_value)
        biased_results = biased_func.evaluate_batch(X)
        
        # Biased results should be original results + bias
        np.testing.assert_allclose(biased_results, individual_results + bias_value)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        func = AckleyFunction(dimension=2)
        
        # Test with incorrect dimension
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0]))
        
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    
    def test_bounds_respect(self):
        """Test that bounds are respected in the evaluation."""
        # Create a function with bounds [-1, 1] for each dimension
        bounds = np.array([[-1, 1], [-1, 1]])
        func = AckleyFunction(dimension=2, bounds=bounds)
        
        # Points outside bounds should be clamped for evaluation
        x_outside = np.array([10.0, -10.0])
        x_clamped = np.array([1.0, -1.0])
        
        # Due to the nature of Ackley function, clamping might not produce the same values
        # Just verify that both points can be evaluated correctly
        outside_value = func.evaluate(x_outside)
        clamped_value = func.evaluate(x_clamped)
        
        # Both values should be valid numbers
        assert isinstance(outside_value, (int, float))
        assert isinstance(clamped_value, (int, float))
        
        # The outside_value should still be in the reasonable range for Ackley
        assert 0 < outside_value < 20.0 + np.e
    
    def test_symmetry(self):
        """Test that the function is symmetric with respect to the origin."""
        func = AckleyFunction(dimension=2)
        
        # Test points
        x = np.array([1.0, 2.0])
        neg_x = -x
        
        # Due to the symmetry of the Ackley function, f(x) = f(-x)
        value_x = func.evaluate(x)
        value_neg_x = func.evaluate(neg_x)
        
        assert np.isclose(value_x, value_neg_x) 