"""
Tests for the Sphere function.
"""

import pytest
import numpy as np
from pyMOFL.functions.unimodal import SphereFunction
from pyMOFL.decorators import BiasedFunction


class TestSphereFunction:
    """Tests for the Sphere function."""
    
    def test_initialization(self):
        """Test initialization with default and custom bounds."""
        # Test with default bounds
        func = SphereFunction(dimension=2)
        assert func.dimension == 2
        assert func.bounds.shape == (2, 2)
        assert np.array_equal(func.bounds, np.array([[-100, 100], [-100, 100]]))
        
        # Test with custom bounds
        custom_bounds = np.array([[-10, 10], [-5, 5]])
        func = SphereFunction(dimension=2, bounds=custom_bounds)
        assert func.dimension == 2
        assert func.bounds.shape == (2, 2)
        assert np.array_equal(func.bounds, custom_bounds)
    
    def test_evaluate(self):
        """Test the evaluate method."""
        func = SphereFunction(dimension=2)
        
        # Test at global minimum
        assert func.evaluate(np.array([0, 0])) == 0
        
        # Test with unit vector
        assert func.evaluate(np.array([1, 1])) == 2
        
        # Test with arbitrary vector
        x = np.array([2, 3])
        assert func.evaluate(x) == 13  # 2^2 + 3^2 = 4 + 9 = 13
        
        # Test with bias using decorator
        bias_value = 10.0
        biased_func = BiasedFunction(func, bias=bias_value)
        assert biased_func.evaluate(np.array([0, 0])) == 10.0
        assert biased_func.evaluate(np.array([1, 1])) == 12.0  # 2 + 10 = 12
        assert biased_func.evaluate(np.array([2, 3])) == 23.0  # 13 + 10 = 23
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        func = SphereFunction(dimension=2)
        
        # Test with batch of vectors
        X = np.array([[0, 0], [1, 1], [2, 3]])
        expected = np.array([0, 2, 13])
        assert np.array_equal(func.evaluate_batch(X), expected)
        
        # Test with bias using decorator
        bias_value = 5.0
        biased_func = BiasedFunction(func, bias=bias_value)
        expected_biased = np.array([5, 7, 18])  # expected + 5
        assert np.array_equal(biased_func.evaluate_batch(X), expected_biased)
    
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
            bias_value = 3.0
            biased_func = BiasedFunction(func, bias=bias_value)
            assert np.isclose(biased_func.evaluate(point), value + bias_value) 