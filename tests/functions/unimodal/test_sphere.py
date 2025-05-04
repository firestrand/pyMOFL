"""
Tests for the Sphere function.
"""

import pytest
import numpy as np
from pyMOFL.functions.unimodal import SphereFunction


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
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        func = SphereFunction(dimension=2)
        
        # Test with batch of vectors
        X = np.array([[0, 0], [1, 1], [2, 3]])
        expected = np.array([0, 2, 13])
        assert np.array_equal(func.evaluate_batch(X), expected)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        func = SphereFunction(dimension=2)
        
        # Test with incorrect dimension
        with pytest.raises(ValueError):
            func.evaluate(np.array([1, 2, 3]))
        
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1, 2, 3], [4, 5, 6]])) 