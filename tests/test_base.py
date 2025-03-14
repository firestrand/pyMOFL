"""
Tests for the base OptimizationFunction class.
"""

import pytest
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from pyMOFL.base import OptimizationFunction


# Create a concrete implementation of OptimizationFunction for testing
class TestFunction(OptimizationFunction):
    """
    A simple test function for testing the base class.
    
    f(x) = sum(x^2)
    """
    
    def evaluate(self, x):
        """Evaluate the function at point x."""
        return float(np.sum(x**2))


class TestOptimizationFunction:
    """Tests for the OptimizationFunction base class."""
    
    def test_initialization(self):
        """Test initialization with default and custom bounds."""
        # Test with default bounds
        func = TestFunction(dimension=2)
        assert func.dimension == 2
        assert func.bounds.shape == (2, 2)
        assert np.array_equal(func.bounds, np.array([[-100, 100], [-100, 100]]))
        
        # Test with custom bounds
        custom_bounds = np.array([[-10, 10], [-5, 5]])
        func = TestFunction(dimension=2, bounds=custom_bounds)
        assert func.dimension == 2
        assert func.bounds.shape == (2, 2)
        assert np.array_equal(func.bounds, custom_bounds)
    
    def test_bounds_validation(self):
        """Test that bounds are validated correctly."""
        # Test with invalid bounds shape
        invalid_bounds = np.array([[-10, 10]])
        with pytest.raises(ValueError):
            TestFunction(dimension=2, bounds=invalid_bounds)
    
    def test_evaluate(self):
        """Test the evaluate method."""
        func = TestFunction(dimension=2)
        
        # Test with zero vector
        assert func.evaluate(np.array([0, 0])) == 0
        
        # Test with unit vector
        assert func.evaluate(np.array([1, 1])) == 2
        
        # Test with arbitrary vector
        x = np.array([2, 3])
        assert func.evaluate(x) == 13  # 2^2 + 3^2 = 4 + 9 = 13
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        func = TestFunction(dimension=2)
        
        # Test with batch of vectors
        X = np.array([[0, 0], [1, 1], [2, 3]])
        expected = np.array([0, 2, 13])
        assert np.array_equal(func.evaluate_batch(X), expected)
        
        # Test with single vector reshaped as batch
        X = np.array([[2, 3]])
        expected = np.array([13])
        assert np.array_equal(func.evaluate_batch(X), expected) 