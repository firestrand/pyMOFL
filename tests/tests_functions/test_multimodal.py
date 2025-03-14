"""
Tests for multimodal benchmark functions.
"""

import pytest
import numpy as np
from pyMOFL.functions.multimodal import RastriginFunction


class TestRastriginFunction:
    """Tests for the Rastrigin function."""
    
    def test_initialization(self):
        """Test initialization with default and custom bounds."""
        # Test with default bounds
        func = RastriginFunction(dimension=2)
        assert func.dimension == 2
        assert func.bounds.shape == (2, 2)
        assert np.array_equal(func.bounds, np.array([[-5.12, 5.12], [-5.12, 5.12]]))
        
        # Test with custom bounds
        custom_bounds = np.array([[-10, 10], [-5, 5]])
        func = RastriginFunction(dimension=2, bounds=custom_bounds)
        assert func.dimension == 2
        assert func.bounds.shape == (2, 2)
        assert np.array_equal(func.bounds, custom_bounds)
    
    def test_evaluate(self):
        """Test the evaluate method."""
        func = RastriginFunction(dimension=2)
        
        # Test at global minimum
        assert func.evaluate(np.array([0, 0])) == 0
        
        # Test with unit vector
        # f(1,1) = 10*2 + (1^2 - 10*cos(2*pi*1) + 1^2 - 10*cos(2*pi*1))
        # = 20 + (1 - 10*1 + 1 - 10*1) = 20 + (2 - 20) = 20 - 18 = 2
        expected = 20 + 2 - 10 * np.cos(2 * np.pi * 1) - 10 * np.cos(2 * np.pi * 1)
        np.testing.assert_allclose(func.evaluate(np.array([1, 1])), expected)
        
        # Test with arbitrary vector
        x = np.array([2, 3])
        expected = 20 + 2**2 - 10 * np.cos(2 * np.pi * 2) + 3**2 - 10 * np.cos(2 * np.pi * 3)
        np.testing.assert_allclose(func.evaluate(x), expected)
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        func = RastriginFunction(dimension=2)
        
        # Test with batch of vectors
        X = np.array([[0, 0], [1, 1], [2, 3]])
        expected = np.array([
            0,
            20 + 2 - 10 * np.cos(2 * np.pi * 1) - 10 * np.cos(2 * np.pi * 1),
            20 + 2**2 - 10 * np.cos(2 * np.pi * 2) + 3**2 - 10 * np.cos(2 * np.pi * 3)
        ])
        np.testing.assert_allclose(func.evaluate_batch(X), expected)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        func = RastriginFunction(dimension=2)
        
        # Test with incorrect dimension
        with pytest.raises(ValueError):
            func.evaluate(np.array([1, 2, 3]))
        
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1, 2, 3], [4, 5, 6]]))
    
    def test_higher_dimensions(self):
        """Test the function in higher dimensions."""
        func = RastriginFunction(dimension=3)
        
        # Test at global minimum
        assert func.evaluate(np.array([0, 0, 0])) == 0
        
        # Test with ones vector
        x = np.array([1, 1, 1])
        expected = 30 + 3 - 10 * np.cos(2 * np.pi * 1) - 10 * np.cos(2 * np.pi * 1) - 10 * np.cos(2 * np.pi * 1)
        np.testing.assert_allclose(func.evaluate(x), expected) 