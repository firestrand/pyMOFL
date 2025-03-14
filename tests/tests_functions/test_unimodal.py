"""
Tests for unimodal benchmark functions.
"""

import pytest
import numpy as np
from pyMOFL.functions.unimodal import SphereFunction, RosenbrockFunction


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


class TestRosenbrockFunction:
    """Tests for the Rosenbrock function."""
    
    def test_initialization(self):
        """Test initialization with default and custom bounds."""
        # Test with default bounds
        func = RosenbrockFunction(dimension=2)
        assert func.dimension == 2
        assert func.bounds.shape == (2, 2)
        assert np.array_equal(func.bounds, np.array([[-30, 30], [-30, 30]]))
        
        # Test with custom bounds
        custom_bounds = np.array([[-10, 10], [-5, 5]])
        func = RosenbrockFunction(dimension=2, bounds=custom_bounds)
        assert func.dimension == 2
        assert func.bounds.shape == (2, 2)
        assert np.array_equal(func.bounds, custom_bounds)
    
    def test_evaluate(self):
        """Test the evaluate method."""
        func = RosenbrockFunction(dimension=2)
        
        # Test at global minimum
        assert func.evaluate(np.array([1, 1])) == 0
        
        # Test at origin
        # f(0,0) = (0-1)^2 + 100*(0-0^2)^2 = 1
        assert func.evaluate(np.array([0, 0])) == 1
        
        # Test with arbitrary vector
        # f(2,3) = (2-1)^2 + 100*(3-2^2)^2 = 1 + 100*1 = 101
        x = np.array([2, 3])
        assert func.evaluate(x) == 101
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        func = RosenbrockFunction(dimension=2)
        
        # Test with batch of vectors
        X = np.array([[1, 1], [0, 0], [2, 3]])
        expected = np.array([0, 1, 101])
        np.testing.assert_allclose(func.evaluate_batch(X), expected)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        func = RosenbrockFunction(dimension=2)
        
        # Test with incorrect dimension
        with pytest.raises(ValueError):
            func.evaluate(np.array([1, 2, 3]))
        
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1, 2, 3], [4, 5, 6]]))
    
    def test_higher_dimensions(self):
        """Test the function in higher dimensions."""
        func = RosenbrockFunction(dimension=3)
        
        # Test at global minimum
        assert func.evaluate(np.array([1, 1, 1])) == 0
        
        # Test with arbitrary vector
        # For 3D: f(x,y,z) = 100*(y-x^2)^2 + (x-1)^2 + 100*(z-y^2)^2 + (y-1)^2
        # f(2,2,2) = 100*(2-4)^2 + (2-1)^2 + 100*(2-4)^2 + (2-1)^2
        #          = 100*4 + 1 + 100*4 + 1 = 802
        x = np.array([2, 2, 2])
        assert func.evaluate(x) == 802 