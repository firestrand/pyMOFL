"""
Tests for the Tripod function.
"""

import pytest
import numpy as np
from pyMOFL.functions.multimodal import TripodFunction


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
    
    def test_evaluate_global_minimum(self):
        """Test the evaluate method at global minimum."""
        func = TripodFunction()
        
        # Test at global minimum (0, -50)
        np.testing.assert_allclose(func.evaluate(np.array([0, -50])), 0, atol=1e-10)
    
    def test_evaluate_quadrants(self):
        """Test the function value in different quadrants of the plane."""
        func = TripodFunction()
        
        # Quadrant I: x1 >= 0, x2 >= 0
        # p(x1) = 1, p(x2) = 1
        x = np.array([5, 10])
        # term1 = 1 * (1 + 1) = 2
        # term2 = |5 + 50 * 1 * (1 - 2 * 1)| = |5 - 50| = 45
        # term3 = |10 + 50 * (1 - 2 * 1)| = |10 - 50| = 40
        expected = 2 + 45 + 40
        np.testing.assert_allclose(func.evaluate(x), expected)
        
        # Quadrant II: x1 < 0, x2 >= 0
        # p(x1) = 0, p(x2) = 1
        x = np.array([-5, 10])
        # term1 = 1 * (1 + 0) = 1
        # term2 = |-5 + 50 * 1 * (1 - 2 * 0)| = |-5 + 50| = 45
        # term3 = |10 + 50 * (1 - 2 * 1)| = |10 - 50| = 40
        expected = 1 + 45 + 40
        np.testing.assert_allclose(func.evaluate(x), expected)
        
        # Quadrant III: x1 < 0, x2 < 0
        # p(x1) = 0, p(x2) = 0
        x = np.array([-5, -10])
        # term1 = 0 * (1 + 0) = 0
        # term2 = |-5 + 50 * 0 * (1 - 2 * 0)| = |-5| = 5
        # term3 = |-10 + 50 * (1 - 2 * 0)| = |-10 + 50| = 40
        expected = 0 + 5 + 40
        np.testing.assert_allclose(func.evaluate(x), expected)
        
        # Quadrant IV: x1 >= 0, x2 < 0
        # p(x1) = 1, p(x2) = 0
        x = np.array([5, -10])
        # term1 = 0 * (1 + 1) = 0
        # term2 = |5 + 50 * 0 * (1 - 2 * 1)| = |5| = 5
        # term3 = |-10 + 50 * (1 - 2 * 0)| = |-10 + 50| = 40
        expected = 0 + 5 + 40
        np.testing.assert_allclose(func.evaluate(x), expected)
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        func = TripodFunction()
        
        # Test with batch of vectors
        X = np.array([
            [0, -50],  # global minimum
            [5, 10],   # quadrant I
            [-5, 10],  # quadrant II
            [-5, -10], # quadrant III
            [5, -10]   # quadrant IV
        ])
        
        expected = np.array([
            0,
            2 + 45 + 40,  # quadrant I
            1 + 45 + 40,  # quadrant II
            0 + 5 + 40,   # quadrant III
            0 + 5 + 40    # quadrant IV
        ])
        
        np.testing.assert_allclose(func.evaluate_batch(X), expected)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        func = TripodFunction()
        
        # Test with incorrect dimension
        with pytest.raises(ValueError):
            func.evaluate(np.array([1, 2, 3]))
        
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1, 2, 3], [4, 5, 6]]))
    
    def test_non_negativity(self):
        """Test that function values are non-negative for points in the domain."""
        func = TripodFunction()
        
        # Generate random points in the domain
        rng = np.random.default_rng(42)
        points = rng.uniform(-100, 100, size=(100, 2))
        
        # Evaluate function at these points
        values = func.evaluate_batch(points)
        
        # Check that all values are non-negative
        assert (values >= 0).all()
        
    def test_with_bias(self):
        """Test function with a bias value."""
        bias = 10.0
        func = TripodFunction(bias=bias)
        
        # Test at global minimum
        np.testing.assert_allclose(func.evaluate(np.array([0, -50])), bias, atol=1e-10)
        
        # Test at another point
        x = np.array([5, 10])
        expected = 2 + 45 + 40 + bias
        np.testing.assert_allclose(func.evaluate(x), expected) 