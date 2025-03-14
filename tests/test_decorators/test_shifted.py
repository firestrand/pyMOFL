"""
Tests for the ShiftedFunction decorator.
"""

import pytest
import numpy as np
from pyMOFL.functions.unimodal import SphereFunction
from pyMOFL.decorators import ShiftedFunction


class TestShiftedFunction:
    """Tests for the ShiftedFunction decorator."""
    
    def test_initialization(self):
        """Test initialization with explicit and random shift."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Test with explicit shift
        shift = np.array([1.0, 2.0])
        shifted_func = ShiftedFunction(base_func, shift)
        assert shifted_func.dimension == 2
        assert np.array_equal(shifted_func.shift, shift)
        assert np.array_equal(shifted_func.bounds, base_func.bounds)
        
        # Test with random shift (just check that it's created with correct dimension)
        shifted_func = ShiftedFunction(base_func)
        assert shifted_func.dimension == 2
        assert shifted_func.shift.shape == (2,)
    
    def test_shift_validation(self):
        """Test that shift vector is validated correctly."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Test with incorrect shift dimension
        invalid_shift = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            ShiftedFunction(base_func, invalid_shift)
    
    def test_evaluate(self):
        """Test the evaluate method."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Create a shifted function
        shift = np.array([1.0, 2.0])
        shifted_func = ShiftedFunction(base_func, shift)
        
        # Test at shifted global minimum
        assert shifted_func.evaluate(np.array([1.0, 2.0])) == 0
        
        # Test at origin
        # f(0,0) = base_func(0-1, 0-2) = base_func(-1, -2) = (-1)^2 + (-2)^2 = 1 + 4 = 5
        assert shifted_func.evaluate(np.array([0.0, 0.0])) == 5
        
        # Test with arbitrary vector
        x = np.array([2.0, 3.0])
        # f(2,3) = base_func(2-1, 3-2) = base_func(1, 1) = 1^2 + 1^2 = 2
        assert shifted_func.evaluate(x) == 2
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Create a shifted function
        shift = np.array([1.0, 2.0])
        shifted_func = ShiftedFunction(base_func, shift)
        
        # Test with batch of vectors
        X = np.array([[1.0, 2.0], [0.0, 0.0], [2.0, 3.0]])
        expected = np.array([0, 5, 2])
        assert np.array_equal(shifted_func.evaluate_batch(X), expected)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Create a shifted function
        shift = np.array([1.0, 2.0])
        shifted_func = ShiftedFunction(base_func, shift)
        
        # Test with incorrect dimension
        with pytest.raises(ValueError):
            shifted_func.evaluate(np.array([1.0, 2.0, 3.0]))
        
        with pytest.raises(ValueError):
            shifted_func.evaluate_batch(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    
    def test_nested_shifting(self):
        """Test that shifting can be nested."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Create a shifted function
        shift1 = np.array([1.0, 2.0])
        shifted_func1 = ShiftedFunction(base_func, shift1)
        
        # Create a nested shifted function
        shift2 = np.array([2.0, 3.0])
        shifted_func2 = ShiftedFunction(shifted_func1, shift2)
        
        # Test at nested shifted global minimum
        # The global minimum should be at shift1 + shift2 = [3.0, 5.0]
        assert shifted_func2.evaluate(np.array([3.0, 5.0])) == 0
        
        # Test at origin
        # f(0,0) = shifted_func1(0-2, 0-3) = shifted_func1(-2, -3)
        #        = base_func(-2-1, -3-2) = base_func(-3, -5)
        #        = (-3)^2 + (-5)^2 = 9 + 25 = 34
        assert shifted_func2.evaluate(np.array([0.0, 0.0])) == 34 