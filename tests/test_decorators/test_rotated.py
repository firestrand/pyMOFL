"""
Tests for the RotatedFunction decorator.
"""

import pytest
import numpy as np
from pyMOFL.functions.unimodal import SphereFunction
from pyMOFL.decorators import RotatedFunction
from pyMOFL.utils.rotation import generate_rotation_matrix


class TestRotatedFunction:
    """Tests for the RotatedFunction decorator."""
    
    def test_initialization(self):
        """Test initialization with explicit and random rotation matrix."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Test with explicit rotation matrix
        rotation_matrix = np.array([[0, 1], [1, 0]])  # 90-degree rotation
        rotated_func = RotatedFunction(base_func, rotation_matrix)
        assert rotated_func.dimension == 2
        assert np.array_equal(rotated_func.rotation_matrix, rotation_matrix)
        assert np.array_equal(rotated_func.bounds, base_func.bounds)
        
        # Test with random rotation matrix (just check that it's created with correct shape)
        rotated_func = RotatedFunction(base_func)
        assert rotated_func.dimension == 2
        assert rotated_func.rotation_matrix.shape == (2, 2)
        
        # Check that the random rotation matrix is orthogonal
        R = rotated_func.rotation_matrix
        np.testing.assert_allclose(np.dot(R, R.T), np.eye(2), atol=1e-10)
    
    def test_rotation_matrix_validation(self):
        """Test that rotation matrix is validated correctly."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Test with incorrect rotation matrix shape
        invalid_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with pytest.raises(ValueError):
            RotatedFunction(base_func, invalid_matrix)
    
    def test_evaluate(self):
        """Test the evaluate method."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Create a rotated function with a 90-degree rotation matrix
        rotation_matrix = np.array([[0, -1], [1, 0]])  # 90-degree rotation
        rotated_func = RotatedFunction(base_func, rotation_matrix)
        
        # Test at global minimum (which should still be at the origin for Sphere)
        assert rotated_func.evaluate(np.array([0.0, 0.0])) == 0
        
        # Test with unit vector
        # f(1,0) = base_func(R.dot([1,0])) = base_func([0,1]) = 0^2 + 1^2 = 1
        assert rotated_func.evaluate(np.array([1.0, 0.0])) == 1
        
        # Test with arbitrary vector
        x = np.array([2.0, 3.0])
        # f(2,3) = base_func(R.dot([2,3])) = base_func([-3,2]) = (-3)^2 + 2^2 = 9 + 4 = 13
        assert rotated_func.evaluate(x) == 13
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Create a rotated function with a 90-degree rotation matrix
        rotation_matrix = np.array([[0, -1], [1, 0]])  # 90-degree rotation
        rotated_func = RotatedFunction(base_func, rotation_matrix)
        
        # Test with batch of vectors
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 3.0]])
        expected = np.array([0, 1, 13])
        assert np.array_equal(rotated_func.evaluate_batch(X), expected)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Create a rotated function
        rotation_matrix = np.array([[0, -1], [1, 0]])  # 90-degree rotation
        rotated_func = RotatedFunction(base_func, rotation_matrix)
        
        # Test with incorrect dimension
        with pytest.raises(ValueError):
            rotated_func.evaluate(np.array([1.0, 2.0, 3.0]))
        
        with pytest.raises(ValueError):
            rotated_func.evaluate_batch(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    
    def test_nested_rotation(self):
        """Test that rotation can be nested."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Create a rotated function with a 90-degree rotation matrix
        rotation_matrix1 = np.array([[0, -1], [1, 0]])  # 90-degree rotation
        rotated_func1 = RotatedFunction(base_func, rotation_matrix1)
        
        # Create a nested rotated function with another 90-degree rotation matrix
        rotation_matrix2 = np.array([[0, -1], [1, 0]])  # 90-degree rotation
        rotated_func2 = RotatedFunction(rotated_func1, rotation_matrix2)
        
        # Test with unit vector
        # The combined rotation should be 180 degrees
        # f(1,0) = rotated_func1(R2.dot([1,0])) = rotated_func1([0,1])
        #        = base_func(R1.dot([0,1])) = base_func([-1,0])
        #        = (-1)^2 + 0^2 = 1
        assert rotated_func2.evaluate(np.array([1.0, 0.0])) == 1
        
        # Test with arbitrary vector
        x = np.array([2.0, 3.0])
        # f(2,3) = rotated_func1(R2.dot([2,3])) = rotated_func1([-3,2])
        #        = base_func(R1.dot([-3,2])) = base_func([-2,-3])
        #        = (-2)^2 + (-3)^2 = 4 + 9 = 13
        assert rotated_func2.evaluate(x) == 13 