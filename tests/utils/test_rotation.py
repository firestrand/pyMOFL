"""
Tests for the rotation utility functions.
"""

import pytest
import numpy as np
from pyMOFL.utils.rotation import generate_rotation_matrix


class TestRotationUtils:
    """Tests for the rotation utility functions."""
    
    def test_generate_rotation_matrix(self):
        """Test the generate_rotation_matrix function."""
        # Test with dimension 2
        R = generate_rotation_matrix(2)
        assert R.shape == (2, 2)
        
        # Check that the matrix is orthogonal (R * R^T = I)
        np.testing.assert_allclose(np.dot(R, R.T), np.eye(2), atol=1e-10)
        
        # Check that the determinant is 1 (proper rotation)
        assert np.abs(np.linalg.det(R) - 1.0) < 1e-10
        
        # Test with dimension 3
        R = generate_rotation_matrix(3)
        assert R.shape == (3, 3)
        
        # Check that the matrix is orthogonal (R * R^T = I)
        np.testing.assert_allclose(np.dot(R, R.T), np.eye(3), atol=1e-10)
        
        # Check that the determinant is 1 (proper rotation)
        assert np.abs(np.linalg.det(R) - 1.0) < 1e-10
        
        # Test with higher dimension
        dim = 10
        R = generate_rotation_matrix(dim)
        assert R.shape == (dim, dim)
        
        # Check that the matrix is orthogonal (R * R^T = I)
        np.testing.assert_allclose(np.dot(R, R.T), np.eye(dim), atol=1e-10)
        
        # Check that the determinant is 1 (proper rotation)
        assert np.abs(np.linalg.det(R) - 1.0) < 1e-10
    
    def test_rotation_properties(self):
        """Test that the generated rotation matrices have the expected properties."""
        # Generate multiple rotation matrices and check their properties
        for _ in range(5):
            R = generate_rotation_matrix(3)
            
            # Check that the matrix is orthogonal (R * R^T = I)
            np.testing.assert_allclose(np.dot(R, R.T), np.eye(3), atol=1e-10)
            
            # Check that the determinant is 1 (proper rotation)
            assert np.abs(np.linalg.det(R) - 1.0) < 1e-10
            
            # Check that the matrix preserves vector lengths
            v = np.random.rand(3)
            Rv = np.dot(R, v)
            assert np.abs(np.linalg.norm(v) - np.linalg.norm(Rv)) < 1e-10 