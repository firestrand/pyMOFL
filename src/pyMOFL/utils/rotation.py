"""
Rotation utility functions.

This module provides utility functions for generating rotation matrices.
"""

import numpy as np


def generate_rotation_matrix(dimension: int) -> np.ndarray:
    """
    Generate a random orthogonal rotation matrix.
    
    This function uses the QR decomposition of a random matrix to generate
    a random orthogonal matrix that can be used for rotation transformations.
    The function ensures that the determinant is 1 (proper rotation).
    
    Args:
        dimension (int): The dimensionality of the rotation matrix.
        
    Returns:
        np.ndarray: A random orthogonal matrix of shape (dimension, dimension) with determinant 1.
    """
    # Generate a random matrix
    random_matrix = np.random.randn(dimension, dimension)
    
    # Use QR decomposition to get an orthogonal matrix
    q, r = np.linalg.qr(random_matrix)
    
    # Ensure the diagonal elements of R are positive to get a unique Q
    d = np.diag(np.sign(np.diag(r)))
    q = np.dot(q, d)
    
    # Ensure the determinant is 1 (proper rotation)
    det = np.linalg.det(q)
    if det < 0:
        # If determinant is -1, flip the sign of one column to make it 1
        q[:, 0] = -q[:, 0]
    
    return q 