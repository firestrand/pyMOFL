"""
Exact transformation implementations for testing and validation.

These implementations use explicit loops to exactly match C reference
implementations. They are slower but useful for validating optimized versions.

NOT FOR PRODUCTION USE - Use pyMOFL.transformations.linear instead.
"""

import numpy as np


def linear_transform_exact(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply linear transformation using explicit loop pattern.

    This implementation uses explicit nested loops for maximum numerical
    precision and exact compatibility with C reference implementations.

    Computes: result[j] = sum(matrix[i][j] * vector[i]) for all j

    This is ONLY for testing/validation. Production code should use
    the optimized version in pyMOFL.transformations.linear.

    Args:
        vector: Input vector to transform
        matrix: Transformation matrix (square)

    Returns:
        Transformed vector

    Raises:
        AssertionError: If matrix dimensions don't match vector
    """
    # Ensure inputs are numpy arrays with consistent precision
    vector = np.asarray(vector, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)

    n = len(vector)
    assert matrix.shape == (n, n), f"Matrix shape {matrix.shape} doesn't match vector dimension {n}"

    # Initialize result array
    result = np.zeros(n, dtype=np.float64)

    # Explicit nested loops matching C pattern exactly
    for j in range(n):
        result[j] = 0.0
        for i in range(n):
            result[j] += matrix[i][j] * vector[i]

    return result


def linear_transform_exact_batch(X: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply exact linear transformation to batch of vectors.

    For testing only - applies exact transformation to each vector individually.

    Args:
        X: Batch of input vectors (shape: batch_size × dimension)
        matrix: Transformation matrix (square)

    Returns:
        Batch of transformed vectors
    """
    X = np.asarray(X, dtype=np.float64)
    result = np.empty_like(X)

    for i in range(X.shape[0]):
        result[i] = linear_transform_exact(X[i], matrix)

    return result
