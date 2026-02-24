"""
Rotate transformation - applies linear transformation to input.

A vector-to-vector transformation function.
Mathematical form: rotate(x) = matrix.T @ x
"""

import numpy as np

from .base import VectorTransform
from .linear import linear_transform, linear_transform_batch


class RotateTransform(VectorTransform):
    """
    Rotate/linear transformation function.

    Takes a vector input and returns a transformed vector.
    Note: Despite the name, CEC "rotation" matrices are general linear transformations.
    """

    def __init__(self, matrix: np.ndarray):
        """
        Initialize rotation transformation.

        Args:
            matrix: The transformation matrix
        """
        self.matrix = np.asarray(matrix, dtype=np.float64)

        # Validate square matrix
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {self.matrix.shape}")

        self.dimension = self.matrix.shape[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply rotation transformation.

        Args:
            x: Input vector

        Returns:
            Rotated vector: matrix.T @ x
        """
        x = np.asarray(x, dtype=np.float64)
        if len(x) != self.dimension:
            raise ValueError(
                f"Input dimension {len(x)} doesn't match matrix dimension {self.dimension}"
            )

        return linear_transform(x, self.matrix)

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Apply rotation transformation to batch.

        Args:
            X: Batch of input vectors (N x dimension)

        Returns:
            Batch of rotated vectors
        """
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.dimension:
            raise ValueError(
                f"Input dimension {X.shape[1]} doesn't match matrix dimension {self.dimension}"
            )

        return linear_transform_batch(X, self.matrix)
