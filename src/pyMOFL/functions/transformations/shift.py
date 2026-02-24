"""
Shift transformation - translates input space.

A vector-to-vector transformation function.
Mathematical form: shift(x) = x - shift_vector
"""

import numpy as np

from .base import VectorTransform


class ShiftTransform(VectorTransform):
    """
    Shift transformation function.

    Takes a vector input and returns a shifted vector.
    """

    def __init__(self, shift: np.ndarray | float | int):
        """
        Initialize shift transformation.

        Args:
            shift: The shift vector or scalar (scalar creates uniform shift)
        """
        if np.isscalar(shift) or isinstance(shift, (int, float)):
            # Store scalar shift - will be broadcast during application
            self.shift = shift
            self.dimension = None  # Dimension determined at call time
        else:
            self.shift = np.asarray(shift, dtype=np.float64)
            self.dimension = len(self.shift)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply shift transformation.

        Args:
            x: Input vector

        Returns:
            Shifted vector: x - shift
        """
        x = np.asarray(x, dtype=np.float64)
        if self.dimension is not None and len(x) != self.dimension:
            raise ValueError(
                f"Input dimension {len(x)} doesn't match shift dimension {self.dimension}"
            )
        # NumPy broadcasting handles scalar shift automatically
        return x - self.shift

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Apply shift transformation to batch.

        Args:
            X: Batch of input vectors (N x dimension)

        Returns:
            Batch of shifted vectors: X - shift
        """
        X = np.asarray(X, dtype=np.float64)
        if self.dimension is not None and X.shape[1] != self.dimension:
            raise ValueError(
                f"Input dimension {X.shape[1]} doesn't match shift dimension {self.dimension}"
            )
        # NumPy broadcasting handles scalar shift automatically
        return X - self.shift
