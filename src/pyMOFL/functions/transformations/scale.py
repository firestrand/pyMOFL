"""
Scale transformation - scales input space.

A vector-to-vector transformation function.
Mathematical form: scale(x) = x / scale_factor
Supports both scalar factors (uniform scaling) and vector factors (diagonal scaling).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import VectorTransform


class ScaleTransform(VectorTransform):
    """
    Scale transformation function.

    Takes a vector input and returns a scaled vector.
    Supports uniform (scalar) and diagonal (vector) scaling.
    """

    def __init__(self, factor: float | NDArray):
        """
        Initialize scale transformation.

        Args:
            factor: The scale factor (division factor for CEC/BBOB convention).
                   Can be a float (uniform) or an array (diagonal).
        """
        if isinstance(factor, (list, np.ndarray)):
            self.factor = np.asarray(factor, dtype=np.float64)
            if np.any(self.factor == 0):
                raise ValueError("Scale factor elements cannot be zero")
        else:
            self.factor = float(factor)
            if self.factor == 0:
                raise ValueError("Scale factor cannot be zero")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply scale transformation.

        Args:
            x: Input vector

        Returns:
            Scaled vector: x / factor (CEC convention)
        """
        x = np.asarray(x, dtype=np.float64)
        return x / self.factor

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Apply scale transformation to batch.

        Args:
            X: Batch of input vectors (N x dimension)

        Returns:
            Batch of scaled vectors: X / factor
        """
        X = np.asarray(X, dtype=np.float64)
        return X / self.factor
