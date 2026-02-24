"""
Non-continuous transformation - introduces discontinuities.

A vector-to-vector transformation used in CEC benchmarks.
Mathematical form: non_continuous(x) applies rounding to elements outside [-0.5, 0.5]
"""

import numpy as np

from .base import VectorTransform


class NonContinuousTransform(VectorTransform):
    """
    Non-continuous transformation function.

    Applies a discontinuous transformation to input vectors,
    rounding values outside the range [-0.5, 0.5].
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply non-continuous transformation.

        Uses round-half-away-from-zero to match C's round() function:
        copysign(floor(|2x| + 0.5), x) / 2.

        Args:
            x: Input vector

        Returns:
            Transformed vector with discontinuities
        """
        x = np.asarray(x, dtype=np.float64)
        y = x.copy()
        mask = np.abs(x) >= 0.5
        y[mask] = np.copysign(np.floor(np.abs(2 * x[mask]) + 0.5), x[mask]) / 2
        return y

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Apply non-continuous transformation to batch.

        Args:
            X: Batch of input vectors (N x dimension)

        Returns:
            Batch of transformed vectors
        """
        X = np.asarray(X, dtype=np.float64)
        Y = X.copy()
        mask = np.abs(X) >= 0.5
        Y[mask] = np.copysign(np.floor(np.abs(2 * X[mask]) + 0.5), X[mask]) / 2
        return Y
