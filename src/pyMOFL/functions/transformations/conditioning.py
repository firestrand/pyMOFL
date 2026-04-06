"""
Conditioning (lambda) transform - diagonal scaling with power conditioning.

Mathematical form: z[i] = x[i] * alpha^(i / (2*(D-1)))

Used by CEC 2013 and similar benchmarks for ill-conditioning.
"""

from __future__ import annotations

import numpy as np

from .base import VectorTransform


class ConditioningTransform(VectorTransform):
    """Diagonal power conditioning transform.

    Applies element-wise scaling: z[i] = x[i] * alpha^(i / (2*(D-1)))
    where alpha is the conditioning number and D is the dimension.
    """

    def __init__(self, alpha: float, dimension: int) -> None:
        if dimension < 2:
            self._factors = np.ones(dimension, dtype=np.float64)
        else:
            exponents = np.arange(dimension, dtype=np.float64) / (2.0 * (dimension - 1))
            self._factors = np.power(alpha, exponents)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return x * self._factors

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X * self._factors
