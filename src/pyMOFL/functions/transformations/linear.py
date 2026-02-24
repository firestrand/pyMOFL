"""
Linear transformation utilities used by rotation transforms and tests.

Provides exact and batch implementations with a simple class wrapper.
"""

from __future__ import annotations

import numpy as np


def linear_transform(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    # Equivalent to M.T @ v in rotate transform semantics
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        return M.T @ v


def linear_transform_batch(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    # Apply to each row: X @ M
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        return X @ M


class LinearTransform:
    def __init__(self, M: np.ndarray) -> None:
        self.M = np.asarray(M, dtype=np.float64)

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return linear_transform(v, self.M)

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        return linear_transform_batch(X, self.M)


# Alias for tests that may reference an optimized version
def linear_transform_optimized(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    return linear_transform(v, M)
