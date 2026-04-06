"""
Step-half rounding transform for CEC 2013 Step Rastrigin (F13).

Mathematical form:
    if |z[i]| > 0.5: z[i] = floor(2*z[i] + 0.5) / 2
    else: z[i] unchanged
"""

from __future__ import annotations

import numpy as np

from .base import VectorTransform


class StepHalfTransform(VectorTransform):
    """Round to nearest 0.5 for elements with |z| > 0.5."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        result = x.copy()
        mask = np.abs(x) > 0.5
        result[mask] = np.floor(2.0 * x[mask] + 0.5) / 2.0
        return result

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        result = X.copy()
        mask = np.abs(X) > 0.5
        result[mask] = np.floor(2.0 * X[mask] + 0.5) / 2.0
        return result
