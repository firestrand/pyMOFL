"""
Fused buffer-alias asymmetric transform (CEC 2013 C code compatibility).

Replicates the C code's ``asyfunc`` buffer aliasing behavior where the output
buffer is not initialised to the input.  When chaining ``inner → asy``, any
element where ``inner(x)[i] <= 0`` retains the **pre-inner** value ``x[i]``
rather than ``inner(x)[i]``.  This is a bug in the reference C implementation
but must be reproduced for golden-data validation.

Usage in the transform pipeline
--------------------------------
This single transform **replaces both** the inner transform (rotation or
oscillation) **and** the subsequent ``AsymmetricTransform``.
"""

from __future__ import annotations

import numpy as np

from .base import VectorTransform


class FusedBufferAliasAsymmetricTransform(VectorTransform):
    """Fused inner + asymmetric with CEC 2013 buffer aliasing.

    Parameters
    ----------
    inner : VectorTransform
        The transform applied before asymmetric (e.g. rotation or oscillation).
    beta : float
        Asymmetric beta parameter.
    dimension : int
        Expected input dimension.
    """

    def __init__(self, inner: VectorTransform, beta: float, dimension: int) -> None:
        self.inner = inner
        self.beta = float(beta)
        self.dimension = dimension
        if dimension <= 1:
            self._ratios = np.zeros(max(dimension, 1))
        else:
            self._ratios = np.arange(dimension, dtype=np.float64) / (dimension - 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        inner_result = self.inner(x)
        return self._apply_fused(x, inner_result)

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        inner_result = self.inner.transform_batch(X)
        return self._apply_fused_batch(X, inner_result)

    def _apply_fused(self, fallback: np.ndarray, transformed: np.ndarray) -> np.ndarray:
        positive = transformed > 0
        result = fallback.copy()
        if np.any(positive):
            exponent = 1.0 + self.beta * self._ratios * np.sqrt(
                np.where(positive, transformed, 0.0)
            )
            result[positive] = np.power(transformed[positive], exponent[positive])
        return result

    def _apply_fused_batch(self, fallback: np.ndarray, transformed: np.ndarray) -> np.ndarray:
        positive = transformed > 0
        result = fallback.copy()
        if np.any(positive):
            ratios = self._ratios[np.newaxis, :]
            exponent = 1.0 + self.beta * ratios * np.sqrt(np.where(positive, transformed, 0.0))
            result[positive] = np.power(transformed[positive], exponent[positive])
        return result
