"""
Offset transformation - translates input space by addition.

A vector-to-vector transformation function that adds a constant offset to
an input vector. This matches CEC configurations that apply a "+c" step
(after shift) before evaluating the base function (e.g. F06/F13 add +1).
"""

from __future__ import annotations

import numpy as np

from .base import VectorTransform


class OffsetTransform(VectorTransform):
    """Add a constant offset to the input vector."""

    def __init__(self, offset: np.ndarray | float | int):
        if isinstance(offset, (int, float)):
            self.offset = float(offset)
            self.dimension = None
        else:
            self.offset = np.asarray(offset, dtype=np.float64)
            self.dimension = self.offset.shape[0]

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.dimension is not None and x.shape[0] != self.dimension:
            raise ValueError(
                f"Input dimension {x.shape[0]} does not match offset dimension {self.dimension}"
            )
        return x + self.offset

    def transform_batch(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.dimension is not None and X.shape[1] != self.dimension:
            raise ValueError(
                f"Input dimension {X.shape[1]} does not match offset dimension {self.dimension}"
            )
        return X + self.offset
