"""
Indexed scale transform: selects a scale factor by component index and
applies it to the input: x -> x / factor.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import VectorTransform


class IndexedScaleTransform(VectorTransform):
    def __init__(
        self,
        factors: float | Sequence[float],
        component_index: int | None = None,
        default_factor: float = 1.0,
    ) -> None:
        self.factors = factors
        self.component_index = component_index or 0
        self.default_factor = float(default_factor)

    def _factor(self) -> float:
        if isinstance(self.factors, (int, float)):
            return float(self.factors)
        idx = self.component_index
        if 0 <= idx < len(self.factors):
            return float(self.factors[idx])
        return self.default_factor

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        f = self._factor()
        if f == 0:
            return x
        return x / f

    def transform_batch(self, X: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        f = self._factor()
        if f == 0:
            if out is not None and out is not X:
                out[:] = X
                return out
            return X
        return np.divide(X, f, out=out)
