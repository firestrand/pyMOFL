"""
Indexed shift transform: selects a shift vector by component index and
applies it to the input: x -> x - shift.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import VectorTransform


class IndexedShiftTransform(VectorTransform):
    def __init__(
        self,
        shifts: np.ndarray | Sequence[np.ndarray],
        component_index: int | None = None,
    ) -> None:
        self.component_index = component_index or 0
        self.shift = self._resolve_shift(shifts)
        self.dimension = self.shift.shape[0]

    def _resolve_shift(self, shifts: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
        if isinstance(shifts, np.ndarray) and shifts.ndim == 1:
            vec = shifts
        elif isinstance(shifts, (list, tuple)):
            idx = min(self.component_index, len(shifts) - 1)
            vec = np.asarray(shifts[idx], dtype=np.float64)
        else:
            vec = np.asarray(shifts, dtype=np.float64)
        return vec

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return x - self.shift
