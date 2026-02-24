"""
Indexed rotation transform: selects a rotation matrix by component index
and applies the CEC rotation: x -> M.T @ x.
"""

from __future__ import annotations

import numpy as np

from .base import VectorTransform


class IndexedRotateTransform(VectorTransform):
    def __init__(
        self,
        matrices: np.ndarray,
        component_index: int | None = None,
        matrix_dimension: int | None = None,
    ) -> None:
        self.component_index = component_index or 0
        self.matrix = self._resolve_matrix(np.asarray(matrices), matrix_dimension)
        self.dimension = self.matrix.shape[0]

    def _resolve_matrix(self, data: np.ndarray, matrix_dimension: int | None) -> np.ndarray:
        # Single square matrix
        if data.ndim == 2 and data.shape[0] == data.shape[1]:
            return data

        # Vertically stacked blocks (k*d, d)
        if data.ndim == 2 and matrix_dimension is not None:
            m = matrix_dimension
            start = self.component_index * m
            end = start + m
            block = data[start:end, :]
            assert block.shape == (m, m)
            return block

        # 3D stack (k, d, d)
        if data.ndim == 3:
            idx = min(self.component_index, data.shape[0] - 1)
            block = data[idx]
            assert block.shape[0] == block.shape[1]
            return block

        # Attempt reshape for flat
        n = int(np.sqrt(data.size))
        if n * n == data.size:
            return data.reshape((n, n))
        raise ValueError("Unsupported matrix shape for IndexedRotateTransform")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return self.matrix.T @ x
