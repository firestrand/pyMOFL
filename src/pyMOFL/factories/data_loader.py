"""DataLoader — loads vectors and matrices from disk with {dim} support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DataLoader:
    """Loads vectors and matrices from disk with {dim} support."""

    base_path: str | Path = Path.cwd()

    def resolve_path(self, filename: str, dimension: int | None = None) -> Path:
        """Resolve a resource path relative to the loader base path."""

        return self._resolve(filename, dimension)

    def _resolve(self, filename: str, dimension: int | None = None) -> Path:
        fname = filename
        if dimension is not None and "{dim}" in fname:
            fname = fname.replace("{dim}", str(dimension))
        return Path(self.base_path) / fname

    def load_vector(self, filename: str, dimension: int | None = None) -> np.ndarray:
        path = self._resolve(filename, dimension)
        if not path.exists():
            raise FileNotFoundError(f"Vector file not found: {path}")
        data = np.loadtxt(path, dtype=np.float64)
        if data.ndim == 2:
            data = data.reshape(-1)
        if dimension is not None and data.shape[0] > dimension:
            data = data[:dimension]
        return data

    def load_matrix(self, filename: str, dimension: int | None = None) -> np.ndarray:
        path = self._resolve(filename, dimension)
        if not path.exists():
            raise FileNotFoundError(f"Matrix file not found: {path}")
        data = np.loadtxt(path, dtype=np.float64)
        if data.ndim == 1:
            n = int(np.sqrt(data.size))
            if n * n != data.size:
                raise ValueError("Cannot reshape 1D data into square matrix")
            data = data.reshape((n, n))
        if dimension is not None and (data.shape[0] != dimension or data.shape[1] != dimension):
            data = data[:dimension, :dimension]
        return data
