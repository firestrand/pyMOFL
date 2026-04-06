"""
Asymmetric transformation (T_asy^beta from Hansen et al. 2009, BBOB).

A parameterized vector-to-vector transformation that scales positive elements asymmetrically.
Element-wise formula:
  For x_i > 0:  T_asy(x_i) = x_i^(1 + beta * (i/(D-1)) * sqrt(x_i))
  For x_i <= 0: T_asy(x_i) = x_i  (identity)
  When D=1: ratio i/(D-1) = 0 for i=0, so exponent = 1 (identity for all beta).
"""

import numpy as np

from .base import VectorTransform


class AsymmetricTransform(VectorTransform):
    """T_asy^beta asymmetric transformation (BBOB).

    Parameters
    ----------
    beta : float
        Controls the strength of asymmetry.
    dimension : int
        Expected input dimension.
    """

    def __init__(self, beta: float, dimension: int) -> None:
        self.beta = float(beta)
        self.dimension = dimension
        if dimension == 1:
            self._ratios = np.array([0.0])
        else:
            self._ratios = np.arange(dimension, dtype=np.float64) / (dimension - 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.dimension,):
            raise ValueError(
                f"Input dimension {x.shape} does not match expected dimension ({self.dimension},)"
            )
        return self._apply(x, self._ratios)

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.dimension:
            raise ValueError(
                f"Batch dimension {X.shape} does not match expected dimension (N, {self.dimension})"
            )
        return self._apply(X, self._ratios)

    def _apply(self, arr: np.ndarray, ratios: np.ndarray) -> np.ndarray:
        result = arr.copy()
        positive = arr > 0

        if not np.any(positive):
            return result

        # For batch (2D): broadcast ratios as (1, D)
        if arr.ndim == 2:
            r = ratios[np.newaxis, :]
        else:
            r = ratios

        exponent = 1.0 + self.beta * r * np.sqrt(np.where(positive, arr, 0.0))
        result = np.where(positive, np.power(arr, exponent), arr)
        return result
