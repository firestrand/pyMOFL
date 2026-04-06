"""
Oscillation transformation (T_osz from Hansen et al. 2009, BBOB).

A stateless vector-to-vector transformation that introduces smooth oscillations.
Element-wise formula:
  T_osz(0) = 0
  T_osz(x_i) = sign(x_i) * exp(x_hat + 0.049 * (sin(c1*x_hat) + sin(c2*x_hat)))
where x_hat = log(|x_i|), c1 = 10 if x_i > 0 else 5.5, c2 = 7.9 if x_i > 0 else 3.1.
"""

import numpy as np

from .base import VectorTransform


class OscillationTransform(VectorTransform):
    """T_osz oscillation transformation (BBOB).

    Stateless, element-wise.

    Parameters
    ----------
    boundary_only : bool
        If True, only transform the first and last elements (CEC 2013 convention).
        If False, transform all elements (BBOB convention, default).
    """

    def __init__(self, boundary_only: bool = False) -> None:
        self.boundary_only = boundary_only

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(
                f"__call__ expects a 1D vector, got {x.ndim}D array with shape {x.shape}"
            )
        return self._apply(x)

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(
                f"transform_batch expects a 2D array, got {X.ndim}D array with shape {X.shape}"
            )
        return self._apply(X)

    def _apply(self, arr: np.ndarray) -> np.ndarray:
        result = arr.copy()

        if self.boundary_only and arr.shape[-1] > 0:
            # CEC 2013: only transform first and last elements
            if arr.ndim == 1:
                indices = [0, arr.shape[0] - 1] if arr.shape[0] > 1 else [0]
                for idx in indices:
                    result[idx] = self._osz_scalar(arr[idx])
            else:
                for idx in [0, arr.shape[1] - 1] if arr.shape[1] > 1 else [0]:
                    result[..., idx] = np.vectorize(self._osz_scalar)(arr[..., idx])
            return result

        nonzero = arr != 0.0
        if not np.any(nonzero):
            return result

        vals = arr[nonzero]
        x_hat = np.log(np.abs(vals))

        c1 = np.where(vals > 0, 10.0, 5.5)
        c2 = np.where(vals > 0, 7.9, 3.1)

        result[nonzero] = np.sign(vals) * np.exp(
            x_hat + 0.049 * (np.sin(c1 * x_hat) + np.sin(c2 * x_hat))
        )
        return result

    @staticmethod
    def _osz_scalar(val: float) -> float:
        if val == 0.0:
            return 0.0
        x_hat = np.log(np.abs(val))
        c1 = 10.0 if val > 0 else 5.5
        c2 = 7.9 if val > 0 else 3.1
        sign = 1.0 if val > 0 else -1.0
        return sign * np.exp(x_hat + 0.049 * (np.sin(c1 * x_hat) + np.sin(c2 * x_hat)))
