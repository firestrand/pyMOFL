"""Inverse Hilbert Matrix function (CEC 2019 F2).

Implements the Inverse Hilbert Matrix benchmark problem, a notoriously ill-conditioned
problem where the objective is to find a matrix X that inverts the Hilbert matrix H.

The Hilbert matrix H_{ij} = 1 / (i + j + 1) is famously ill-conditioned with condition
number growing exponentially with matrix size.

References
----------
.. [1] Price, K. V., Awad, N. H., Ali, M. Z., & Suganthan, P. N. (2019).
       "Problem Definitions and Evaluation Criteria for the 100-Digit Challenge
       on Single Objective Numerical Optimization." Technical Report, CEC 2019.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Hilbert")
@register("hilbert")
class HilbertFunction(OptimizationFunction):
    """Inverse Hilbert Matrix benchmark function (CEC 2019 F2).

    Parameters
    ----------
    dimension : int, optional
        Dimensionality of the problem (must be a perfect square, default is 16 = 4x4).
    initialization_bounds : Bounds, optional
        Bounds for random initialization. Defaults to [-16384, 16384]^D.
    operational_bounds : Bounds, optional
        Bounds for domain enforcement. Defaults to [-16384, 16384]^D.
    """

    def __init__(
        self,
        dimension: int = 16,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        b = int(np.round(np.sqrt(dimension)))
        if b * b != dimension or b < 2:
            raise ValueError(
                f"HilbertFunction requires dimension to be a square >= 4 (e.g. 4, 9, 16), got {dimension}"
            )

        self._b = b
        bound_val = 16384.0 if dimension == 16 else float(2**dimension)

        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.full(dimension, -bound_val, dtype=np.float64),
                high=np.full(dimension, bound_val, dtype=np.float64),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, -bound_val, dtype=np.float64),
                high=np.full(dimension, bound_val, dtype=np.float64),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )

        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )

        # Precompute static Hilbert matrix H and identity matrix I
        i_idx, j_idx = np.indices((self._b, self._b))
        self._hilbert = (1.0 / (i_idx + j_idx + 1.0)).astype(np.float64)
        self._eye = np.eye(self._b, dtype=np.float64)

    def evaluate(self, x: NDArray) -> float:
        """Evaluate L1 deviation ||H * X - I||_1."""
        x = self._validate_input(x)
        # Reshape vector into b x b matrix (row-major order per C code: x[k + b * i])
        mat_x = x.reshape((self._b, self._b))
        # Y = H @ X
        y = self._hilbert @ mat_x
        # L1 norm of (Y - I)
        return float(np.sum(np.abs(y - self._eye)))

    def evaluate_batch(self, X: NDArray) -> NDArray:
        """Batch evaluate L1 deviations."""
        X = self._validate_batch_input(X)
        N = X.shape[0]
        # Reshape into (N, b, b)
        mat_X = X.reshape((N, self._b, self._b))
        # Batch matrix multiplication: (b, b) @ (N, b, b) -> (N, b, b)
        Y = np.einsum("ij,njk->nik", self._hilbert, mat_X)
        diff = np.abs(Y - self._eye)
        return np.sum(diff, axis=(1, 2))

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Return the known inverse Hilbert matrix optimum."""
        if self._b == 3:
            # 3x3 inverse Hilbert matrix
            inv_h = np.array(
                [
                    [9.0, -36.0, 30.0],
                    [-36.0, 192.0, -180.0],
                    [30.0, -180.0, 180.0],
                ]
            )
            return inv_h.ravel(), 0.0
        if self._b == 4:
            # 4x4 inverse Hilbert matrix (CEC 2019 D=16 standard)
            inv_h = np.array(
                [
                    [16.0, -120.0, 240.0, -140.0],
                    [-120.0, 1200.0, -2700.0, 1680.0],
                    [240.0, -2700.0, 6480.0, -4200.0],
                    [-140.0, 1680.0, -4200.0, 2800.0],
                ]
            )
            return inv_h.ravel(), 0.0

        # Exact analytical inverse Hilbert matrix entry formula:
        # (H^-1)_{ij} = (-1)^{i+j} * (i+j-1) * binom(n+i-1, n-j) * binom(n+j-1, n-i) * binom(i+j-2, i-1)^2
        try:
            inv_h = np.linalg.inv(self._hilbert)
            return inv_h.ravel(), 0.0
        except np.linalg.LinAlgError:
            return np.zeros(self.dimension), 0.0
