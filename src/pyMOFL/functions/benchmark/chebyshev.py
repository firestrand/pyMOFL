"""Storn's Chebyshev Polynomial Fitting function (CEC 2019 F1).

Implements Storn's Chebyshev polynomial fitting problem, a highly ill-conditioned
multimodal problem where the objective is to find polynomial coefficients that
approximate the Chebyshev polynomial T_{D-1}(y) on [-1, 1] while satisfying boundary
growth conditions at y = 1.2.

References
----------
.. [1] Storn, R., & Price, K. (1997). "Differential Evolution – A Simple and Efficient
       Heuristic for global Optimization over Continuous Spaces." Journal of Global
       Optimization, 11(4), 341-359.
.. [2] Price, K. V., Awad, N. H., Ali, M. Z., & Suganthan, P. N. (2019).
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


@register("Chebyshev")
@register("chebyshev")
class ChebyshevFunction(OptimizationFunction):
    """Storn's Chebyshev Polynomial Fitting benchmark function (CEC 2019 F1).

    Parameters
    ----------
    dimension : int, optional
        Dimensionality of the problem (number of polynomial coefficients), default is 9.
    initialization_bounds : Bounds, optional
        Bounds for random initialization. Defaults to [-8192, 8192]^D.
    operational_bounds : Bounds, optional
        Bounds for domain enforcement. Defaults to [-8192, 8192]^D.
    """

    def __init__(
        self,
        dimension: int = 9,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension < 3:
            raise ValueError(f"ChebyshevFunction requires dimension >= 3, got {dimension}")

        bound_val = 8192.0 if dimension == 9 else float(2**dimension)
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

        # Precompute dx = T_{D-1}(1.2) via three-term recurrence
        a = 1.0
        b = 1.2
        dx = b
        for _ in range(dimension - 2):
            dx = 2.4 * b - a
            a = b
            b = dx
        self._dx = float(dx)

        self._sample = 32 * dimension
        self._sample_y = np.linspace(-1.0, 1.0, self._sample + 1, dtype=np.float64)

    def evaluate(self, x: NDArray) -> float:
        """Evaluate Chebyshev polynomial fitting function at point x."""
        x = self._validate_input(x)
        # Evaluate polynomial P(y) = sum_{j=0}^{D-1} x[j] * y^{D-1-j} at sample points
        # Using Horner's method across sample points
        px = np.full_like(self._sample_y, x[0])
        for j in range(1, self.dimension):
            px = self._sample_y * px + x[j]

        # Penalize deviations outside [-1, 1]
        violations = np.abs(px) > 1.0
        sum_val = float(np.sum((1.0 - np.abs(px[violations])) ** 2))

        # Boundary condition at 1.2
        p_12 = float(x[0])
        for j in range(1, self.dimension):
            p_12 = 1.2 * p_12 + float(x[j])

        # CEC 2019 reference implementation runs the boundary check twice (i in {-1, 1})
        if p_12 < self._dx:
            sum_val += 2.0 * (p_12**2)

        return float(sum_val)

    def evaluate_batch(self, X: NDArray, out: NDArray | None = None) -> NDArray:
        """Batch evaluation of Chebyshev polynomial fitting function."""
        X = self._validate_batch_input(X)
        # Evaluate polynomial P(y) across sample points for all batch rows using Horner's method
        px = np.tile(X[:, 0:1], (1, len(self._sample_y)))
        sample_y_row = self._sample_y[None, :]
        for j in range(1, self.dimension):
            px = sample_y_row * px + X[:, j : j + 1]

        # Penalize deviations outside [-1, 1]
        violations = np.abs(px) > 1.0
        sum_val = np.sum(np.where(violations, (1.0 - np.abs(px)) ** 2, 0.0), axis=1)

        # Boundary condition at 1.2
        p_12 = X[:, 0].copy()
        for j in range(1, self.dimension):
            p_12 = 1.2 * p_12 + X[:, j]

        # CEC 2019 reference implementation runs the boundary check twice (i in {-1, 1})
        boundary_violation = p_12 < self._dx
        sum_val += np.where(boundary_violation, 2.0 * (p_12**2), 0.0)

        if out is not None:
            out[:] = sum_val
            return out
        return sum_val

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Return the known global minimum point and value for standard dimensions."""
        if self.dimension == 9:
            # T_8(x) = 128*x^8 - 256*x^6 + 160*x^4 - 32*x^2 + 1
            x_opt = np.array([128.0, 0.0, -256.0, 0.0, 160.0, 0.0, -32.0, 0.0, 1.0])
            return x_opt, 0.0
        if self.dimension == 17:
            x_opt = np.array(
                [
                    32768.0,
                    0.0,
                    -131072.0,
                    0.0,
                    212992.0,
                    0.0,
                    -180224.0,
                    0.0,
                    84480.0,
                    0.0,
                    -21504.0,
                    0.0,
                    2688.0,
                    0.0,
                    -128.0,
                    0.0,
                    1.0,
                ]
            )
            return x_opt, 0.0
        return np.zeros(self.dimension), 0.0
