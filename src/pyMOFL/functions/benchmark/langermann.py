"""Langermann benchmark function."""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Langermann")
@register("langermann")
class LangermannFunction(OptimizationFunction):
    """
    Langermann function.

    f(x) = -sum_j c_j * exp(-1/pi * sum_i (x_i - A_ji)^2) * cos(pi * sum_i (x_i - A_ji)^2)

    Uses standard 5-term coefficients from Jamil & Yang (2013).

    For dimensions > 2, the coefficient matrix A is extended by zero-padding:
    columns beyond the 2nd are filled with zeros. This means additional
    dimensions contribute only through their x_i^2 terms (distance from zero),
    preserving the 2D landscape structure in the first two coordinates while
    adding a radial penalty in higher dimensions.

    References
    ----------
    .. [1] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions
           for global optimization problems".
    """

    # Standard coefficients (m=5 terms)
    _C = np.array([1.0, 2.0, 5.0, 2.0, 3.0])
    _A_2D = np.array(
        [
            [3.0, 5.0],
            [5.0, 2.0],
            [2.0, 1.0],
            [1.0, 4.0],
            [7.0, 9.0],
        ]
    )

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.full(dimension, 0.0),
                high=np.full(dimension, 10.0),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, 0.0),
                high=np.full(dimension, 10.0),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )
        # Build A matrix: for dim <= 2 use standard, for higher dims pad with zeros
        if dimension <= 2:
            self._A = self._A_2D[:, :dimension]
        else:
            self._A = np.zeros((5, dimension))
            self._A[:, :2] = self._A_2D

    def _compute(self, x: np.ndarray) -> float:
        """Compute Langermann for a single point."""
        # diff[j, i] = x_i - A_ji
        diff = x - self._A  # (5, D)
        sq_sums = np.sum(diff**2, axis=1)  # (5,)
        return float(-np.sum(self._C * np.exp(-sq_sums / np.pi) * np.cos(np.pi * sq_sums)))

    def evaluate(self, x: np.ndarray) -> float:
        x = self._validate_input(x)
        return self._compute(x)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        # X: (N, D), A: (5, D)
        # diff: (N, 5, D)
        diff = X[:, np.newaxis, :] - self._A[np.newaxis, :, :]
        sq_sums = np.sum(diff**2, axis=2)  # (N, 5)
        return -np.sum(self._C * np.exp(-sq_sums / np.pi) * np.cos(np.pi * sq_sums), axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        # The Langermann function minimum depends on dimension and is found numerically.
        # For 2D, the approximate known minimum is around (2.00299, 1.006272).
        if self.dimension == 2:
            # Approximate known minimum for 2D
            min_point = np.array([2.00299219, 1.00627194])
            min_value = self._compute(min_point)
            return min_point, min_value
        # For higher dimensions, return the 2D optimum padded
        min_point = np.zeros(self.dimension)
        min_point[0] = 2.00299219
        if self.dimension > 1:
            min_point[1] = 1.00627194
        min_value = self._compute(min_point)
        return min_point, min_value
