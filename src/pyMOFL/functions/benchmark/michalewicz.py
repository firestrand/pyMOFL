"""Michalewicz benchmark function."""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Michalewicz")
@register("michalewicz")
class MichalewiczFunction(OptimizationFunction):
    """
    Michalewicz function.

    f(x) = -sum(sin(x_i) * sin(i * x_i^2 / pi)^(2*m))

    The parameter m defines the steepness of valleys (default m=10).

    Global minimum depends on dimension:
    - 2D: f* ~ -1.8013 at approximately (2.20, 1.57)
    - 5D: f* ~ -4.6877
    - 10D: f* ~ -9.6602

    References
    ----------
    .. [1] Michalewicz, Z. (1996). "Genetic Algorithms + Data Structures = Evolution Programs".
    """

    # Known approximate optima for common dimensions
    _KNOWN_OPTIMA = {
        2: (np.array([2.20290552, 1.57079633]), -1.8013034),
        5: (None, -4.687658),
        10: (None, -9.660152),
    }

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        m: int = 10,
        **kwargs,
    ):
        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.full(dimension, 0.0),
                high=np.full(dimension, np.pi),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, 0.0),
                high=np.full(dimension, np.pi),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )
        self._m = m

    def evaluate(self, x: np.ndarray) -> float:
        x = self._validate_input(x)
        i = np.arange(1, self.dimension + 1, dtype=float)
        return float(-np.sum(np.sin(x) * np.sin(i * x**2 / np.pi) ** (2 * self._m)))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        i = np.arange(1, self.dimension + 1, dtype=float)
        return -np.sum(np.sin(X) * np.sin(i * X**2 / np.pi) ** (2 * self._m), axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        if self.dimension in self._KNOWN_OPTIMA:
            point, value = self._KNOWN_OPTIMA[self.dimension]
            if point is not None:
                return point.copy(), value
        # For dimensions without known exact optima, return approximate
        # Use the 2D known point structure as hint
        point = np.full(self.dimension, np.pi / 2.0)
        value = self.evaluate(point)
        return point, value
