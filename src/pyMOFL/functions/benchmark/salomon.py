"""Salomon benchmark function."""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Salomon")
@register("salomon")
class SalomonFunction(OptimizationFunction):
    """
    Salomon function.

    f(x) = 1 - cos(2*pi*||x||) + 0.1*||x||

    Global minimum: f(0, ..., 0) = 0

    References
    ----------
    .. [1] Salomon, R. (1996). "Re-evaluating genetic algorithm performance under coordinate
           rotation of benchmark functions".
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.full(dimension, -100.0),
                high=np.full(dimension, 100.0),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, -100.0),
                high=np.full(dimension, 100.0),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )

    def evaluate(self, x: np.ndarray) -> float:
        x = self._validate_input(x)
        norm = np.sqrt(np.sum(x**2))
        return float(1.0 - np.cos(2.0 * np.pi * norm) + 0.1 * norm)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        norms = np.sqrt(np.sum(X**2, axis=1))
        return 1.0 - np.cos(2.0 * np.pi * norms) + 0.1 * norms

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.zeros(self.dimension), 0.0
