"""Jennrich-Sampson function implementation."""

import numpy as np

from pyMOFL.core.bounds import BoundModeEnum, Bounds, QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import register


@register("jennrich_sampson")
class JennrichSampsonFunction(OptimizationFunction):
    """
    Jennrich-Sampson function (2D).

    f(x) = sum_{i=1}^{10} (2 + 2*i - (exp(i*x1) + exp(i*x2)))^2

    Global minimum: f(0.257825, 0.257825) ≈ 124.3622
    Bounds: [-1, 1]^2
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Jennrich-Sampson function requires dimension=2")
        default_bounds = Bounds(
            low=np.full(2, -1.0),
            high=np.full(2, 1.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

    def evaluate(self, x: np.ndarray) -> float:
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]
        i = np.arange(1, 11)
        terms = (2 + 2 * i - (np.exp(i * x1) + np.exp(i * x2))) ** 2
        return float(np.sum(terms))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        i = np.arange(1, 11)
        # (batch, 10)
        terms = (2 + 2 * i - (np.exp(np.outer(x1, i)) + np.exp(np.outer(x2, i)))) ** 2
        return np.sum(terms, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        point = np.array([0.257825, 0.257825])
        return point, float(self.evaluate(point))
