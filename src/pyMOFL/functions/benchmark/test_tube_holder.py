"""Test Tube Holder function implementation."""

import numpy as np

from pyMOFL.core.bounds import BoundModeEnum, Bounds, QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import register


@register("test_tube_holder")
class TestTubeHolderFunction(OptimizationFunction):
    __test__ = False  # Prevent pytest from collecting this class
    """
    Test Tube Holder function (2D).

    f(x) = -4 * |exp(|cos((x1^2 + x2^2)/200)|) * sin(x1) * cos(x2)|

    Global minimum: f(-pi/2, 0) ≈ -10.8723
    Bounds: [-10, 10]^2
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Test Tube Holder function requires dimension=2")
        default_bounds = Bounds(
            low=np.full(2, -10.0),
            high=np.full(2, 10.0),
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
        inner = np.abs(np.cos((x1**2 + x2**2) / 200.0))
        return float(-4 * np.abs(np.exp(inner) * np.sin(x1) * np.cos(x2)))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        inner = np.abs(np.cos((x1**2 + x2**2) / 200.0))
        return -4 * np.abs(np.exp(inner) * np.sin(x1) * np.cos(x2))

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        point = np.array([-np.pi / 2, 0.0])
        return point, float(self.evaluate(point))
