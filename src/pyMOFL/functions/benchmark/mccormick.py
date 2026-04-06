"""
McCormick function implementation.

This module implements the McCormick function, a 2-dimensional multimodal
test function commonly used in optimization benchmarks.

References
----------
.. [1] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions for global
       optimization problems". International Journal of Mathematical Modelling and Numerical
       Optimisation, 4(2), 150-194. arXiv:1308.4008
       Local documentation: docs/literature_schwefel/jamil_yang_2013_literature_survey.md
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("McCormick")
class McCormickFunction(OptimizationFunction):
    """
    McCormick function.

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal

    Mathematical definition:
        f(x,y) = sin(x + y) + (x - y)² - 1.5x + 2.5y + 1

    Global minimum: f(-0.54719, -1.54719) ≈ -1.9133

    Literature reference: Jamil & Yang 2013 #72 - Classic 2D multimodal benchmark
    """

    def __init__(
        self,
        dimension: int = 2,  # Non-scalable per literature
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("McCormick function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.array([-1.5, -3.0]),
            high=np.array([4.0, 4.0]),
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
        """Evaluate McCormick function."""
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]
        return float(np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate McCormick function for batch."""
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        return np.array([-0.54719, -1.54719]), -1.9133
