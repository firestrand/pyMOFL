"""
Easom function implementation.

This module implements the Easom function, a 2-dimensional test function with
a very small area containing the global minimum compared to the whole search space.
This characteristic makes it challenging for optimization algorithms.

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


@register("Easom")
class EasomFunction(OptimizationFunction):
    """
    Easom function.

    Properties: Continuous, Differentiable, Separable, Non-Scalable, Multimodal

    The Easom function has a very small area containing the global minimum compared
    to the whole search space, making it challenging for optimization algorithms to
    locate the global optimum.

    Mathematical definition:
        f(x,y) = -cos(x)cos(y)exp(-(x-π)² - (y-π)²)

    Global minimum: f(π, π) = -1

    Literature reference: Jamil & Yang 2013 #50 - Separable function with very small global minimum area
    """

    def __init__(
        self,
        dimension: int = 2,  # Non-scalable per literature
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Easom function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.array([-100.0, -100.0]),
            high=np.array([100.0, 100.0]),
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
        """Evaluate Easom function."""
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]
        return float(-np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi) ** 2) - (x2 - np.pi) ** 2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Easom function for batch."""
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi) ** 2) - (x2 - np.pi) ** 2)

    @staticmethod
    def get_global_minimum(dimension: int = 2) -> tuple:
        """Get global minimum."""
        if dimension != 2:
            raise ValueError("Easom requires dimension=2")
        return np.array([np.pi, np.pi]), -1.0
