"""
Box-Betts exponential quadratic sum benchmark function.

A 3-dimensional test function involving exponential terms.

References
----------
.. [1] Box, M.J. (1966). "A comparison of several current optimization methods, and the use of
       transformations in constrained problems". The Computer Journal, 9(1), 67-77.
.. [2] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions for global
       optimization problems". International Journal of Mathematical Modelling and Numerical
       Optimisation, 4(2), 150-194.
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("BoxBetts")
@register("box_betts")
class BoxBettsFunction(OptimizationFunction):
    """
    Box-Betts exponential quadratic sum function (3D).

    Mathematical definition:
        f(x) = sum_{i=1}^{10} g_i^2
        g_i = exp(-0.1*i*x1) - exp(-0.1*i*x2) - (exp(-0.1*i) - exp(-i))*x3

    Global minimum: f(1, 10, 1) = 0

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal
    """

    def __init__(
        self,
        dimension: int = 3,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 3:
            raise ValueError("Box-Betts function is Non-Scalable and requires dimension=3")

        default_bounds = Bounds(
            low=np.array([0.9, 9.0, 0.9]),
            high=np.array([1.2, 11.2, 1.2]),
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
        """Evaluate Box-Betts function."""
        x = self._validate_input(x)
        x1, x2, x3 = x[0], x[1], x[2]
        result = 0.0
        for i in range(1, 11):
            g = (
                np.exp(-0.1 * i * x1)
                - np.exp(-0.1 * i * x2)
                - (np.exp(-0.1 * i) - np.exp(-float(i))) * x3
            )
            result += g**2
        return float(result)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Box-Betts function for batch."""
        X = self._validate_batch_input(X)
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        result = np.zeros(len(X))
        for i in range(1, 11):
            g = (
                np.exp(-0.1 * i * x1)
                - np.exp(-0.1 * i * x2)
                - (np.exp(-0.1 * i) - np.exp(-float(i))) * x3
            )
            result += g**2
        return result

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        return np.array([1.0, 10.0, 1.0]), 0.0
