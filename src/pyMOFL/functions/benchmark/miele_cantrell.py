"""
Miele-Cantrell benchmark function.

A 4-dimensional test function with mixed polynomial, exponential, and trigonometric terms.

References
----------
.. [1] Miele, A. & Cantrell, J.W. (1969). "Study on a memory gradient method for the
       minimization of functions". Journal of Optimization Theory and Applications, 3(6), 459-470.
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


@register("MieleCantrell")
@register("miele_cantrell")
class MieleCantrellFunction(OptimizationFunction):
    """
    Miele-Cantrell function (4D).

    Mathematical definition:
        f(x) = (exp(-x1) - x2)^4 + 100*(x2 - x3)^6 + (tan(x3 - x4))^4 + x1^8

    Global minimum: f(0, 1, 1, 1) = 0

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal
    """

    def __init__(
        self,
        dimension: int = 4,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 4:
            raise ValueError("Miele-Cantrell function is Non-Scalable and requires dimension=4")

        default_bounds = Bounds(
            low=np.full(4, -1.0),
            high=np.full(4, 1.0),
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
        """Evaluate Miele-Cantrell function."""
        x = self._validate_input(x)
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        return float(
            (np.exp(-x1) - x2) ** 4 + 100.0 * (x2 - x3) ** 6 + np.tan(x3 - x4) ** 4 + x1**8
        )

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Miele-Cantrell function for batch."""
        X = self._validate_batch_input(X)
        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        return (np.exp(-x1) - x2) ** 4 + 100.0 * (x2 - x3) ** 6 + np.tan(x3 - x4) ** 4 + x1**8

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        return np.array([0.0, 1.0, 1.0, 1.0]), 0.0
