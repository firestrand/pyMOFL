"""
Zirilli (Aluffi-Pentini) benchmark function.

A 2-dimensional test function with a quartic term in the first variable.

References
----------
.. [1] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions for global
       optimization problems". International Journal of Mathematical Modelling and Numerical
       Optimisation, 4(2), 150-194. arXiv:1308.4008
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Zirilli")
@register("zirilli")
class ZirilliFunction(OptimizationFunction):
    """
    Zirilli (Aluffi-Pentini) function (2D).

    Mathematical definition:
        f(x) = 0.25*x1^4 - 0.5*x1^2 + 0.1*x1 + 0.5*x2^2

    Global minimum: f(-1.0465, 0) ~ -0.3523

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Zirilli function is Non-Scalable and requires dimension=2")

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
        """Evaluate Zirilli function."""
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]
        return float(0.25 * x1**4 - 0.5 * x1**2 + 0.1 * x1 + 0.5 * x2**2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Zirilli function for batch."""
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return 0.25 * x1**4 - 0.5 * x1**2 + 0.1 * x1 + 0.5 * x2**2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        min_point = np.array([-1.0465, 0.0])
        min_value = self.evaluate(min_point)
        return min_point, min_value
