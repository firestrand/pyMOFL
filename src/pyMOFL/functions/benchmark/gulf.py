"""
Gulf research benchmark function.

A 3-dimensional test function involving exponential and power terms.

References
----------
.. [1] More, J.J., Garbow, B.S. & Hillstrom, K.E. (1981). "Testing unconstrained optimization
       software". ACM Transactions on Mathematical Software, 7(1), 17-41.
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


@register("Gulf")
@register("gulf")
class GulfFunction(OptimizationFunction):
    """
    Gulf research function (3D).

    Mathematical definition:
        f(x) = sum_{i=1}^{99} (exp(-|y_i - x2|^x3 / x1) - t_i)^2
        t_i = i/100
        y_i = 25 + (-50*ln(t_i))^(2/3)

    Global minimum: f(50, 25, 1.5) = 0

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal
    """

    def __init__(
        self,
        dimension: int = 3,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 3:
            raise ValueError("Gulf function is Non-Scalable and requires dimension=3")

        default_bounds = Bounds(
            low=np.array([0.1, 0.1, 0.1]),
            high=np.array([100.0, 100.0, 100.0]),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

        # Pre-compute t and y values
        self._t = np.arange(1, 100) / 100.0
        self._y = 25.0 + (-50.0 * np.log(self._t)) ** (2.0 / 3.0)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate Gulf function."""
        x = self._validate_input(x)
        x1, x2, x3 = x[0], x[1], x[2]
        diff = np.abs(self._y - x2)
        exponent = -(diff**x3) / x1
        residuals = np.exp(exponent) - self._t
        return float(np.sum(residuals**2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Gulf function for batch."""
        X = self._validate_batch_input(X)
        x1 = X[:, 0]  # (N,)
        x2 = X[:, 1]  # (N,)
        x3 = X[:, 2]  # (N,)

        # Broadcast: y (99,), x2 (N,) -> diff (N, 99)
        diff = np.abs(self._y[np.newaxis, :] - x2[:, np.newaxis])
        exponent = -(diff ** x3[:, np.newaxis]) / x1[:, np.newaxis]
        residuals = np.exp(exponent) - self._t[np.newaxis, :]
        return np.sum(residuals**2, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        return np.array([50.0, 25.0, 1.5]), 0.0
