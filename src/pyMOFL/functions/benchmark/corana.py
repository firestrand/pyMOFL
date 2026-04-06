"""
Corana benchmark function.

A 4-dimensional test function with a piecewise quadratic structure that creates
a flat region around the origin with evenly spaced local minima.

References
----------
.. [1] Corana, A., Marchesi, M., Martini, C. & Ridella, S. (1987). "Minimizing multimodal
       functions of continuous variables with the simulated annealing algorithm". ACM Transactions
       on Mathematical Software, 13(3), 262-280.
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

_D = np.array([1.0, 1000.0, 10.0, 100.0])


@register("Corana")
@register("corana")
class CoranaFunction(OptimizationFunction):
    """
    Corana function (4D).

    Mathematical definition:
        For each dimension i:
            z_i = floor(|x_i/0.2| + 0.49999) * sign(x_i) * 0.2
            if |x_i - z_i| < 0.05:
                f_i = 0.15 * (z_i - 0.05*sign(z_i))^2 * d_i
            else:
                f_i = d_i * x_i^2
        f(x) = sum(f_i)
        d = [1, 1000, 10, 100]

    Global minimum: f(0, 0, 0, 0) = 0

    Properties: Continuous, Non-Differentiable, Separable, Non-Scalable, Multimodal
    """

    def __init__(
        self,
        dimension: int = 4,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 4:
            raise ValueError("Corana function is Non-Scalable and requires dimension=4")

        default_bounds = Bounds(
            low=np.full(4, -5.0),
            high=np.full(4, 5.0),
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
        """Evaluate Corana function."""
        x = self._validate_input(x)
        result = 0.0
        for i in range(4):
            z_i = np.floor(np.abs(x[i] / 0.2) + 0.49999) * np.sign(x[i]) * 0.2
            if np.abs(x[i] - z_i) < 0.05:
                result += 0.15 * (z_i - 0.05 * np.sign(z_i)) ** 2 * _D[i]
            else:
                result += _D[i] * x[i] ** 2
        return float(result)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Corana function for batch."""
        X = self._validate_batch_input(X)
        z = np.floor(np.abs(X / 0.2) + 0.49999) * np.sign(X) * 0.2
        close = np.abs(X - z) < 0.05

        # When |x_i - z_i| < 0.05
        term_close = 0.15 * (z - 0.05 * np.sign(z)) ** 2 * _D[np.newaxis, :]
        # When |x_i - z_i| >= 0.05
        term_far = _D[np.newaxis, :] * X**2

        terms = np.where(close, term_close, term_far)
        return np.sum(terms, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        return np.array([0.0, 0.0, 0.0, 0.0]), 0.0
