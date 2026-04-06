"""
Kowalik benchmark function.

A 4-dimensional test function involving rational expressions with tabulated data.

References
----------
.. [1] Kowalik, J.S. & Osborne, M.R. (1968). "Methods for Unconstrained Optimization Problems".
       Elsevier.
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

# Tabulated data constants for the Kowalik function
_A = np.array(
    [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]
)
_B = np.array([4.0, 2.0, 1.0, 0.5, 0.25, 1.0 / 6.0, 0.125, 0.1, 1.0 / 12.0, 1.0 / 14.0, 1.0 / 16.0])


@register("Kowalik")
@register("kowalik")
class KowalikFunction(OptimizationFunction):
    """
    Kowalik function (4D).

    Mathematical definition:
        f(x) = sum_{i=1}^{11} (a_i - x1*(b_i^2 + b_i*x2) / (b_i^2 + b_i*x3 + x4))^2

    Global minimum: f(0.1928, 0.1909, 0.1231, 0.1358) ~ 3.075e-4

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
            raise ValueError("Kowalik function is Non-Scalable and requires dimension=4")

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
        """Evaluate Kowalik function."""
        x = self._validate_input(x)
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        b2 = _B**2
        numerator = x1 * (b2 + _B * x2)
        denominator = b2 + _B * x3 + x4
        residuals = _A - numerator / denominator
        return float(np.sum(residuals**2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Kowalik function for batch."""
        X = self._validate_batch_input(X)
        x1 = X[:, 0:1]  # (N, 1)
        x2 = X[:, 1:2]
        x3 = X[:, 2:3]
        x4 = X[:, 3:4]

        b = _B[np.newaxis, :]  # (1, 11)
        b2 = b**2
        a = _A[np.newaxis, :]

        numerator = x1 * (b2 + b * x2)
        denominator = b2 + b * x3 + x4
        residuals = a - numerator / denominator
        return np.sum(residuals**2, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        min_point = np.array([0.1928, 0.1909, 0.1231, 0.1358])
        min_value = self.evaluate(min_point)
        return min_point, min_value
