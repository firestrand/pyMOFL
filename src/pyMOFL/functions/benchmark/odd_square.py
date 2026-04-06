"""
Odd Square function implementation.

The Odd Square function combines distance-based and norm-based terms with
cosine modulation, creating a complex landscape.

References
----------
.. [1] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions for global
       optimization problems". International Journal of Mathematical Modelling and Numerical
       Optimisation, 4(2), 150-194.
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("OddSquare")
@register("odd_square")
class OddSquareFunction(OptimizationFunction):
    """
    Odd Square function.

    f(x) = -exp(-d/(2*pi)) * cos(pi*d) * (1 + 0.02*h/(d + 0.01))

    where d = sqrt(sum((x_i - b_i)^2)), h = sum(x_i^2), b_i = 1 for all i.

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Multimodal
    Domain: [-5*pi, 5*pi]^D
    Global minimum: f(b) ~ -1.0
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, -5.0 * np.pi),
            high=np.full(dimension, 5.0 * np.pi),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )
        self._b = np.ones(dimension)

    def evaluate(self, x: np.ndarray) -> float:
        """Compute the Odd Square function value."""
        x = self._validate_input(x)
        d = np.sqrt(np.sum((x - self._b) ** 2))
        h = np.sum(x**2)
        return float(
            -np.exp(-d / (2.0 * np.pi)) * np.cos(np.pi * d) * (1.0 + 0.02 * h / (d + 0.01))
        )

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Odd Square function for a batch of points."""
        X = self._validate_batch_input(X)
        d = np.sqrt(np.sum((X - self._b) ** 2, axis=1))
        h = np.sum(X**2, axis=1)
        return -np.exp(-d / (2.0 * np.pi)) * np.cos(np.pi * d) * (1.0 + 0.02 * h / (d + 0.01))

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Odd Square function.

        At x = b, d = 0, so f(b) = -exp(0) * cos(0) * (1 + 0.02*h/0.01)
        = -1 * 1 * (1 + 2*h) = -(1 + 2*sum(b_i^2)) = -(1 + 2*D).

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        x_opt = self._b.copy()
        return x_opt, float(self.evaluate(x_opt))
