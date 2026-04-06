"""
Keane function implementation.

Keane's function is a multimodal function with a complex landscape that is
challenging for optimization algorithms.

References
----------
.. [1] Keane, A.J. (1994). "Experiences with optimizers in structural design."
       Proc. Conf. on Adaptive Computing in Engineering Design and Control, 14-27.
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


@register("Keane")
@register("keane")
class KeaneFunction(OptimizationFunction):
    """
    Keane function.

    f(x) = -|sum(cos(x_i)^4) - 2*prod(cos(x_i)^2)| / sqrt(sum(i*x_i^2))

    where i is 1-indexed. Returns 0 when the denominator is zero.

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Multimodal
    Domain: [0, 10]^D
    Global minimum: dimension-dependent
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, 0.0),
            high=np.full(dimension, 10.0),
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
        """Compute the Keane function value."""
        x = self._validate_input(x)
        cos_x = np.cos(x)
        numerator = np.abs(np.sum(cos_x**4) - 2.0 * np.prod(cos_x**2))
        indices = np.arange(1, self.dimension + 1, dtype=float)
        denominator = np.sqrt(np.sum(indices * x**2))
        if denominator == 0.0:
            return 0.0
        return float(-numerator / denominator)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Keane function for a batch of points."""
        X = self._validate_batch_input(X)
        cos_X = np.cos(X)
        sum_cos4 = np.sum(cos_X**4, axis=1)
        prod_cos2 = np.prod(cos_X**2, axis=1)
        numerator = np.abs(sum_cos4 - 2.0 * prod_cos2)
        indices = np.arange(1, self.dimension + 1, dtype=float)
        denominator = np.sqrt(np.sum(indices * X**2, axis=1))
        safe_denom = np.where(denominator == 0.0, 1.0, denominator)
        result = np.where(denominator == 0.0, 0.0, -numerator / safe_denom)
        return result

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Keane function.

        The global minimum is dimension-dependent. For D=2, it is approximately
        f(1.39325, 0) ~ -0.67368 (or permutations).

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        if self.dimension == 2:
            x_opt = np.array([1.39325, 0.0])
            return x_opt, float(self.evaluate(x_opt))
        # For general D, return a numerically evaluated point
        # Use a heuristic point and evaluate
        x_opt = np.zeros(self.dimension)
        x_opt[0] = 1.39325
        return x_opt, float(self.evaluate(x_opt))
