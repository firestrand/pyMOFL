"""
Deceptive function implementation.

The Deceptive function is designed to deceive optimization algorithms by having
global optima that are not near the center of the search space, with deceptive
local basins.

References
----------
.. [1] Goldberg, D.E. (1987). "Simple genetic algorithms and the minimal deceptive
       problem." Genetic Algorithms and Simulated Annealing, 74-88.
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


@register("Deceptive")
@register("deceptive")
class DeceptiveFunction(OptimizationFunction):
    """
    Deceptive function.

    f(x) = -(1/D * sum(g_i(x_i)))^2

    where alpha_i = i/(D+1) for i = 1..D, and:
        g_i(x_i) = -x_i/alpha_i + 4/5,                          if 0 <= x_i <= 4*alpha_i/5
        g_i(x_i) = 5*x_i/alpha_i - 4,                           if 4*alpha_i/5 < x_i <= alpha_i
        g_i(x_i) = 5*(x_i - alpha_i)/(alpha_i - 1) + 1,         if alpha_i < x_i <= (1 + 4*alpha_i)/5
        g_i(x_i) = (x_i - 1)/(1 - alpha_i) + 4/5,               if (1 + 4*alpha_i)/5 < x_i <= 1

    Properties: Continuous, Non-Differentiable (piecewise), Separable, Scalable, Multimodal
    Domain: [0, 1]^D
    Global minimum: f(alpha_1, ..., alpha_D) = -1
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
            high=np.full(dimension, 1.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )
        # alpha_i = i / (D + 1) for i = 1..D
        self._alpha = np.arange(1, dimension + 1, dtype=float) / (dimension + 1)

    def _g(self, x: np.ndarray) -> np.ndarray:
        """Compute the piecewise deceptive function g_i for each component."""
        alpha = self._alpha
        g = np.empty_like(x)

        # Region 1: 0 <= x_i <= 4*alpha_i/5
        mask1 = x <= 4.0 * alpha / 5.0
        # Region 2: 4*alpha_i/5 < x_i <= alpha_i
        mask2 = (~mask1) & (x <= alpha)
        # Region 3: alpha_i < x_i <= (1 + 4*alpha_i)/5
        mask3 = (~mask1) & (~mask2) & (x <= (1.0 + 4.0 * alpha) / 5.0)
        # Region 4: (1 + 4*alpha_i)/5 < x_i <= 1
        mask4 = ~(mask1 | mask2 | mask3)

        g[mask1] = -x[mask1] / alpha[mask1] + 4.0 / 5.0
        g[mask2] = 5.0 * x[mask2] / alpha[mask2] - 4.0
        g[mask3] = 5.0 * (x[mask3] - alpha[mask3]) / (alpha[mask3] - 1.0) + 1.0
        g[mask4] = (x[mask4] - 1.0) / (1.0 - alpha[mask4]) + 4.0 / 5.0

        return g

    def evaluate(self, x: np.ndarray) -> float:
        """Compute the Deceptive function value."""
        x = self._validate_input(x)
        g = self._g(x)
        avg_g = np.sum(g) / self.dimension
        return float(-(avg_g**2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Deceptive function for a batch of points."""
        X = self._validate_batch_input(X)
        n = X.shape[0]
        results = np.empty(n)
        for i in range(n):
            g = self._g(X[i])
            avg_g = np.sum(g) / self.dimension
            results[i] = -(avg_g**2)
        return results

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Deceptive function.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        return self._alpha.copy(), -1.0
