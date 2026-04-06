"""
Dixon-Price function implementation.

The Dixon-Price function has a non-trivial global minimum location where each
coordinate is a different power of 2, making it useful for testing optimization
algorithms' ability to find asymmetric optima.

References
----------
.. [1] Dixon, L.C.W. & Price, R.C. (1988). "The truncated Newton method for
       sparse unconstrained optimization using automatic differentiation."
       J. Optimization Theory and Applications, 56(1).
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("DixonPrice")
@register("dixon_price")
class DixonPriceFunction(OptimizationFunction):
    """
    Dixon-Price function.

    f(x) = (x₁ - 1)² + Σᵢ₌₂ᴰ i·(2xᵢ² - xᵢ₋₁)²

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Unimodal
    Domain: [-10, 10]^D
    Global minimum: f* = 0 at xᵢ = 2^(-(2^i - 2)/2^i) for i = 1..D
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, -10.0),
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
        # Pre-compute i = 2..D for the sum
        self._indices = np.arange(2, dimension + 1, dtype=float)

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Dixon-Price: (x₁-1)² + Σ i(2xᵢ²-xᵢ₋₁)²."""
        x = self._validate_input(x)
        term1 = (x[0] - 1) ** 2
        term2 = np.sum(self._indices * (2 * x[1:] ** 2 - x[:-1]) ** 2)
        return float(term1 + term2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Dixon-Price for a batch of points."""
        X = self._validate_batch_input(X)
        term1 = (X[:, 0] - 1) ** 2
        term2 = np.sum(self._indices * (2 * X[:, 1:] ** 2 - X[:, :-1]) ** 2, axis=1)
        return term1 + term2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Dixon-Price function.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        # x_i = 2^(-(2^i - 2) / 2^i) for i = 1..D (1-based)
        i = np.arange(1, self.dimension + 1, dtype=float)
        x_opt = 2.0 ** (-(2.0**i - 2.0) / 2.0**i)
        return x_opt, 0.0
