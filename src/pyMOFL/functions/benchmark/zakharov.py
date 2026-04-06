"""
Zakharov function implementation.

The Zakharov function combines a quadratic bowl with polynomial terms involving
dimension-weighted sums, creating a unimodal landscape with strong coupling.

References
----------
.. [1] Liang, J.J., et al. (2017). "Problem definitions and evaluation criteria
       for the CEC 2017 special session on single objective real-parameter
       numerical optimization."
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Zakharov")
@register("zakharov")
class ZakharovFunction(OptimizationFunction):
    """
    Zakharov function.

    f(x) = Σxᵢ² + (Σ 0.5·i·xᵢ)² + (Σ 0.5·i·xᵢ)⁴   (1-based i)

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Unimodal
    Domain: [-5, 10]^D
    Global minimum: f(0, ..., 0) = 0
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, -5.0),
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
        # Pre-compute 0.5 * i for i = 1..D (1-based)
        self._weights = 0.5 * np.arange(1, dimension + 1, dtype=float)

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Zakharov: Σxᵢ² + (Σ0.5ixᵢ)² + (Σ0.5ixᵢ)⁴."""
        x = self._validate_input(x)
        sum_sq = np.sum(x**2)
        weighted = np.sum(self._weights * x)
        return float(sum_sq + weighted**2 + weighted**4)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Zakharov for a batch of points."""
        X = self._validate_batch_input(X)
        sum_sq = np.sum(X**2, axis=1)
        weighted = np.sum(self._weights * X, axis=1)
        return sum_sq + weighted**2 + weighted**4

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Zakharov function.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        return np.zeros(self.dimension), 0.0
