"""
Katsuura function implementation.

The Katsuura function uses nested sums with rounding operations inside a product,
creating a highly multimodal landscape that is challenging for optimization algorithms.

References
----------
.. [1] Katsuura, H. (1991). "Limits of indeterminate forms associated to
       multivariate power series."
.. [2] Liang, J.J., et al. (2013). "Problem definitions and evaluation criteria
       for the CEC 2013 special session on real-parameter optimization."
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Katsuura")
@register("katsuura")
class KatsuuraFunction(OptimizationFunction):
    """
    Katsuura function (CEC form with normalization).

    f(x) = (10/D²) · Πᵢ₌₁ᴰ (1 + i · Σⱼ₌₁³² |2ʲxᵢ - round(2ʲxᵢ)|/2ʲ)^(10/D^1.2) - 10/D²

    Properties: Continuous, Non-Differentiable, Non-Separable, Scalable, Multimodal
    Domain: [-100, 100]^D
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
            low=np.full(dimension, -100.0),
            high=np.full(dimension, 100.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )
        # Pre-compute constants
        self._norm = 10.0 / (dimension * dimension)
        self._exp = 10.0 / dimension**1.2
        # Pre-compute 2^j for j=1..32
        self._pow2 = 2.0 ** np.arange(1, 33)  # shape (32,)
        # Pre-compute 1/2^j for j=1..32
        self._inv_pow2 = 1.0 / self._pow2  # shape (32,)
        # Pre-compute (i+1) for 0-based i=0..D-1 (gives 1..D)
        self._indices = np.arange(1, dimension + 1, dtype=float)

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Katsuura function."""
        x = self._validate_input(x)
        product = 1.0
        for i in range(self.dimension):
            scaled = self._pow2 * x[i]
            inner_sum = np.sum(np.abs(scaled - np.round(scaled)) * self._inv_pow2)
            product *= (1.0 + (i + 1) * inner_sum) ** self._exp
        return float(self._norm * product - self._norm)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Katsuura for a batch of points."""
        X = self._validate_batch_input(X)
        n = X.shape[0]
        product = np.ones(n)
        for i in range(self.dimension):
            # X[:, i] shape (n,), pow2 shape (32,) -> scaled shape (n, 32)
            scaled = X[:, i : i + 1] * self._pow2  # broadcast to (n, 32)
            inner_sum = np.sum(np.abs(scaled - np.round(scaled)) * self._inv_pow2, axis=1)
            product *= (1.0 + (i + 1) * inner_sum) ** self._exp
        return self._norm * product - self._norm

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Katsuura function.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        return np.zeros(self.dimension), 0.0
