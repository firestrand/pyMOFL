"""
Sum of Different Powers function implementation.

The Sum of Different Powers function applies increasing integer exponents to
each variable's absolute value, with the first variable squared and the last
raised to the (D+1)-th power.

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


@register("SumDifferentPowers")
@register("sum_different_powers")
class SumDifferentPowersFunction(OptimizationFunction):
    """
    Sum of Different Powers function.

    f(x) = Σᵢ₌₁ᴰ |xᵢ|^(i+1)   (1-based i)

    Exponents range from 2 (i=1) to D+1 (i=D).

    Properties: Continuous, Differentiable, Separable, Scalable, Unimodal
    Domain: [-1, 1]^D
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
            low=np.full(dimension, -1.0),
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
        # Pre-compute exponents: i+1 for i=1..D (1-based) = [2, 3, ..., D+1]
        self._exponents = np.arange(2, dimension + 2, dtype=float)

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Sum of Different Powers: Σ |xᵢ|^(i+1)."""
        x = self._validate_input(x)
        return float(np.sum(np.abs(x) ** self._exponents))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Sum of Different Powers for a batch of points."""
        X = self._validate_batch_input(X)
        return np.sum(np.abs(X) ** self._exponents, axis=1)

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        """Get the global minimum of the Sum of Different Powers function.

        Parameters
        ----------
        dimension : int
            The dimension of the function.

        Returns
        -------
        tuple
            (global_min_point, global_min_value)
        """
        return np.zeros(dimension), 0.0
