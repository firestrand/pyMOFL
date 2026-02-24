"""
Different Powers function implementation.

The Different Powers function applies variable exponents to each dimension,
ranging from 2 (first) to 6 (last), then takes the square root. This creates
asymmetric sensitivity across dimensions.

References
----------
.. [1] Liang, J.J., et al. (2013). "Problem definitions and evaluation criteria
       for the CEC 2013 special session on real-parameter optimization."
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("DifferentPowers")
@register("different_powers")
class DifferentPowersFunction(OptimizationFunction):
    """
    Different Powers function (CEC form with outer sqrt).

    f(x) = √(Σᵢ₌₁ᴰ |xᵢ|^(2 + 4(i-1)/(D-1)))

    Exponents range from 2 (i=1) to 6 (i=D). For D=1, exponent is 2.

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Unimodal
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
        # Pre-compute exponents: 2 + 4*(i-1)/(D-1) for i=1..D (1-based)
        if dimension == 1:
            self._exponents = np.array([2.0])
        else:
            self._exponents = 2.0 + 4.0 * np.arange(dimension) / (dimension - 1)

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Different Powers: √(Σ |xᵢ|^eᵢ)."""
        x = self._validate_input(x)
        return float(np.sqrt(np.sum(np.abs(x) ** self._exponents)))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Different Powers for a batch of points."""
        X = self._validate_batch_input(X)
        return np.sqrt(np.sum(np.abs(X) ** self._exponents, axis=1))

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        """Get the global minimum of the Different Powers function.

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
