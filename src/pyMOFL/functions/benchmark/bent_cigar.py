"""
Ill-conditioned unimodal functions: Bent Cigar and Discus.

These are complementary inverse problems from the BBOB-2009 testbed:
- Bent Cigar: x₁² + 10⁶ Σᵢ₌₂ xᵢ² (first variable normal, rest penalized)
- Discus:     10⁶ x₁² + Σᵢ₌₂ xᵢ² (first variable penalized, rest normal)

References
----------
.. [1] Hansen, N., et al. (2010). "Comparing results of 31 algorithms from the
       BBOB-2009 function testbed." GECCO Workshop.
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("BentCigar")
@register("bent_cigar")
class BentCigarFunction(OptimizationFunction):
    """
    Bent Cigar function.

    f(x) = x₁² + 10⁶ Σᵢ₌₂ᴰ xᵢ²

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

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Bent Cigar: x₁² + 10⁶ Σᵢ₌₂ xᵢ²."""
        x = self._validate_input(x)
        return float(x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Bent Cigar for a batch of points."""
        X = self._validate_batch_input(X)
        return X[:, 0] ** 2 + 1e6 * np.sum(X[:, 1:] ** 2, axis=1)

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        """Get the global minimum of the Bent Cigar function.

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


@register("Discus")
@register("discus")
class DiscusFunction(OptimizationFunction):
    """
    Discus (Tablet) function.

    The inverse problem of the Bent Cigar function — the first variable is
    scaled by 10^6 while all others have normal conditioning.

    f(x) = 10⁶ x₁² + Σᵢ₌₂ᴰ xᵢ²

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

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Discus: 10⁶ x₁² + Σᵢ₌₂ xᵢ²."""
        x = self._validate_input(x)
        return float(1e6 * x[0] ** 2 + np.sum(x[1:] ** 2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Discus for a batch of points."""
        X = self._validate_batch_input(X)
        return 1e6 * X[:, 0] ** 2 + np.sum(X[:, 1:] ** 2, axis=1)

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        """Get the global minimum of the Discus function.

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
