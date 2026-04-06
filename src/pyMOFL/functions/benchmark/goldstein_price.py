"""
Goldstein-Price function implementation.

This module implements the Goldstein-Price function, a classic 2-dimensional
non-convex optimization test function with four local minima and one global minimum.

References
----------
.. [1] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions for global
       optimization problems". International Journal of Mathematical Modelling and Numerical
       Optimisation, 4(2), 150-194. arXiv:1308.4008
       Local documentation: docs/literature_schwefel/jamil_yang_2013_literature_survey.md
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Goldstein_Price")
@register("GoldsteinPrice")
class GoldsteinPriceFunction(OptimizationFunction):
    """
    Goldstein-Price function.

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal

    The Goldstein-Price function is a 2D test function with 4 local minima and 1 global minimum.
    It features scaling problems with many orders of magnitude differences between the domain
    and the function hyper-surface.

    Mathematical definition:
        f(x,y) = [1 + (x + y + 1)²(19 - 14x + 3x² - 14y + 6xy + 3y²)] ×
                 [30 + (2x - 3y)²(18 - 32x + 12x² + 48y - 36xy + 27y²)]

    Global minimum: f(0, -1) = 3

    Literature reference: Jamil & Yang 2013 #58 - Classic 2D multimodal with scaling challenges
    """

    def __init__(
        self,
        dimension: int = 2,  # Non-scalable per literature
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Goldstein-Price function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.array([-2.0, -2.0]),
            high=np.array([2.0, 2.0]),
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
        """Evaluate Goldstein-Price function."""
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]

        # First bracket: [1 + (x + y + 1)²(19 - 14x + 3x² - 14y + 6xy + 3y²)]
        term1_inner = 19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
        term1 = 1 + (x1 + x2 + 1) ** 2 * term1_inner

        # Second bracket: [30 + (2x - 3y)²(18 - 32x + 12x² + 48y - 36xy + 27y²)]
        term2_inner = 18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
        term2 = 30 + (2 * x1 - 3 * x2) ** 2 * term2_inner

        return float(term1 * term2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Goldstein-Price function for batch."""
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]

        # First bracket: [1 + (x + y + 1)²(19 - 14x + 3x² - 14y + 6xy + 3y²)]
        term1_inner = 19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
        term1 = 1 + (x1 + x2 + 1) ** 2 * term1_inner

        # Second bracket: [30 + (2x - 3y)²(18 - 32x + 12x² + 48y - 36xy + 27y²)]
        term2_inner = 18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
        term2 = 30 + (2 * x1 - 3 * x2) ** 2 * term2_inner

        return term1 * term2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        # Known global minimum location
        return np.array([0.0, -1.0]), 3.0
