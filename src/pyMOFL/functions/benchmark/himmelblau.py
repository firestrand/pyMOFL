"""
Himmelblau function implementation.

This module implements the Himmelblau function, a classic 2-dimensional multimodal
test function with four local minima that are all global minima.

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


@register("Himmelblau")
class HimmelblauFunction(OptimizationFunction):
    """
    Himmelblau function.

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal

    The Himmelblau function is a classic 2D multimodal function with four global minima.
    This makes it useful for testing an algorithm's ability to find multiple optima.

    Mathematical definition:
        f(x,y) = (x² + y - 11)² + (x + y² - 7)²

    Global minima (all equal to 0):
        - (3.0, 2.0)
        - (-2.805118, 3.131312)
        - (-3.779310, -3.283186)
        - (3.584428, -1.848126)

    Literature reference: Jamil & Yang 2013 #65 - Classic 2D multimodal with multiple global optima
    """

    def __init__(
        self,
        dimension: int = 2,  # Non-scalable per literature
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Himmelblau function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.array([-5.0, -5.0]),
            high=np.array([5.0, 5.0]),
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
        """Evaluate Himmelblau function."""
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]
        term1 = (x1**2 + x2 - 11) ** 2
        term2 = (x1 + x2**2 - 7) ** 2
        return float(term1 + term2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Himmelblau function for batch."""
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        term1 = (x1**2 + x2 - 11) ** 2
        term2 = (x1 + x2**2 - 7) ** 2
        return term1 + term2

    @staticmethod
    def get_global_minimum(dimension: int = 2) -> tuple:
        """Get global minimum - returns first of four global minima."""
        if dimension != 2:
            raise ValueError("Himmelblau requires dimension=2")
        # Returns first global minimum - function has 4 equal global minima
        return np.array([3.0, 2.0]), 0.0
