"""
Matyas function implementation.

This module implements the Matyas function, a classic unimodal test function
with a flat surface that poses difficulties for optimization algorithms since
the flatness provides little directional information for search guidance.

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


@register("Matyas")
class MatyasFunction(OptimizationFunction):
    """
    Matyas function.

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal

    The Matyas function features a flat surface that makes optimization difficult
    since the flatness provides little directional information for search algorithms.

    Mathematical definition:
        f(x,y) = 0.26(x² + y²) - 0.48xy

    Global minimum: f(0, 0) = 0

    Literature reference: Jamil & Yang 2013 #71 - Classic 2D unimodal with flat surface challenges
    """

    def __init__(
        self,
        dimension: int = 2,  # Non-scalable per literature
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Matyas function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.array([-10.0, -10.0]),
            high=np.array([10.0, 10.0]),
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
        """Evaluate Matyas function."""
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]
        return float(0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Matyas function for batch."""
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        return np.array([0.0, 0.0]), 0.0
