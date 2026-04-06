"""
Csendes (Infinity) function implementation.

The Csendes function features a product of polynomial and trigonometric terms
with a singularity at the origin that must be handled carefully.

References
----------
.. [1] Csendes, T. (1993). "Nonlinear parameter estimation by global optimization -
       efficiency and reliability." Acta Cybernetica, 8, 361-370.
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


@register("Csendes")
@register("csendes")
@register("infinity")
class CsendesFunction(OptimizationFunction):
    """
    Csendes (Infinity) function.

    f(x) = sum(x_i^6 * (2 + sin(1/x_i)))

    The limit as x_i -> 0 is 0 (since x_i^6 dominates), so we define f = 0
    for any component equal to zero.

    Properties: Continuous, Differentiable (except at 0), Separable, Scalable, Multimodal
    Domain: [-1, 1]^D
    Global minimum: f(0, ..., 0) = 0
    """

    def __init__(
        self,
        dimension: int = 2,
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

    def evaluate(self, x: np.ndarray) -> float:
        """Compute the Csendes function value."""
        x = self._validate_input(x)
        # Where x_i == 0, the term is 0 (limit of x^6 * (2 + sin(1/x)) as x->0)
        nonzero = x != 0.0
        terms = np.where(
            nonzero,
            x**6 * (2.0 + np.sin(np.where(nonzero, 1.0 / np.where(nonzero, x, 1.0), 0.0))),
            0.0,
        )
        return float(np.sum(terms))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Csendes function for a batch of points."""
        X = self._validate_batch_input(X)
        nonzero = X != 0.0
        safe_X = np.where(nonzero, X, 1.0)
        terms = np.where(nonzero, X**6 * (2.0 + np.sin(1.0 / safe_X)), 0.0)
        return np.sum(terms, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Csendes function.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        return np.zeros(self.dimension), 0.0
