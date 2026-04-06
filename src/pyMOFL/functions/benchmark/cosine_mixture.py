"""
Cosine Mixture function implementation.

The Cosine Mixture function combines a quadratic bowl with cosine modulations,
creating a multimodal landscape with a known global minimum.

References
----------
.. [1] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions for global
       optimization problems". International Journal of Mathematical Modelling and Numerical
       Optimisation, 4(2), 150-194.
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("CosineMixture")
@register("cosine_mixture")
class CosineMixtureFunction(OptimizationFunction):
    """
    Cosine Mixture function.

    f(x) = -0.1 * sum(cos(5*pi*x_i)) - sum(x_i^2)

    Properties: Continuous, Differentiable, Separable, Scalable, Multimodal
    Domain: [-1, 1]^D
    Global minimum: f(0, ..., 0) = -0.1 * D
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
        """Compute the Cosine Mixture function value."""
        x = self._validate_input(x)
        return float(-0.1 * np.sum(np.cos(5.0 * np.pi * x)) - np.sum(x**2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Cosine Mixture function for a batch of points."""
        X = self._validate_batch_input(X)
        return -0.1 * np.sum(np.cos(5.0 * np.pi * X), axis=1) - np.sum(X**2, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Cosine Mixture function.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        return np.zeros(self.dimension), -0.1 * self.dimension
