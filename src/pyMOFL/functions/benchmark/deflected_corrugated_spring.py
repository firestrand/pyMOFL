"""
Deflected Corrugated Spring function implementation.

This function models a corrugated spring potential with a single basin
modulated by a cosine wave.

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


@register("DeflectedCorrugatedSpring")
@register("deflected_corrugated_spring")
class DeflectedCorrugatedSpringFunction(OptimizationFunction):
    """
    Deflected Corrugated Spring function.

    f(x) = 0.1 * sum((x_i - alpha)^2) - cos(K * sqrt(sum((x_i - alpha)^2)))

    Parameters
    ----------
    alpha : float
        Shift parameter (default: 5).
    K : float
        Frequency parameter (default: 5).

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Multimodal
    Domain: [0, 2*alpha]^D
    Global minimum: f(alpha, ..., alpha) = -1
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        alpha: float = 5.0,
        K: float = 5.0,
        **kwargs,
    ):
        self._alpha = alpha
        self._K = K
        default_bounds = Bounds(
            low=np.full(dimension, 0.0),
            high=np.full(dimension, 2.0 * alpha),
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
        """Compute the Deflected Corrugated Spring function value."""
        x = self._validate_input(x)
        diff = x - self._alpha
        r2 = np.sum(diff**2)
        return float(0.1 * r2 - np.cos(self._K * np.sqrt(r2)))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Deflected Corrugated Spring function for a batch of points."""
        X = self._validate_batch_input(X)
        diff = X - self._alpha
        r2 = np.sum(diff**2, axis=1)
        return 0.1 * r2 - np.cos(self._K * np.sqrt(r2))

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Deflected Corrugated Spring function.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        return np.full(self.dimension, self._alpha), -1.0
