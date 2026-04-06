"""
Multi-Modal function implementation.

A simple multimodal function defined as the product of the sum and product
of absolute values of the input variables.

References
----------
.. [1] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions
       for global optimization problems". IJMMNO, 4(2), 150-194.
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("MultiModal")
@register("multi_modal")
class MultiModalFunction(OptimizationFunction):
    """
    Multi-Modal function.

    f(x) = sum(|x_i|) * prod(|x_i|)

    Properties: Continuous, Non-Differentiable, Separable, Scalable, Multimodal
    Domain: [-10, 10]^D
    Global minimum: f(0, ..., 0) = 0

    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, -10.0),
            high=np.full(dimension, 10.0),
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
        """Evaluate the Multi-Modal function."""
        x = self._validate_input(x)
        abs_x = np.abs(x)
        return float(np.sum(abs_x) * np.prod(abs_x))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the Multi-Modal function for a batch."""
        X = self._validate_batch_input(X)
        abs_X = np.abs(X)
        return np.sum(abs_X, axis=1) * np.prod(abs_X, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum: f(0, ..., 0) = 0."""
        return np.zeros(self.dimension), 0.0
