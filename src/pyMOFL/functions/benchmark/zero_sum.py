"""
Zero Sum function implementation.

The Zero Sum function has a global minimum of 0 along the hyperplane
where the sum of all variables equals zero.

References
----------
.. [1] Gavana, A. (2013). "Global Optimization Benchmarks and AMPGO".
.. [2] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions
       for global optimization problems". IJMMNO, 4(2), 150-194.
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("ZeroSum")
@register("zero_sum")
class ZeroSumFunction(OptimizationFunction):
    """
    Zero Sum function.

    f(x) = 0                                       if sum(x_i) == 0
    f(x) = 1 + (10000 * |sum(x_i)|)^0.5            otherwise

    Properties: Continuous, Non-Differentiable, Non-Separable, Scalable
    Domain: [-10, 10]^D
    Global minimum: f = 0 when sum(x_i) = 0

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
        """Evaluate the Zero Sum function."""
        x = self._validate_input(x)
        s = np.sum(x)
        if s == 0.0:
            return 0.0
        return float(1.0 + (10000.0 * np.abs(s)) ** 0.5)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the Zero Sum function for a batch."""
        X = self._validate_batch_input(X)
        s = np.sum(X, axis=1)
        result = np.where(s == 0.0, 0.0, 1.0 + (10000.0 * np.abs(s)) ** 0.5)
        return result

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum: f(0, ..., 0) = 0."""
        return np.zeros(self.dimension), 0.0
