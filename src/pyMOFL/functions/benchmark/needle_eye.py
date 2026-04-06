"""
Needle Eye function implementation.

The Needle Eye function has a narrow "eye" region near the origin where the
function value is minimal (1.0). Outside the eye, a large penalty is applied.

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


@register("NeedleEye")
@register("needle_eye")
class NeedleEyeFunction(OptimizationFunction):
    """
    Needle Eye function.

    f(x) = 1                        if all |x_i| < eye
    f(x) = sum(100 + |x_i|)         if any |x_i| >= eye

    Properties: Discontinuous, Non-Differentiable, Separable, Scalable
    Domain: [-10, 10]^D
    Global minimum: f = 1.0 at any point where all |x_i| < eye

    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    eye : float
        The radius of the needle eye (default 0.0001).
    """

    def __init__(
        self,
        dimension: int,
        eye: float = 0.0001,
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
        self.eye = eye

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Needle Eye function."""
        x = self._validate_input(x)
        abs_x = np.abs(x)
        if np.all(abs_x < self.eye):
            return 1.0
        return float(np.sum(100.0 + abs_x))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the Needle Eye function for a batch."""
        X = self._validate_batch_input(X)
        abs_X = np.abs(X)
        inside = np.all(abs_X < self.eye, axis=1)
        penalties = np.sum(100.0 + abs_X, axis=1)
        return np.where(inside, 1.0, penalties)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum: f(0, ..., 0) = 1.0."""
        return np.zeros(self.dimension), 1.0
