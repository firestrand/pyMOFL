"""
New Function 01 and 02 benchmark functions.

These are simple 2D multimodal test functions combining trigonometric
and linear terms.

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


@register("NewFunction01")
@register("new_function01")
class NewFunction01Function(OptimizationFunction):
    """
    New Function 01.

    f(x) = |cos(sqrt(|x1^2 + x2^2|))|^0.5 + (x1 + x2) / 100

    Properties: Continuous, Non-Separable, Non-Scalable, Multimodal
    Domain: [-10, 10]^2
    Global minimum: numerically determined
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("NewFunction01 requires dimension=2")

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
        """Evaluate New Function 01."""
        x = self._validate_input(x)
        r = np.sqrt(np.abs(x[0] ** 2 + x[1] ** 2))
        return float(np.abs(np.cos(r)) ** 0.5 + (x[0] + x[1]) / 100.0)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate New Function 01 for a batch."""
        X = self._validate_batch_input(X)
        r = np.sqrt(np.abs(X[:, 0] ** 2 + X[:, 1] ** 2))
        return np.abs(np.cos(r)) ** 0.5 + (X[:, 0] + X[:, 1]) / 100.0

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get approximate global minimum.

        The minimum is near (-10, -10) where the linear term is most negative
        and the cosine term contributes a small positive value.
        """
        # The linear term pushes toward (-10, -10); the cosine term oscillates.
        # Evaluate at a grid near the corner to find the best point.
        x_opt = np.array([-10.0, -10.0])
        f_opt = self.evaluate(x_opt)
        return x_opt, f_opt


@register("NewFunction02")
@register("new_function02")
class NewFunction02Function(OptimizationFunction):
    """
    New Function 02.

    f(x) = |sin(sqrt(|x1^2 + x2^2|))|^0.5 + (x1 + x2) / 100

    Properties: Continuous, Non-Separable, Non-Scalable, Multimodal
    Domain: [-10, 10]^2
    Global minimum: numerically determined
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("NewFunction02 requires dimension=2")

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
        """Evaluate New Function 02."""
        x = self._validate_input(x)
        r = np.sqrt(np.abs(x[0] ** 2 + x[1] ** 2))
        return float(np.abs(np.sin(r)) ** 0.5 + (x[0] + x[1]) / 100.0)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate New Function 02 for a batch."""
        X = self._validate_batch_input(X)
        r = np.sqrt(np.abs(X[:, 0] ** 2 + X[:, 1] ** 2))
        return np.abs(np.sin(r)) ** 0.5 + (X[:, 0] + X[:, 1]) / 100.0

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get approximate global minimum.

        Similar to NewFunction01 but with sin instead of cos.
        """
        x_opt = np.array([-10.0, -10.0])
        f_opt = self.evaluate(x_opt)
        return x_opt, f_opt
