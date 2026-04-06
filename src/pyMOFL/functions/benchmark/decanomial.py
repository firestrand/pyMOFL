"""
Decanomial function implementation.

A 2D function featuring high-degree polynomial terms (degree 10 in x1,
degree 4 in x2). The polynomials are constructed so that the global minimum
is exactly zero at (2, -3).

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


@register("Decanomial")
@register("decanomial")
class DecanomialFunction(OptimizationFunction):
    """
    Decanomial function.

    f(x) = 0.001 * (|x2^4 + 12*x2^3 + 54*x2^2 + 108*x2 + 81|
                   + |x1^10 - 20*x1^9 + 180*x1^8 - 960*x1^7 + 3360*x1^6
                     - 8064*x1^5 + 13340*x1^4 - 15360*x1^3 + 11520*x1^2
                     - 5120*x1 + 2624|)^2

    The x2 polynomial is (x2+3)^4 and the x1 polynomial evaluates to 0 at x1=2.

    Properties: Continuous, Differentiable, Separable, Non-Scalable
    Domain: [-10, 10]^2
    Global minimum: f(2, -3) = 0
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Decanomial function requires dimension=2")

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
        """Evaluate the Decanomial function."""
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]

        # Polynomial in x2: x2^4 + 12*x2^3 + 54*x2^2 + 108*x2 + 81 = (x2+3)^4
        poly_x2 = x2**4 + 12.0 * x2**3 + 54.0 * x2**2 + 108.0 * x2 + 81.0

        # Polynomial in x1 (degree 10)
        poly_x1 = (
            x1**10
            - 20.0 * x1**9
            + 180.0 * x1**8
            - 960.0 * x1**7
            + 3360.0 * x1**6
            - 8064.0 * x1**5
            + 13340.0 * x1**4
            - 15360.0 * x1**3
            + 11520.0 * x1**2
            - 5120.0 * x1
            + 2624.0
        )

        return float(0.001 * (np.abs(poly_x2) + np.abs(poly_x1)) ** 2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the Decanomial function for a batch."""
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]

        poly_x2 = x2**4 + 12.0 * x2**3 + 54.0 * x2**2 + 108.0 * x2 + 81.0

        poly_x1 = (
            x1**10
            - 20.0 * x1**9
            + 180.0 * x1**8
            - 960.0 * x1**7
            + 3360.0 * x1**6
            - 8064.0 * x1**5
            + 13340.0 * x1**4
            - 15360.0 * x1**3
            + 11520.0 * x1**2
            - 5120.0 * x1
            + 2624.0
        )

        return 0.001 * (np.abs(poly_x2) + np.abs(poly_x1)) ** 2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum: f(2, -3) = 0."""
        return np.array([2.0, -3.0]), 0.0
