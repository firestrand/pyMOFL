"""
De Villiers-Glasser benchmark functions (01 and 02 variants).

Nonlinear least-squares fitting problems with trigonometric and exponential terms.

References
----------
.. [1] De Villiers, N. & Glasser, D. (1981). "A continuation method for nonlinear regression".
       SIAM Journal on Numerical Analysis, 18(6), 1139-1154.
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


@register("DeVilliersGlasser01")
@register("devillers_glasser01")
class DeVilliersGlasser01Function(OptimizationFunction):
    """
    De Villiers-Glasser 01 function (4D).

    Mathematical definition:
        f(x) = sum_{i=1}^{24} (x1 * x2^t_i * sin(x3*t_i + x4) - y_i)^2
        t_i = 0.1*(i-1)
        y_i = 60.137 * 1.371^t_i * sin(3.112*t_i + 1.761)

    Global minimum: f(60.137, 1.371, 3.112, 1.761) = 0

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal
    """

    def __init__(
        self,
        dimension: int = 4,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 4:
            raise ValueError(
                "De Villiers-Glasser 01 function is Non-Scalable and requires dimension=4"
            )

        default_bounds = Bounds(
            low=np.array([1.0, 1.0, 1.0, 0.0]),
            high=np.array([60.0, 2.0, 5.0, 2.0]),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

        # Pre-compute t and y values
        self._t = 0.1 * np.arange(24)
        self._y = 60.137 * 1.371**self._t * np.sin(3.112 * self._t + 1.761)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate De Villiers-Glasser 01 function."""
        x = self._validate_input(x)
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        model = x1 * x2**self._t * np.sin(x3 * self._t + x4)
        residuals = model - self._y
        return float(np.sum(residuals**2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate De Villiers-Glasser 01 function for batch."""
        X = self._validate_batch_input(X)
        x1 = X[:, 0:1]  # (N, 1)
        x2 = X[:, 1:2]
        x3 = X[:, 2:3]
        x4 = X[:, 3:4]

        t = self._t[np.newaxis, :]  # (1, 24)
        model = x1 * x2**t * np.sin(x3 * t + x4)
        residuals = model - self._y[np.newaxis, :]
        return np.sum(residuals**2, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        return np.array([60.137, 1.371, 3.112, 1.761]), 0.0


@register("DeVilliersGlasser02")
@register("devillers_glasser02")
class DeVilliersGlasser02Function(OptimizationFunction):
    """
    De Villiers-Glasser 02 function (5D).

    Mathematical definition:
        f(x) = sum_{i=1}^{16} (x1 * x2^t_i * tanh(x3*t_i + sin(x4*t_i))
                                * cos(t_i * exp(x5)) - y_i)^2
        t_i = 0.1*(i-1)
        y_i = 53.81 * 1.27^t_i * tanh(3.012*t_i + sin(2.13*t_i))
              * cos(t_i * exp(0.507))

    Global minimum: f(53.81, 1.27, 3.012, 2.13, 0.507) ~ 0

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal
    """

    def __init__(
        self,
        dimension: int = 5,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 5:
            raise ValueError(
                "De Villiers-Glasser 02 function is Non-Scalable and requires dimension=5"
            )

        default_bounds = Bounds(
            low=np.array([1.0, 1.0, 1.0, 0.0, 0.0]),
            high=np.array([60.0, 2.0, 5.0, 2.0, 2.0]),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

        # Pre-compute t and y values
        self._t = 0.1 * np.arange(16)
        self._y = (
            53.81
            * 1.27**self._t
            * np.tanh(3.012 * self._t + np.sin(2.13 * self._t))
            * np.cos(self._t * np.exp(0.507))
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate De Villiers-Glasser 02 function."""
        x = self._validate_input(x)
        x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]
        model = (
            x1
            * x2**self._t
            * np.tanh(x3 * self._t + np.sin(x4 * self._t))
            * np.cos(self._t * np.exp(x5))
        )
        residuals = model - self._y
        return float(np.sum(residuals**2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate De Villiers-Glasser 02 function for batch."""
        X = self._validate_batch_input(X)
        x1 = X[:, 0:1]  # (N, 1)
        x2 = X[:, 1:2]
        x3 = X[:, 2:3]
        x4 = X[:, 3:4]
        x5 = X[:, 4:5]

        t = self._t[np.newaxis, :]  # (1, 16)
        model = x1 * x2**t * np.tanh(x3 * t + np.sin(x4 * t)) * np.cos(t * np.exp(x5))
        residuals = model - self._y[np.newaxis, :]
        return np.sum(residuals**2, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        min_point = np.array([53.81, 1.27, 3.012, 2.13, 0.507])
        min_value = self.evaluate(min_point)
        return min_point, min_value
