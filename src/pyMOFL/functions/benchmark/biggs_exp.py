"""
Biggs EXP family of benchmark functions (EXP02 through EXP05).

These are sum-of-squares fitting problems where the objective is to match
exponential model parameters to known target values.

References
----------
.. [1] Biggs, M.C. (1971). "Minimization algorithms making use of non-quadratic
       properties of the objective function". JIMA, 8(3), 315-327.
.. [2] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions
       for global optimization problems". IJMMNO, 4(2), 150-194.
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("BiggsExp02")
@register("biggs_exp02")
class BiggsExp02Function(OptimizationFunction):
    """
    Biggs EXP02 function.

    f(x) = sum_{i=1}^{10} (exp(-t_i*x1) - 5*exp(-t_i*x2) - y_i)^2
    where t_i = 0.1*i, y_i = exp(-t_i) - 5*exp(-10*t_i)

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal
    Domain: [0, 20]^2
    Global minimum: f(1, 10) = 0
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("BiggsExp02 function requires dimension=2")

        default_bounds = Bounds(
            low=np.full(dimension, 0.0),
            high=np.full(dimension, 20.0),
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
        """Evaluate BiggsExp02 function."""
        x = self._validate_input(x)
        t = np.arange(1, 11) * 0.1
        y = np.exp(-t) - 5.0 * np.exp(-10.0 * t)
        residuals = np.exp(-t * x[0]) - 5.0 * np.exp(-t * x[1]) - y
        return float(np.sum(residuals**2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate BiggsExp02 function for a batch."""
        X = self._validate_batch_input(X)
        t = np.arange(1, 11) * 0.1  # shape (10,)
        y = np.exp(-t) - 5.0 * np.exp(-10.0 * t)  # shape (10,)
        # X[:, 0:1] shape (n, 1), t shape (10,) -> broadcast to (n, 10)
        residuals = (
            np.exp(-t[None, :] * X[:, 0:1]) - 5.0 * np.exp(-t[None, :] * X[:, 1:2]) - y[None, :]
        )
        return np.sum(residuals**2, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum: f(1, 10) = 0."""
        return np.array([1.0, 10.0]), 0.0


@register("BiggsExp03")
@register("biggs_exp03")
class BiggsExp03Function(OptimizationFunction):
    """
    Biggs EXP03 function.

    f(x) = sum_{i=1}^{10} (exp(-t_i*x1) - x3*exp(-t_i*x2) - y_i)^2
    where t_i = 0.1*i, y_i = exp(-t_i) - 5*exp(-10*t_i)

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal
    Domain: [0, 20]^3
    Global minimum: f(1, 10, 5) = 0
    """

    def __init__(
        self,
        dimension: int = 3,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 3:
            raise ValueError("BiggsExp03 function requires dimension=3")

        default_bounds = Bounds(
            low=np.full(dimension, 0.0),
            high=np.full(dimension, 20.0),
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
        """Evaluate BiggsExp03 function."""
        x = self._validate_input(x)
        t = np.arange(1, 11) * 0.1
        y = np.exp(-t) - 5.0 * np.exp(-10.0 * t)
        residuals = np.exp(-t * x[0]) - x[2] * np.exp(-t * x[1]) - y
        return float(np.sum(residuals**2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate BiggsExp03 function for a batch."""
        X = self._validate_batch_input(X)
        t = np.arange(1, 11) * 0.1
        y = np.exp(-t) - 5.0 * np.exp(-10.0 * t)
        residuals = (
            np.exp(-t[None, :] * X[:, 0:1])
            - X[:, 2:3] * np.exp(-t[None, :] * X[:, 1:2])
            - y[None, :]
        )
        return np.sum(residuals**2, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum: f(1, 10, 5) = 0."""
        return np.array([1.0, 10.0, 5.0]), 0.0


@register("BiggsExp04")
@register("biggs_exp04")
class BiggsExp04Function(OptimizationFunction):
    """
    Biggs EXP04 function.

    f(x) = sum_{i=1}^{10} (x3*exp(-t_i*x1) - x4*exp(-t_i*x2) - y_i)^2
    where t_i = 0.1*i, y_i = exp(-t_i) - 5*exp(-10*t_i)

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal
    Domain: [0, 20]^4
    Global minimum: f(1, 10, 1, 5) = 0
    """

    def __init__(
        self,
        dimension: int = 4,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 4:
            raise ValueError("BiggsExp04 function requires dimension=4")

        default_bounds = Bounds(
            low=np.full(dimension, 0.0),
            high=np.full(dimension, 20.0),
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
        """Evaluate BiggsExp04 function."""
        x = self._validate_input(x)
        t = np.arange(1, 11) * 0.1
        y = np.exp(-t) - 5.0 * np.exp(-10.0 * t)
        residuals = x[2] * np.exp(-t * x[0]) - x[3] * np.exp(-t * x[1]) - y
        return float(np.sum(residuals**2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate BiggsExp04 function for a batch."""
        X = self._validate_batch_input(X)
        t = np.arange(1, 11) * 0.1
        y = np.exp(-t) - 5.0 * np.exp(-10.0 * t)
        residuals = (
            X[:, 2:3] * np.exp(-t[None, :] * X[:, 0:1])
            - X[:, 3:4] * np.exp(-t[None, :] * X[:, 1:2])
            - y[None, :]
        )
        return np.sum(residuals**2, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum: f(1, 10, 1, 5) = 0."""
        return np.array([1.0, 10.0, 1.0, 5.0]), 0.0


@register("BiggsExp05")
@register("biggs_exp05")
class BiggsExp05Function(OptimizationFunction):
    """
    Biggs EXP05 function.

    f(x) = sum_{i=1}^{10} (x3*exp(-t_i*x1) - x4*exp(-t_i*x2) + 3*exp(-t_i*x5) - y_i)^2
    where t_i = 0.1*i, y_i = exp(-t_i) - 5*exp(-10*t_i) + 3*exp(-4*t_i)

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal
    Domain: [0, 20]^5
    Global minimum: f(1, 10, 1, 5, 4) = 0
    """

    def __init__(
        self,
        dimension: int = 5,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 5:
            raise ValueError("BiggsExp05 function requires dimension=5")

        default_bounds = Bounds(
            low=np.full(dimension, 0.0),
            high=np.full(dimension, 20.0),
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
        """Evaluate BiggsExp05 function."""
        x = self._validate_input(x)
        t = np.arange(1, 11) * 0.1
        y = np.exp(-t) - 5.0 * np.exp(-10.0 * t) + 3.0 * np.exp(-4.0 * t)
        residuals = (
            x[2] * np.exp(-t * x[0]) - x[3] * np.exp(-t * x[1]) + 3.0 * np.exp(-t * x[4]) - y
        )
        return float(np.sum(residuals**2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate BiggsExp05 function for a batch."""
        X = self._validate_batch_input(X)
        t = np.arange(1, 11) * 0.1
        y = np.exp(-t) - 5.0 * np.exp(-10.0 * t) + 3.0 * np.exp(-4.0 * t)
        residuals = (
            X[:, 2:3] * np.exp(-t[None, :] * X[:, 0:1])
            - X[:, 3:4] * np.exp(-t[None, :] * X[:, 1:2])
            + 3.0 * np.exp(-t[None, :] * X[:, 4:5])
            - y[None, :]
        )
        return np.sum(residuals**2, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum: f(1, 10, 1, 5, 4) = 0."""
        return np.array([1.0, 10.0, 1.0, 5.0, 4.0]), 0.0
