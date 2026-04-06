"""
Mishra benchmark function family (Mishra 01-11).

A collection of benchmark functions proposed by S.K. Mishra for testing
global optimization algorithms. Includes both scalable and fixed-dimension
variants with diverse landscape characteristics.

References
----------
.. [1] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions for global
       optimization problems". International Journal of Mathematical Modelling and Numerical
       Optimisation, 4(2), 150-194.
.. [2] Mishra, S.K. (2006). "Some new test functions for global optimization and
       performance of repulsive particle swarm method." SSRN Electronic Journal.
"""

import math

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Mishra01")
@register("mishra01")
class Mishra01Function(OptimizationFunction):
    """
    Mishra 01 function (scalable).

    Mathematical definition:
        xn = n - sum_{i=1}^{n-1} x_i
        f(x) = (1 + xn)^xn

    Properties: Continuous, Differentiable, Separable, Scalable, Multimodal
    Domain: [0, 1]^D
    Global minimum: f(1, ..., 1) = 2
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, 0.0),
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
        """Compute Mishra 01 function."""
        x = self._validate_input(x)
        n = self.dimension
        xn = n - np.sum(x[:-1])
        return float((1.0 + xn) ** xn)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Mishra 01 function for batch."""
        X = self._validate_batch_input(X)
        n = self.dimension
        xn = n - np.sum(X[:, :-1], axis=1)
        return (1.0 + xn) ** xn

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum at (1, ..., 1) with value 2."""
        return np.ones(self.dimension), 2.0


@register("Mishra02")
@register("mishra02")
class Mishra02Function(OptimizationFunction):
    """
    Mishra 02 function (scalable).

    Mathematical definition:
        xn = n - sum_{i=1}^{n-1} (x_i + x_{i+1}) / 2
        f(x) = (1 + xn)^xn

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Multimodal
    Domain: [0, 1]^D
    Global minimum: f(1, ..., 1) = 2
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, 0.0),
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
        """Compute Mishra 02 function."""
        x = self._validate_input(x)
        n = self.dimension
        avg_sum = np.sum((x[:-1] + x[1:]) / 2.0)
        xn = n - avg_sum
        return float((1.0 + xn) ** xn)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Mishra 02 function for batch."""
        X = self._validate_batch_input(X)
        n = self.dimension
        avg_sum = np.sum((X[:, :-1] + X[:, 1:]) / 2.0, axis=1)
        xn = n - avg_sum
        return (1.0 + xn) ** xn

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum at (1, ..., 1) with value 2."""
        return np.ones(self.dimension), 2.0


@register("Mishra03")
@register("mishra03")
class Mishra03Function(OptimizationFunction):
    """
    Mishra 03 function (2D fixed).

    Mathematical definition:
        f(x) = sqrt(|cos(sqrt(|x1^2 + x2^2|))|) + 0.01*(x1 + x2)

    Properties: Continuous, Non-Differentiable, Non-Separable, Non-Scalable, Multimodal
    Domain: [-10, 10]^2
    Global minimum: f(-10, -10) (computed numerically)
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Mishra03 function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.full(2, -10.0),
            high=np.full(2, 10.0),
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
        """Compute Mishra 03 function."""
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]
        return float(np.sqrt(np.abs(np.cos(np.sqrt(np.abs(x1**2 + x2**2))))) + 0.01 * (x1 + x2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Mishra 03 function for batch."""
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return np.sqrt(np.abs(np.cos(np.sqrt(np.abs(x1**2 + x2**2))))) + 0.01 * (x1 + x2)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum (computed numerically)."""
        opt_point = np.array([-10.0, -10.0])
        opt_value = self.evaluate(opt_point)
        return opt_point, opt_value


@register("Mishra04")
@register("mishra04")
class Mishra04Function(OptimizationFunction):
    """
    Mishra 04 function (2D fixed).

    Mathematical definition:
        f(x) = sqrt(|sin(sqrt(|x1^2 + x2^2|))|) + 0.01*(x1 + x2)

    Properties: Continuous, Non-Differentiable, Non-Separable, Non-Scalable, Multimodal
    Domain: [-10, 10]^2
    Global minimum: f(-10, -10) (computed numerically)
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Mishra04 function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.full(2, -10.0),
            high=np.full(2, 10.0),
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
        """Compute Mishra 04 function."""
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]
        return float(np.sqrt(np.abs(np.sin(np.sqrt(np.abs(x1**2 + x2**2))))) + 0.01 * (x1 + x2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Mishra 04 function for batch."""
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return np.sqrt(np.abs(np.sin(np.sqrt(np.abs(x1**2 + x2**2))))) + 0.01 * (x1 + x2)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum (computed numerically)."""
        opt_point = np.array([-10.0, -10.0])
        opt_value = self.evaluate(opt_point)
        return opt_point, opt_value


@register("Mishra05")
@register("mishra05")
class Mishra05Function(OptimizationFunction):
    """
    Mishra 05 function (2D fixed).

    Mathematical definition:
        f(x) = (sin((cos(x1) + cos(x2))^2)^2 + cos((sin(x1) + sin(x2))^2)^2 + x1)^2
               + 0.01*(x1 + x2)

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal
    Domain: [-10, 10]^2
    Global minimum: f(-1.98682, -10) (computed numerically)
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Mishra05 function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.full(2, -10.0),
            high=np.full(2, 10.0),
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
        """Compute Mishra 05 function."""
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]
        term_sin = np.sin((np.cos(x1) + np.cos(x2)) ** 2) ** 2
        term_cos = np.cos((np.sin(x1) + np.sin(x2)) ** 2) ** 2
        return float((term_sin + term_cos + x1) ** 2 + 0.01 * (x1 + x2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Mishra 05 function for batch."""
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        term_sin = np.sin((np.cos(x1) + np.cos(x2)) ** 2) ** 2
        term_cos = np.cos((np.sin(x1) + np.sin(x2)) ** 2) ** 2
        return (term_sin + term_cos + x1) ** 2 + 0.01 * (x1 + x2)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum (computed numerically)."""
        opt_point = np.array([-1.98682, -10.0])
        opt_value = self.evaluate(opt_point)
        return opt_point, opt_value


@register("Mishra06")
@register("mishra06")
class Mishra06Function(OptimizationFunction):
    """
    Mishra 06 function (2D fixed).

    Mathematical definition:
        f(x) = -log((sin((cos(x1)+cos(x2))^2)^2 - cos((sin(x1)+sin(x2))^2)^2 + x1)^2)
               + 0.01*((x1-1)^2 + (x2-1)^2)

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal
    Domain: [-10, 10]^2
    Global minimum: f(2.886, 1.823) ~ -2.2840 (computed numerically)
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Mishra06 function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.full(2, -10.0),
            high=np.full(2, 10.0),
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
        """Compute Mishra 06 function."""
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]
        term_sin = np.sin((np.cos(x1) + np.cos(x2)) ** 2) ** 2
        term_cos = np.cos((np.sin(x1) + np.sin(x2)) ** 2) ** 2
        inner = (term_sin - term_cos + x1) ** 2
        return float(-np.log(inner) + 0.01 * ((x1 - 1.0) ** 2 + (x2 - 1.0) ** 2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Mishra 06 function for batch."""
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        term_sin = np.sin((np.cos(x1) + np.cos(x2)) ** 2) ** 2
        term_cos = np.cos((np.sin(x1) + np.sin(x2)) ** 2) ** 2
        inner = (term_sin - term_cos + x1) ** 2
        return -np.log(inner) + 0.01 * ((x1 - 1.0) ** 2 + (x2 - 1.0) ** 2)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum (computed numerically)."""
        opt_point = np.array([2.886, 1.823])
        opt_value = self.evaluate(opt_point)
        return opt_point, opt_value


@register("Mishra07")
@register("mishra07")
class Mishra07Function(OptimizationFunction):
    """
    Mishra 07 function (scalable).

    Mathematical definition:
        f(x) = (prod(x_i) - n!)^2

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Multimodal
    Domain: [-10, 10]^D
    Global minimum: f(n!^{1/n}, ..., n!^{1/n}) = 0
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
        """Compute Mishra 07 function."""
        x = self._validate_input(x)
        n_fact = math.factorial(self.dimension)
        return float((np.prod(x) - n_fact) ** 2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Mishra 07 function for batch."""
        X = self._validate_batch_input(X)
        n_fact = math.factorial(self.dimension)
        return (np.prod(X, axis=1) - n_fact) ** 2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum where all components equal n!^{1/n}."""
        n = self.dimension
        n_fact = math.factorial(n)
        opt_val = n_fact ** (1.0 / n)
        return np.full(n, opt_val), 0.0


@register("Mishra08")
@register("mishra08")
class Mishra08Function(OptimizationFunction):
    """
    Mishra 08 function (2D fixed), also known as Mishra-Decanomial.

    Mathematical definition:
        g1(x1) = x1^10 - 20*x1^9 + 180*x1^8 - 960*x1^7 + 3360*x1^6
                 - 8064*x1^5 + 13340*x1^4 - 15360*x1^3 + 11520*x1^2
                 - 5120*x1 + 2624
        g2(x2) = x2^4 + 12*x2^3 + 54*x2^2 + 108*x2 + 81
        f(x) = 0.001 * (|g1(x1)| * |g2(x2)|)^2

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal
    Domain: [-10, 10]^2
    Global minimum: f(2, -3) = 0

    Note: g1(x1) = (x1 - 2)^10 and g2(x2) = (x2 + 3)^4 in factored form.
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Mishra08 function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.full(2, -10.0),
            high=np.full(2, 10.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

    @staticmethod
    def _g1(x1):
        """Compute the degree-10 polynomial in x1."""
        return (
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

    @staticmethod
    def _g2(x2):
        """Compute the degree-4 polynomial in x2."""
        return x2**4 + 12.0 * x2**3 + 54.0 * x2**2 + 108.0 * x2 + 81.0

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Mishra 08 function."""
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]
        return float(0.001 * (np.abs(self._g1(x1)) * np.abs(self._g2(x2))) ** 2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Mishra 08 function for batch."""
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return 0.001 * (np.abs(self._g1(x1)) * np.abs(self._g2(x2))) ** 2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum at (2, -3) with value 0."""
        return np.array([2.0, -3.0]), 0.0


@register("Mishra09")
@register("mishra09")
class Mishra09Function(OptimizationFunction):
    """
    Mishra 09 function (3D fixed).

    Mathematical definition:
        a = 2*x1^3 + 5*x1*x2 + 4*x3 - 2*x1^2*x3 - 18
        b = x1 + x2^3 + x1*x2^2 + x1*x3^2 - 22
        c = 8*x1^2 + 2*x2*x3 + 2*x2^2 + 3*x3^3 - 52
        f(x) = (a*b^2*c + a*b*c^2 + b^2 + (x1+x2-x3)^2)^2

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal
    Domain: [-10, 10]^3
    Global minimum: f(1, 2, 3) = 0
    """

    def __init__(
        self,
        dimension: int = 3,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 3:
            raise ValueError("Mishra09 function is Non-Scalable and requires dimension=3")

        default_bounds = Bounds(
            low=np.full(3, -10.0),
            high=np.full(3, 10.0),
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
        """Compute Mishra 09 function."""
        x = self._validate_input(x)
        x1, x2, x3 = x[0], x[1], x[2]
        a = 2.0 * x1**3 + 5.0 * x1 * x2 + 4.0 * x3 - 2.0 * x1**2 * x3 - 18.0
        b = x1 + x2**3 + x1 * x2**2 + x1 * x3**2 - 22.0
        c = 8.0 * x1**2 + 2.0 * x2 * x3 + 2.0 * x2**2 + 3.0 * x3**3 - 52.0
        return float((a * b**2 * c + a * b * c**2 + b**2 + (x1 + x2 - x3) ** 2) ** 2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Mishra 09 function for batch."""
        X = self._validate_batch_input(X)
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        a = 2.0 * x1**3 + 5.0 * x1 * x2 + 4.0 * x3 - 2.0 * x1**2 * x3 - 18.0
        b = x1 + x2**3 + x1 * x2**2 + x1 * x3**2 - 22.0
        c = 8.0 * x1**2 + 2.0 * x2 * x3 + 2.0 * x2**2 + 3.0 * x3**3 - 52.0
        return (a * b**2 * c + a * b * c**2 + b**2 + (x1 + x2 - x3) ** 2) ** 2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum at (1, 2, 3) with value 0."""
        return np.array([1.0, 2.0, 3.0]), 0.0


@register("Mishra10")
@register("mishra10")
class Mishra10Function(OptimizationFunction):
    """
    Mishra 10 function (2D fixed).

    Mathematical definition:
        f(x) = (floor(x1) XOR floor(x2))^2

    Uses integer XOR on floored values.

    Properties: Discontinuous, Non-Differentiable, Non-Separable, Non-Scalable, Multimodal
    Domain: [-10, 10]^2
    Global minimum: f(2, 2) = 0 (any point where floors are equal)
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Mishra10 function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.full(2, -10.0),
            high=np.full(2, 10.0),
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
        """Compute Mishra 10 function."""
        x = self._validate_input(x)
        f1 = int(np.floor(x[0]))
        f2 = int(np.floor(x[1]))
        return float((f1 ^ f2) ** 2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Mishra 10 function for batch."""
        X = self._validate_batch_input(X)
        results = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            f1 = int(np.floor(X[i, 0]))
            f2 = int(np.floor(X[i, 1]))
            results[i] = (f1 ^ f2) ** 2
        return results

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum at (2, 2) with value 0."""
        return np.array([2.0, 2.0]), 0.0


@register("Mishra11")
@register("mishra11")
class Mishra11Function(OptimizationFunction):
    """
    Mishra 11 function (scalable).

    Mathematical definition:
        f(x) = ((1/n)*sum(|x_i|) - (prod(|x_i|))^(1/n))^2

    Properties: Continuous, Non-Differentiable, Non-Separable, Scalable, Multimodal
    Domain: [-10, 10]^D
    Global minimum: f(0, ..., 0) = 0
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
        """Compute Mishra 11 function."""
        x = self._validate_input(x)
        n = self.dimension
        abs_x = np.abs(x)
        mean_abs = np.sum(abs_x) / n
        geom_mean = np.prod(abs_x) ** (1.0 / n)
        return float((mean_abs - geom_mean) ** 2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Mishra 11 function for batch."""
        X = self._validate_batch_input(X)
        n = self.dimension
        abs_X = np.abs(X)
        mean_abs = np.sum(abs_X, axis=1) / n
        geom_mean = np.prod(abs_X, axis=1) ** (1.0 / n)
        return (mean_abs - geom_mean) ** 2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum at (0, ..., 0) with value 0."""
        return np.zeros(self.dimension), 0.0
