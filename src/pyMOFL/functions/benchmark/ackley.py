"""
Ackley function implementation.

The Ackley function is a widely used multimodal test function for optimization algorithms.
It has a global minimum surrounded by an almost flat outer region with many local minima.

References:
    .. [1] Ackley, D. H. (1987). "A connectionist machine for genetic hillclimbing".
           Kluwer Academic Publishers.
    .. [2] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
           "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
           optimization". Nanyang Technological University, Singapore, Tech. Rep.
    .. [3] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions for global
           optimization problems". International Journal of Mathematical Modelling and Numerical
           Optimisation, 4(2), 150-194. arXiv:1308.4008
           Local documentation: docs/literature_ackley/jamil_yang_2013_literature_survey.md
    .. [4] Kumar, K.E.S., et al. (2024). "Benchmarking of GPU-optimized Quantum-Inspired
           Evolutionary Optimization Algorithm using Functional Analysis". arXiv:2412.08992
           Local documentation: docs/literature_ackley/kumar_2024_gpu_benchmarking.md
    .. [5] Demo, N., Tezzele, M., & Rozza, G. (2020). "A supervised learning approach involving
           active subspaces for an efficient genetic algorithm in high-dimensional optimization
           problems". arXiv:2006.07282
           Local documentation: docs/literature_ackley/demo_2020_active_subspaces.md
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Ackley")
@register("ackley")
class AckleyFunction(OptimizationFunction):
    """
    Ackley function: f(x) = -20·exp(-0.2·sqrt(sum(x_i^2)/D)) - exp(sum(cos(2π·x_i))/D) + 20 + e

    Global minimum: f(0, 0, ..., 0) = 0

    Attributes:
        dimension (int): The dimensionality of the function.
        initialization_bounds (Bounds): Bounds for initialization.
        operational_bounds (Bounds): Bounds for operation.
        a (float): Coefficient for the first exponential term. Default is 20.
        b (float): Coefficient for the squared term. Default is 0.2.
        c (float): Coefficient for the cosine term. Default is 2π.

    References:
        .. [1] Ackley, D. H. (1987). "A connectionist machine for genetic hillclimbing".
               Kluwer Academic Publishers.
        .. [2] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
               "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
               optimization". Nanyang Technological University, Singapore, Tech. Rep.
        .. [3] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions for global
               optimization problems". Contains 4 Ackley variants with detailed mathematical definitions.
               Local documentation: docs/literature_ackley/jamil_yang_2013_literature_survey.md
        .. [4] Kumar, K.E.S., et al. (2024). "Benchmarking of GPU-optimized Quantum-Inspired
               Evolutionary Optimization Algorithm using Functional Analysis". Modern GPU benchmarking.
               Local documentation: docs/literature_ackley/kumar_2024_gpu_benchmarking.md

    Note:
        To add a bias to the function, use the BiasWrapper from the transformations module.
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        a: float = 20.0,
        b: float = 0.2,
        c: float = 2.0 * np.pi,
        **kwargs,
    ):
        """
        Initialize the Ackley function.

        Args:
            dimension (int): The dimensionality of the function.
            input_transforms (List[Transform], optional): Input transforms to apply before computing.
            initialization_bounds (Bounds, optional): Bounds for initialization. Defaults to [-32.768, 32.768] for each dimension.
            operational_bounds (Bounds, optional): Bounds for operation. Defaults to [-32.768, 32.768] for each dimension.
            a (float, optional): Coefficient for the first exponential term. Defaults to 20.
            b (float, optional): Coefficient for the squared term. Defaults to 0.2.
            c (float, optional): Coefficient for the cosine term. Defaults to 2π.

        Note:
            To add a bias to the function, use the BiasWrapper from the transformations module.
        """
        default_bounds = Bounds(
            low=np.full(dimension, -32.768),
            high=np.full(dimension, 32.768),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )
        self.a = a
        self.b = b
        self.c = c
        self._const = self.a + np.e
        self._inv_d = 1.0 / dimension

    def evaluate(self, x: np.ndarray) -> float:
        """Compute the Ackley function value."""
        x = self._validate_input(x)
        s1 = np.dot(x, x)  # Σ x²
        s2 = np.cos(self.c * x).sum()  # Σ cos(2πx)

        term1 = -self.a * np.exp(-self.b * np.sqrt(s1 * self._inv_d))
        term2 = -np.exp(s2 * self._inv_d)
        return float(term1 + term2 + self._const)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Ackley function for batch."""
        X = self._validate_batch_input(X)
        # Σ x²  per row
        s1 = np.einsum("ij,ij->i", X, X, optimize="greedy")  # faster than (X**2).sum(axis=1)

        # Σ cos(2πx)  per row - create a copy of X to avoid modifying the input
        s2 = np.cos(self.c * X).sum(axis=1)  # don't reuse X's buffer to prevent side effects

        term1 = -self.a * np.exp(-self.b * np.sqrt(s1 * self._inv_d))
        term2 = -np.exp(s2 * self._inv_d)
        return term1 + term2 + self._const

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        """
        Get the global minimum of the function.

        Args:
            dimension (int): The dimension of the function.

        Returns:
            tuple: A tuple containing the global minimum point and the function value at that point.
        """
        global_min_point = np.zeros(dimension)
        global_min_value = 0.0
        return global_min_point, global_min_value


class Ackley2Function(OptimizationFunction):
    """
    Ackley 2 function: f(x, y) = -200 * exp(-0.02 * sqrt(x_1^2 + x_2^2))

    A smooth, unimodal function with an exponentially steep basin at the origin.
    Non-scalable — only defined for dimension 2.

    Global minimum: f(0, 0) = -200

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal

    References:
        .. [1] Ackley, D. H. (1987). "A connectionist machine for genetic hillclimbing".
        .. [2] Jamil, M., & Yang, X.S. (2013). #2 in literature survey. arXiv:1308.4008
        .. [3] BenchmarkFcns: https://benchmarkfcns.info/doc/ackleyn2fcn.html
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Ackley 2 function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.full(dimension, -32.0),
            high=np.full(dimension, 32.0),
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
        """Compute Ackley 2: f(x,y) = -200 * exp(-0.02 * sqrt(x^2 + y^2))."""
        x = self._validate_input(x)
        return float(-200.0 * np.exp(-0.02 * np.sqrt(x[0] ** 2 + x[1] ** 2)))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Ackley 2 for a batch of points."""
        X = self._validate_batch_input(X)
        return -200.0 * np.exp(-0.02 * np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2))

    @staticmethod
    def get_global_minimum(dimension: int = 2) -> tuple:
        if dimension != 2:
            raise ValueError("Ackley 2 requires dimension=2")
        return np.zeros(2), -200.0


class Ackley3Function(OptimizationFunction):
    """
    Ackley 3 function: f(x,y) = -200*exp(-0.02*sqrt(x^2+y^2)) + 5*exp(cos(3x)+sin(3y))

    Combines the exponential basin of Ackley 2 with a trigonometric perturbation,
    creating an asymmetric landscape. Non-scalable — only defined for dimension 2.

    Global minimum: f(±0.6826, -0.3608) ≈ -195.6290

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal

    References:
        .. [1] Ackley, D. H. (1987). "A connectionist machine for genetic hillclimbing".
        .. [2] Jamil, M., & Yang, X.S. (2013). #3 in literature survey. arXiv:1308.4008
        .. [3] BenchmarkFcns: https://benchmarkfcns.info/doc/ackleyn3fcn.html
    """

    # Pre-computed global minimum from high-precision numerical optimization
    _GLOBAL_MIN_X = np.array([0.682584587365898, -0.36075325513719])
    _GLOBAL_MIN_VALUE = -195.62902823841935

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Ackley 3 function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.full(dimension, -32.0),
            high=np.full(dimension, 32.0),
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
        """Compute Ackley 3: -200*exp(-0.02*sqrt(x^2+y^2)) + 5*exp(cos(3x)+sin(3y))."""
        x = self._validate_input(x)
        term1 = -200.0 * np.exp(-0.02 * np.sqrt(x[0] ** 2 + x[1] ** 2))
        term2 = 5.0 * np.exp(np.cos(3.0 * x[0]) + np.sin(3.0 * x[1]))
        return float(term1 + term2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Ackley 3 for a batch of points."""
        X = self._validate_batch_input(X)
        term1 = -200.0 * np.exp(-0.02 * np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2))
        term2 = 5.0 * np.exp(np.cos(3.0 * X[:, 0]) + np.sin(3.0 * X[:, 1]))
        return term1 + term2

    @staticmethod
    def get_global_minimum(dimension: int = 2) -> tuple:
        if dimension != 2:
            raise ValueError("Ackley 3 requires dimension=2")
        return Ackley3Function._GLOBAL_MIN_X.copy(), Ackley3Function._GLOBAL_MIN_VALUE


class Ackley4Function(OptimizationFunction):
    """
    Ackley 4 (Modified Ackley) function:
        f(x) = sum_{i=0}^{D-2} [exp(-0.2)*sqrt(x_i^2 + x_{i+1}^2)
                                 + 3*(cos(2*x_i) + sin(2*x_{i+1}))]

    A scalable, multimodal function with variable coupling between adjacent dimensions.

    Global minimum (D=2): f(±1.51, -0.755) ≈ -4.5901

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Multimodal

    References:
        .. [1] Jamil, M., & Yang, X.S. (2013). #4 in literature survey. arXiv:1308.4008
        .. [2] BenchmarkFcns: https://benchmarkfcns.info/doc/ackleyn4fcn.html
    """

    _EXP_NEG_02 = np.exp(-0.2)

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, -35.0),
            high=np.full(dimension, 35.0),
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
        """Compute Modified Ackley function value."""
        x = self._validate_input(x)
        x_i = x[:-1]
        x_ip1 = x[1:]
        terms = self._EXP_NEG_02 * np.sqrt(x_i**2 + x_ip1**2) + 3.0 * (
            np.cos(2.0 * x_i) + np.sin(2.0 * x_ip1)
        )
        return float(terms.sum())

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Modified Ackley function for a batch of points."""
        X = self._validate_batch_input(X)
        X_i = X[:, :-1]
        X_ip1 = X[:, 1:]
        terms = self._EXP_NEG_02 * np.sqrt(X_i**2 + X_ip1**2) + 3.0 * (
            np.cos(2.0 * X_i) + np.sin(2.0 * X_ip1)
        )
        return terms.sum(axis=1)

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        """Get approximate global minimum.

        For D=2, the global minimum is well-characterized.
        For higher dimensions, the minimum is problem-specific and
        the returned point is an approximation.
        """
        if dimension == 2:
            return np.array([-1.51, -0.755]), -4.590100665150724
        # For D>2, return zeros as a placeholder — the true minimum
        # depends on the dimension and has no closed-form solution.
        return np.zeros(dimension), float("nan")
