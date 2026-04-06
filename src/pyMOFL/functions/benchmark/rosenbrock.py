"""
Rosenbrock functions implementation.

This module contains all variants of Rosenbrock functions used in optimization benchmarks.
The Rosenbrock function is a non-convex benchmark function with a narrow, curved valley
which is difficult for many optimization algorithms to navigate.

References
----------
.. [1] Rosenbrock, H.H. (1960). "An automatic method for finding the greatest or least value of a function".
       The Computer Journal, 3(3), 175-184.
.. [2] Kumar, K.E.S., et al. (2024). "Benchmarking of GPU-optimized Quantum-Inspired
       Evolutionary Optimization Algorithm using Functional Analysis". arXiv:2412.08992
       Local documentation: docs/literature_rosenbrock/kumar_2024_gpu_benchmarking.md
.. [3] Adorio, E.P., & Diliman, U.P. (2005). MVF - "Multivariate Test Functions Library in C for
       Unconstrained Global Optimization", 2005.
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Rosenbrock")
@register("rosenbrock")
class RosenbrockFunction(OptimizationFunction):
    """
    Rosenbrock function (unimodal).

    The function is defined as:
        f(x) = sum_{i=1}^{n-1} [100*(x_{i}^2 - x_{i+1})^2 + (x_i - 1)^2]

    Global minimum: f(1, 1, ..., 1) = 0

    Properties: unimodal, non-separable, non-rotational_invariance

    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    initialization_bounds : Bounds, optional
        Bounds for random initialization. If None, defaults to [-30, 30]^d.
    operational_bounds : Bounds, optional
        Bounds for domain enforcement. If None, defaults to [-30, 30]^d.

    References
    ----------
    .. [1] Rosenbrock, H.H. (1960). "An automatic method for finding the greatest or least value of a function".
           The Computer Journal, 3(3), 175-184.
    .. [2] Kumar, K.E.S., et al. (2024). "Benchmarking of GPU-optimized Quantum-Inspired
           Evolutionary Optimization Algorithm using Functional Analysis". arXiv:2412.08992
           Local documentation: docs/literature_rosenbrock/kumar_2024_gpu_benchmarking.md
    .. [3] Adorio, E.P., & Diliman, U.P. (2005). MVF - "Multivariate Test Functions Library in C for
           Unconstrained Global Optimization", 2005.
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.full(dimension, -30.0),
                high=np.full(dimension, 30.0),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, -30.0),
                high=np.full(dimension, 30.0),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
            **kwargs,
        )

    @property
    def function_properties(self):
        """Function properties metadata."""
        return {
            "type": "unimodal",
            "separable": False,
            "rotational_invariance": False,
            "family": "rosenbrock",
        }

    def evaluate(self, x: np.ndarray) -> float:
        """Compute the Rosenbrock function value."""
        x = self._validate_input(x)
        term1 = 100 * (x[:-1] ** 2 - x[1:]) ** 2
        term2 = (x[:-1] - 1) ** 2
        return float(np.sum(term1 + term2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Rosenbrock function for batch."""
        X = self._validate_batch_input(X)
        term1 = 100 * (X[:, :-1] ** 2 - X[:, 1:]) ** 2
        term2 = (X[:, :-1] - 1) ** 2
        return np.sum(term1 + term2, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """
        Get the global minimum of the function.

        Parameters
        ----------
        dimension : int
            The dimension of the function.

        Returns
        -------
        tuple
            (global_min_point, global_min_value)
        """
        global_min_point = np.ones(self.dimension)
        global_min_value = 0.0
        return global_min_point, global_min_value


@register("GriewankOfRosenbrock")
@register("griewankOfRosenbrock")
@register("griewank_of_rosenbrock")
class GriewankOfRosenbrock(OptimizationFunction):
    """
    Griewank of Rosenbrock (F8F2) expanded function (multimodal).

    This function applies Griewank function to pairs of variables transformed
    by the Rosenbrock function. It's commonly used in CEC benchmarks as F8F2.

    The function is defined as:
        F8F2(x) = sum(F8(F2(x_i, x_{i+1}))) for i = 1 to D-1
                + F8(F2(x_D, x_1))

    where:
        F2(x, y) = 100 * (x^2 - y)^2 + (x - 1)^2  (Rosenbrock)
        F8(z) = z^2/4000 - cos(z) + 1              (Griewank)

    Global minimum: f(1, 1, ..., 1) = 0

    Properties: multimodal, non-separable, non-rotational_invariance

    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    initialization_bounds : Bounds, optional
        Bounds for initialization. If None, defaults to [-5, 5]^d.
    operational_bounds : Bounds, optional
        Bounds for operation. If None, defaults to [-5, 5]^d.
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
    ):
        default_init_bounds = Bounds(
            low=np.full(dimension, -5.0),
            high=np.full(dimension, 5.0),
            mode=BoundModeEnum.INITIALIZATION,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        default_oper_bounds = Bounds(
            low=np.full(dimension, -5.0),
            high=np.full(dimension, 5.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_init_bounds,
            operational_bounds=operational_bounds or default_oper_bounds,
        )

    @property
    def function_properties(self):
        """Function properties metadata."""
        return {
            "type": "multimodal",
            "separable": False,
            "rotational_invariance": False,
            "family": "rosenbrock",
        }

    def _rosenbrock(self, x: float, y: float) -> float:
        """Apply Rosenbrock function F2 to a pair of variables."""
        return 100.0 * (x**2 - y) ** 2 + (x - 1.0) ** 2

    def _griewank(self, z: float) -> float:
        """Apply Griewank function F8 to a single value."""
        return z**2 / 4000.0 - np.cos(z) + 1.0

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Griewank of Rosenbrock function at a single point.

        Parameters
        ----------
        x : np.ndarray
            Input vector of shape (dimension,).

        Returns
        -------
        float
            Function value at x.
        """
        x = self._validate_input(x)
        total = 0.0
        # Apply F8(F2) to consecutive pairs
        for i in range(self.dimension - 1):
            rosenbrock_val = self._rosenbrock(x[i], x[i + 1])
            total += self._griewank(rosenbrock_val)

        # Apply F8(F2) to the last and first variables (circular)
        rosenbrock_val = self._rosenbrock(x[-1], x[0])
        total += self._griewank(rosenbrock_val)

        return float(total)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch evaluation of the Griewank of Rosenbrock function.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_points, dimension).

        Returns
        -------
        np.ndarray
            Function values of shape (n_points,).
        """
        X = self._validate_batch_input(X)
        n_points = X.shape[0]

        result = np.zeros(n_points)

        # Apply F8(F2) to consecutive pairs
        for i in range(self.dimension - 1):
            rosenbrock_vals = 100.0 * (X[:, i] ** 2 - X[:, i + 1]) ** 2 + (X[:, i] - 1.0) ** 2
            result += rosenbrock_vals**2 / 4000.0 - np.cos(rosenbrock_vals) + 1.0

        # Apply F8(F2) to the last and first variables (circular)
        rosenbrock_vals = 100.0 * (X[:, -1] ** 2 - X[:, 0]) ** 2 + (X[:, -1] - 1.0) ** 2
        result += rosenbrock_vals**2 / 4000.0 - np.cos(rosenbrock_vals) + 1.0

        return result

    @property
    def bounds(self) -> np.ndarray:
        """
        Legacy-compatible property for test and composite compatibility.
        Returns the operational bounds as a (dimension, 2) array [[low, high], ...].
        Prefer using .operational_bounds for new code.
        """
        assert self.operational_bounds is not None
        return np.stack([self.operational_bounds.low, self.operational_bounds.high], axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """
        Get the global minimum of the function.

        Parameters
        ----------
        dimension : int
            The dimension of the function.

        Returns
        -------
        tuple
            (global_min_point, global_min_value)
        """
        global_min_point = np.ones(self.dimension)
        global_min_value = 0.0
        return global_min_point, global_min_value
