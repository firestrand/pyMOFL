"""
Alpine functions family implementation.

This module contains Alpine function variants used in optimization benchmarks.
The Alpine functions feature different characteristics regarding differentiability
and separability, making them useful for testing different algorithm behaviors.

References
----------
.. [1] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions for global
       optimization problems". International Journal of Mathematical Modelling and Numerical
       Optimisation, 4(2), 150-194. arXiv:1308.4008
       Local documentation: docs/literature_schwefel/jamil_yang_2013_literature_survey.md
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Alpine_1")
@register("Alpine1")
class Alpine1Function(OptimizationFunction):
    """
    Alpine 1 function.

    Properties: Continuous, Non-Differentiable, Separable, Non-Scalable, Multimodal

    The Alpine 1 function is a non-differentiable, separable function that creates
    challenges for optimization algorithms due to its absolute value terms.

    Mathematical definition:
        f(x) = Σ|x_i * sin(x_i) + 0.1 * x_i| for i=1 to n

    Global minimum: f(0, 0) = 0 (for 2D version)

    Literature reference: Jamil & Yang 2013 #6 - Non-differentiable separable multimodal function
    """

    def __init__(
        self,
        dimension: int = 2,  # Non-scalable per literature
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Alpine 1 function is Non-Scalable and requires dimension=2")
        default_bounds = Bounds(
            low=np.array([-10.0, -10.0]),
            high=np.array([10.0, 10.0]),
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
        """Evaluate Alpine 1 function."""
        x = self._validate_input(x)

        result = 0.0
        for i in range(self.dimension):
            result += abs(x[i] * np.sin(x[i]) + 0.1 * x[i])

        return float(result)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Alpine 1 function for batch."""
        X = self._validate_batch_input(X)
        n_points, n_dims = X.shape
        results = np.zeros(n_points)

        for j in range(n_points):
            result = 0.0
            for i in range(n_dims):
                result += abs(X[j, i] * np.sin(X[j, i]) + 0.1 * X[j, i])
            results[j] = result

        return results

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        return np.zeros(self.dimension), 0.0


@register("Alpine_2")
@register("Alpine2")
class Alpine2Function(OptimizationFunction):
    """
    Alpine 2 function.

    Properties: Continuous, Differentiable, Separable, Scalable, Multimodal

    The Alpine 2 function is a differentiable, separable, scalable function that
    uses a product formulation making it challenging for optimization algorithms.

    Mathematical definition:
        f(x) = ∏(√x_i * sin(x_i)) for i=1 to n

    Global minimum: Approximately at x_i ≈ 7.917 for all i
    Global minimum value: Approximately 2.808^n

    Literature reference: Jamil & Yang 2013 #7 - Differentiable separable scalable multimodal function
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
        """Evaluate Alpine 2 function."""
        x = self._validate_input(x)

        result = 1.0
        for i in range(self.dimension):
            if x[i] <= 0:
                return 0.0  # Product becomes 0 if any component is <= 0
            result *= np.sqrt(x[i]) * np.sin(x[i])

        return float(result)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Alpine 2 function for batch."""
        X = self._validate_batch_input(X)
        n_points, n_dims = X.shape
        results = np.zeros(n_points)

        for j in range(n_points):
            result = 1.0
            zero_found = False

            for i in range(n_dims):
                if X[j, i] <= 0:
                    zero_found = True
                    break
                result *= np.sqrt(X[j, i]) * np.sin(X[j, i])

            results[j] = 0.0 if zero_found else result

        return results

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        # Global minimum approximately at x_i ≈ 7.917 for all i
        min_point = np.full(self.dimension, 7.917)

        # Global minimum value approximately 2.808^n
        # More precisely: (√7.917 * sin(7.917))^n ≈ 2.808^n
        component_value = np.sqrt(7.917) * np.sin(7.917)
        min_value = component_value**self.dimension

        return min_point, float(min_value)
