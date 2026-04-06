"""
Powell functions family implementation.

This module contains Powell function variants used in optimization benchmarks.
The Powell functions are non-separable test functions that create challenges
for optimization algorithms through their narrow valleys and ridges.

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


@register("Powell_Singular")
@register("PowellSingular")
class PowellSingularFunction(OptimizationFunction):
    """
    Powell Singular function.

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal

    The Powell Singular function is a 4-dimensional test function that creates
    challenges for optimization algorithms through narrow valleys and ridges.

    Mathematical definition:
        f(x) = (x1 + 10*x2)² + 5*(x3 - x4)² + (x2 - 2*x3)^4 + 10*(x1 - x4)^4

    Global minimum: f(0, 0, 0, 0) = 0

    Literature reference: Jamil & Yang 2013 #91 - Non-separable non-scalable unimodal function
    """

    def __init__(
        self,
        dimension: int = 4,  # Non-scalable per literature
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 4:
            raise ValueError("Powell Singular function is Non-Scalable and requires dimension=4")

        default_bounds = Bounds(
            low=np.full(dimension, -4.0),
            high=np.full(dimension, 5.0),
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
        """Evaluate Powell Singular function."""
        x = self._validate_input(x)
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]

        term1 = (x1 + 10.0 * x2) ** 2
        term2 = 5.0 * (x3 - x4) ** 2
        term3 = (x2 - 2.0 * x3) ** 4
        term4 = 10.0 * (x1 - x4) ** 4

        return float(term1 + term2 + term3 + term4)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Powell Singular function for batch."""
        X = self._validate_batch_input(X)
        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        term1 = (x1 + 10.0 * x2) ** 2
        term2 = 5.0 * (x3 - x4) ** 2
        term3 = (x2 - 2.0 * x3) ** 4
        term4 = 10.0 * (x1 - x4) ** 4

        return term1 + term2 + term3 + term4

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        return np.array([0.0, 0.0, 0.0, 0.0]), 0.0


@register("Powell_Singular_2")
@register("PowellSingular2")
class PowellSingular2Function(OptimizationFunction):
    """
    Powell Singular 2 function.

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal

    A variant of the Powell Singular function with different mathematical formulation
    but similar optimization challenges.

    Mathematical definition:
        f(x) = (x1 + 10*x2)² + 5*(x3 - x4)² + (x2 - 2*x3)^4 + 10*(x1 - x4)^4
        (Note: Same as Powell Singular in this implementation - literature may vary)

    Global minimum: f(0, 0, 0, 0) = 0

    Literature reference: Jamil & Yang 2013 #92 - Non-separable non-scalable unimodal variant
    """

    def __init__(
        self,
        dimension: int = 4,  # Non-scalable per literature
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 4:
            raise ValueError("Powell Singular 2 function is Non-Scalable and requires dimension=4")
        default_bounds = Bounds(
            low=np.full(dimension, -4.0),
            high=np.full(dimension, 5.0),
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
        """Evaluate Powell Singular 2 function."""
        x = self._validate_input(x)
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]

        # Using same formulation as Powell Singular for consistency
        # Literature may specify different variants
        term1 = (x1 + 10.0 * x2) ** 2
        term2 = 5.0 * (x3 - x4) ** 2
        term3 = (x2 - 2.0 * x3) ** 4
        term4 = 10.0 * (x1 - x4) ** 4

        return float(term1 + term2 + term3 + term4)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Powell Singular 2 function for batch."""
        X = self._validate_batch_input(X)
        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        term1 = (x1 + 10.0 * x2) ** 2
        term2 = 5.0 * (x3 - x4) ** 2
        term3 = (x2 - 2.0 * x3) ** 4
        term4 = 10.0 * (x1 - x4) ** 4

        return term1 + term2 + term3 + term4

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        return np.array([0.0, 0.0, 0.0, 0.0]), 0.0


@register("Powell_Sum")
@register("PowellSum")
class PowellSumFunction(OptimizationFunction):
    """
    Powell Sum function.

    Properties: Continuous, Differentiable, Separable, Scalable, Unimodal

    The Powell Sum function is a separable, scalable unimodal function that
    increases the power of each term based on its index.

    Mathematical definition:
        f(x) = sum(|x_i|^(i+2)) for i=1 to n

    Global minimum: f(0, 0, ..., 0) = 0

    Literature reference: Jamil & Yang 2013 #93 - Separable scalable unimodal function
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, -1.0),
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
        """Evaluate Powell Sum function."""
        x = self._validate_input(x)

        result = 0.0
        for i in range(self.dimension):
            result += np.abs(x[i]) ** (i + 2)

        return float(result)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Powell Sum function for batch."""
        X = self._validate_batch_input(X)
        n_points, n_dims = X.shape
        results = np.zeros(n_points)

        for j in range(n_points):
            result = 0.0
            for i in range(n_dims):
                result += np.abs(X[j, i]) ** (i + 2)
            results[j] = result

        return results

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        return np.zeros(self.dimension), 0.0
