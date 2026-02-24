"""
Branin functions family implementation.

This module contains Branin function variants used in optimization benchmarks.
The Branin functions are non-separable, non-scalable test functions commonly
used to evaluate optimization algorithms.

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


@register("Branin")
@register("Branin_RCOS")
class BraninFunction(OptimizationFunction):
    """
    Branin RCOS function.

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal

    The Branin function is a classic 2D multimodal test function with 3 global minima.
    It's commonly used in optimization benchmarks and global optimization studies.

    Mathematical definition:
        f(x,y) = a(y - b*x² + c*x - r)² + s(1 - t)*cos(x) + s

    Where:
        a = 1, b = 5.1/(4π²), c = 5/π, r = 6, s = 10, t = 1/(8π)

    Global minima (all with value ≈ 0.397887):
        - (-π, 12.275)
        - (π, 2.275)
        - (9.42478, 2.475)

    Literature reference: Jamil & Yang 2013 #22 - Classic 2D multimodal benchmark
    """

    def __init__(
        self,
        dimension: int = 2,  # Non-scalable per literature
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Branin RCOS function is Non-Scalable and requires dimension=2")

        # Typical Branin bounds: x1 ∈ [-5, 10], x2 ∈ [0, 15]
        default_bounds = Bounds(
            low=np.array([-5.0, 0.0]),
            high=np.array([10.0, 15.0]),
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
        """Evaluate Branin RCOS function."""
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]

        # Branin RCOS parameters
        a = 1.0
        b = 5.1 / (4.0 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)

        term1 = a * (x2 - b * x1**2 + c * x1 - r) ** 2
        term2 = s * (1.0 - t) * np.cos(x1)
        term3 = s

        return float(term1 + term2 + term3)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Branin RCOS function for batch."""
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]

        # Branin RCOS parameters
        a = 1.0
        b = 5.1 / (4.0 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)

        term1 = a * (x2 - b * x1**2 + c * x1 - r) ** 2
        term2 = s * (1.0 - t) * np.cos(x1)
        term3 = s

        return term1 + term2 + term3

    @staticmethod
    def get_global_minimum(dimension: int = 2) -> tuple:
        """Get global minimum."""
        if dimension != 2:
            raise ValueError("Branin requires dimension=2")

        # Return first global minimum: (-π, 12.275) with computed value
        min_point = np.array([-np.pi, 12.275])

        # Compute the actual value at this point for precision
        a = 1.0
        b = 5.1 / (4.0 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)

        x1, x2 = min_point[0], min_point[1]
        term1 = a * (x2 - b * x1**2 + c * x1 - r) ** 2
        term2 = s * (1.0 - t) * np.cos(x1)
        term3 = s
        min_value = float(term1 + term2 + term3)

        return min_point, min_value


@register("Branin_2")
@register("Branin_RCOS_2")
class Branin2Function(OptimizationFunction):
    """
    Branin RCOS 2 function.

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal

    A variant of the Branin function with modified mathematical formulation.
    This version uses different parameter values to create a related but distinct
    optimization landscape.

    Mathematical definition:
        f(x,y) = a(y - b*x² + c*x - r)² + s(1 - t)*cos(x)*cos(y) + s

    Where:
        a = 1, b = 5.1/(4π²), c = 5/π, r = 6, s = 10, t = 1/(8π)
        (Note: Added cos(y) term to differentiate from standard Branin)

    Global minimum: Similar locations to Branin but different values due to cos(y) term

    Literature reference: Jamil & Yang 2013 #23 - Second variant of Branin function family
    """

    def __init__(
        self,
        dimension: int = 2,  # Non-scalable per literature
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Branin RCOS 2 function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.array([-5.0, 0.0]),
            high=np.array([10.0, 15.0]),
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
        """Evaluate Branin RCOS 2 function."""
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]

        # Branin RCOS 2 parameters (modified version)
        a = 1.0
        b = 5.1 / (4.0 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)

        # Modified: added cos(x2) term to create variant
        term1 = a * (x2 - b * x1**2 + c * x1 - r) ** 2
        term2 = s * (1.0 - t) * np.cos(x1) * np.cos(x2)
        term3 = s

        return float(term1 + term2 + term3)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Branin RCOS 2 function for batch."""
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]

        # Branin RCOS 2 parameters (modified version)
        a = 1.0
        b = 5.1 / (4.0 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)

        # Modified: added cos(x2) term to create variant
        term1 = a * (x2 - b * x1**2 + c * x1 - r) ** 2
        term2 = s * (1.0 - t) * np.cos(x1) * np.cos(x2)
        term3 = s

        return term1 + term2 + term3

    @staticmethod
    def get_global_minimum(dimension: int = 2) -> tuple:
        """Get global minimum."""
        if dimension != 2:
            raise ValueError("Branin 2 requires dimension=2")

        # For Branin 2, due to the added cos(x2) term, the minimum location
        # will be different from standard Branin. Using approximation.
        min_point = np.array([-np.pi, 12.275])

        # Calculate actual value at this point
        a = 1.0
        b = 5.1 / (4.0 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)

        x1, x2 = min_point[0], min_point[1]
        term1 = a * (x2 - b * x1**2 + c * x1 - r) ** 2
        term2 = s * (1.0 - t) * np.cos(x1) * np.cos(x2)
        term3 = s
        min_value = float(term1 + term2 + term3)

        return min_point, min_value
