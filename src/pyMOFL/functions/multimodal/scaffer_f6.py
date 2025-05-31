"""
Schaffer's F6 function implementation.

This function is a challenging multimodal test function with a single global minimum
surrounded by many local minima. The function is symmetric, and the global minimum
is located at the origin.

References
----------
.. [1] Schaffer, J.D. (1985). "Multiple Objective Optimization with Vector Evaluated Genetic
       Algorithms". In Proceedings of the 1st International Conference on Genetic Algorithms. 
       L. Erlbaum Associates Inc., pp. 93-100.
.. [2] Molga, M., & Smutnicki, C. (2005). "Test functions for optimization needs".
"""

import numpy as np
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction

class SchafferF6Function(OptimizationFunction):
    """
    Schaffer's F6 function.

    The function is defined as:
        f(x) = 0.5 + (sin^2(sqrt(x1^2 + x2^2)) - 0.5) / (1 + 0.001 * (x1^2 + x2^2))^2

    Global minimum: f(0, 0) = 0

    Parameters
    ----------
    dimension : int, optional
        The dimensionality of the function. Defaults to 2.
        Warning: This function is typically defined for 2 dimensions, though it can be extended.
    initialization_bounds : Bounds, optional
        Bounds for initialization. If None, defaults to [-100, 100]^d.
    operational_bounds : Bounds, optional
        Bounds for operation. If None, defaults to [-100, 100]^d.

    References
    ----------
    .. [1] Schaffer, J.D. (1985). "Multiple Objective Optimization with Vector Evaluated Genetic
           Algorithms". In Proceedings of the 1st International Conference on Genetic Algorithms. 
           L. Erlbaum Associates Inc., pp. 93-100.
    .. [2] Molga, M., & Smutnicki, C. (2005). "Test functions for optimization needs".
    """
    def __init__(self, dimension: int = 2,
                 initialization_bounds: Bounds = None,
                 operational_bounds: Bounds = None):
        default_init_bounds = Bounds(
            low=np.full(dimension, -100.0),
            high=np.full(dimension, 100.0),
            mode=BoundModeEnum.INITIALIZATION,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        default_oper_bounds = Bounds(
            low=np.full(dimension, -100.0),
            high=np.full(dimension, 100.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_init_bounds,
            operational_bounds=operational_bounds or default_oper_bounds
        )

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Schaffer's F6 function at a single point.

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
        sum_squares = np.sum(x**2)
        numerator = np.sin(np.sqrt(sum_squares))**2 - 0.5
        denominator = (1 + 0.001 * sum_squares)**2
        return float(0.5 + numerator / denominator)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch evaluation of the Schaffer's F6 function.

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
        sum_squares = np.sum(X**2, axis=1)
        numerator = np.sin(np.sqrt(sum_squares))**2 - 0.5
        denominator = (1 + 0.001 * sum_squares)**2
        return 0.5 + numerator / denominator

    @property
    def bounds(self) -> np.ndarray:
        """
        Legacy-compatible property for test and composite compatibility.
        Returns the operational bounds as a (dimension, 2) array [[low, high], ...].
        Prefer using .operational_bounds for new code.
        """
        return np.stack([self.operational_bounds.low, self.operational_bounds.high], axis=1)

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
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
        global_min_point = np.zeros(dimension)
        global_min_value = 0.0
        return global_min_point, global_min_value

# For backward compatibility and simpler importing
ScafferF6Function = SchafferF6Function  # Common alternative spelling 