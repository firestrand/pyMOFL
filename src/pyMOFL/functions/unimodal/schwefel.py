"""
Schwefel function implementations.

This module provides implementations of various Schwefel problems used as benchmark functions.
They are named according to their original numbering in Schwefel's work.

References
----------
.. [1] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
       "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
       optimization". Nanyang Technological University, Singapore, Tech. Rep.
.. [2] Schwefel, H.-P. (1981). "Numerical optimization of computer models".
       John Wiley & Sons, Inc.
"""

import numpy as np
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.base import register_function

@register_function("Schwefel1.2")
class SchwefelFunction12(OptimizationFunction):
    """
    Schwefel's Problem 1.2 function.

    The function is defined as:
        f(x) = sum_{i=1}^D (sum_{j=1}^i x_j)^2
    It is unimodal, non-separable, continuous, convex, and differentiable.

    Global minimum: f(0, ..., 0) = 0

    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    initialization_bounds : Bounds, optional
        Bounds for random initialization. If None, defaults to [-100, 100]^d.
    operational_bounds : Bounds, optional
        Bounds for domain enforcement. If None, defaults to [-100, 100]^d.

    References
    ----------
    .. [1] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
           "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
           optimization". Nanyang Technological University, Singapore, Tech. Rep.
    .. [2] Schwefel, H.-P. (1981). "Numerical optimization of computer models".
           John Wiley & Sons, Inc.
    """
    def __init__(self, dimension: int,
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
        Evaluate the Schwefel 1.2 function at a single point.

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
        prefix = np.cumsum(x, dtype=np.float64)
        return float(np.dot(prefix, prefix))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch evaluation of the Schwefel 1.2 function.

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
        prefix = np.cumsum(X, axis=1)
        return np.sum(prefix ** 2, axis=1)

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        """
        Get the global minimum point and value for the Schwefel 1.2 function.

        Parameters
        ----------
        dimension : int
            The dimensionality of the function.

        Returns
        -------
        tuple
            (global_min_point, global_min_value)
        """
        global_min_point = np.zeros(dimension)
        global_min_value = 0.0
        return global_min_point, global_min_value

@register_function("Schwefel2.6")
class SchwefelFunction26(OptimizationFunction):
    """
    Schwefel's Problem 2.6 base function.

    The function is defined as:
        f(x) = max_i |A_i · x - B_i|
    where A is a (dimension, dimension) matrix and B is a (dimension,) vector.

    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    A : np.ndarray
        Matrix of shape (dimension, dimension).
    B : np.ndarray
        Vector of shape (dimension,).
    optimum_point : np.ndarray, optional
        The known optimum point, if available.
    initialization_bounds : Bounds, optional
        Bounds for random initialization. If None, defaults to [-100, 100]^d.
    operational_bounds : Bounds, optional
        Bounds for domain enforcement. If None, defaults to [-100, 100]^d.
    """
    def __init__(self, dimension: int,
                 A: np.ndarray,
                 B: np.ndarray,
                 optimum_point: np.ndarray = None,
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
        if A is None or B is None:
            raise ValueError("A matrix and B vector are required for SchwefelFunction26.")
        if A.shape != (dimension, dimension):
            raise ValueError(f"A matrix shape {A.shape} does not match dimension {dimension}")
        if B.shape != (dimension,):
            raise ValueError(f"B vector shape {B.shape} does not match dimension {dimension}")
        self.A = np.array(A)
        self.B = np.array(B)
        self.optimum_point = np.array(optimum_point) if optimum_point is not None else None

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Schwefel 2.6 function at a single point.

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
        Ax = np.dot(self.A, x)
        diff = np.abs(Ax - self.B)
        return float(np.max(diff))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch evaluation of the Schwefel 2.6 function.

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
        Ax = np.dot(X, self.A.T)
        diff = np.abs(Ax - self.B[np.newaxis, :])
        return np.max(diff, axis=1)

    def get_global_minimum(self):
        """
        Get the global minimum point and value for the Schwefel 2.6 function.

        Returns
        -------
        tuple
            (optimum_point, 0.0) if optimum_point is set, else raises NotImplementedError.
        """
        if self.optimum_point is not None:
            return self.optimum_point, 0.0
        raise NotImplementedError("Optimum point not set for this instance.")

# Aliases for backward compatibility
SchwefelProblem12Function = SchwefelFunction12
SchwefelProblem26Function = SchwefelFunction26