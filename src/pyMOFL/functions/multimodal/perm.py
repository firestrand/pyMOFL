"""
Perm (0,d,β) function implementation.

This module implements the Perm (0,d,β) function, a multimodal optimization problem
featuring a nested summation that creates a ridge-valley structure. This function
is part of the SPSO benchmark suite as function ID 20.

References
----------
.. [1] Surjanovic, S., & Bingham, D. (2013). *Virtual Library of Simulation Experiments: 
       Test Functions and Datasets*. Simon Fraser University.
       https://www.sfu.ca/~ssurjano/permdb.html
"""

import numpy as np
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction

class PermFunction(OptimizationFunction):
    """
    Perm (0,d,β) function with d=5, β=0.5 (SPSO ID-20).

    This multimodal function features a nested summation and a pronounced ridge-valley 
    structure. Although SPSO classifies this as discrete (the global optimum is at integer 
    coordinates), the analytic form is continuous. Following SPSO convention, this 
    implementation rounds particle positions to the nearest integer before evaluation.

    The function is defined as:
        f(x) = sum_{i=1}^{5} { sum_{j=1}^{5} [(j + β)·(x_j/j)^i - 1] }^2

    Global minimum: f = 0 at x = (1, 2, 3, 4, 5)

    Parameters
    ----------
    beta : float, optional
        The β parameter. Defaults to 0.5 for SPSO.
    initialization_bounds : Bounds, optional
        Bounds for initialization. If None, defaults to [-5, 5]^5.
    operational_bounds : Bounds, optional
        Bounds for operation. If None, defaults to [-5, 5]^5.

    References
    ----------
    .. [1] Surjanovic, S., & Bingham, D. (2013). *Virtual Library of 
           Simulation Experiments: Test Functions and Datasets*.
           Simon Fraser University.
           https://www.sfu.ca/~ssurjano/permdb.html
    """
    def __init__(self, beta: float = 0.5,
                 initialization_bounds: Bounds = None,
                 operational_bounds: Bounds = None):
        dimension = 5
        default_init_bounds = Bounds(
            low=np.full(dimension, -5.0),
            high=np.full(dimension, 5.0),
            mode=BoundModeEnum.INITIALIZATION,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        default_oper_bounds = Bounds(
            low=np.full(dimension, -5.0),
            high=np.full(dimension, 5.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_init_bounds,
            operational_bounds=operational_bounds or default_oper_bounds
        )
        self.beta = beta

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Perm function at a single point.

        Parameters
        ----------
        x : np.ndarray
            Input vector of shape (5,).

        Returns
        -------
        float
            Function value at x.
        """
        x = self._validate_input(x)
        z = np.rint(x).astype(float)
        result = 0.0
        for i in range(1, self.dimension + 1):
            inner_sum = 0.0
            for j in range(1, self.dimension + 1):
                term = (j + self.beta) * (np.power(z[j-1] / j, i) - 1)
                inner_sum += term
            result += inner_sum ** 2
        return float(result)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch evaluation of the Perm function.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_points, 5).

        Returns
        -------
        np.ndarray
            Function values of shape (n_points,).
        """
        X = self._validate_batch_input(X)
        Z = np.rint(X).astype(float)
        result = np.zeros(len(Z))
        for idx, z in enumerate(Z):
            result_val = 0.0
            for i in range(1, self.dimension + 1):
                inner_sum = 0.0
                for j in range(1, self.dimension + 1):
                    term = (j + self.beta) * (np.power(z[j-1] / j, i) - 1)
                    inner_sum += term
                result_val += inner_sum ** 2
            result[idx] = result_val
        return result

    @property
    def bounds(self) -> np.ndarray:
        """
        Legacy-compatible property for test and composite compatibility.
        Returns the operational bounds as a (dimension, 2) array [[low, high], ...].
        Prefer using .operational_bounds for new code.
        """
        return np.stack([self.operational_bounds.low, self.operational_bounds.high], axis=1)

    @staticmethod
    def get_global_minimum(dimension: int = 5) -> tuple:
        """
        Get the global minimum of the function.

        Parameters
        ----------
        dimension : int, optional
            The dimension of the function. Should be 5 for the standard Perm function.

        Returns
        -------
        tuple
            (global_min_point, global_min_value)
        """
        if dimension != 5:
            raise ValueError("Perm function is only defined for 5 dimensions in this implementation")
        global_min_point = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        global_min_value = 0.0
        return global_min_point, global_min_value 