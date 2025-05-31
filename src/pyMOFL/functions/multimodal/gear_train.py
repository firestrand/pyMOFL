"""
Gear Train function implementation.

This module implements the Gear Train function, a discrete optimization problem where
the goal is to choose the number of teeth on a compound gear train to achieve a target
speed-reduction ratio. This engineering problem was first published by Sandgren (1990).

References:
    .. [1] Sandgren, E. (1990). "Nonlinear Integer and Discrete Programming
           in Mechanical Design Optimization." *Journal of Mechanical Design*,
           112(2), 223-229.
"""

import numpy as np
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction


class GearTrainFunction(OptimizationFunction):
    """
    Gear Train function (SPSO ID-18).
    
    This discrete optimization problem aims to find the number of teeth on four gears
    in a compound gear train to match a target ratio of approximately 1:6.931 as
    closely as possible.
    
    Mathematical definition:
    f(z₁, z₂, z₃, z₄) = | (z₁·z₂)/(z₃·z₄) - Rₜₐᵣgₑₜ |
    
    where Rₜₐᵣgₑₜ = 1/6.931 ≈ 0.144279, and 12 ≤ zᵢ ≤ 60 (integers).
    
    Global minimum: f = 1.643428e-6 at (z₁, z₂, z₃, z₄) = (16, 19, 43, 49)
    Note: Multiple symmetry-equivalent designs achieve the same error.
    
    Parameters
    ----------
    initialization_bounds : Bounds, optional
        Bounds for initialization. If None, defaults to [12, 60]^4.
    operational_bounds : Bounds, optional
        Bounds for operation. If None, defaults to [12, 60]^4.
    
    References
    ----------
    .. [1] Sandgren, E. (1990). "Nonlinear Integer and Discrete Programming
           in Mechanical Design Optimization." *Journal of Mechanical Design*,
           112(2), 223-229.
    """
    
    _LOW = 12
    _HIGH = 60
    _R_TARGET = 1.0 / 6.931

    def __init__(self,
                 initialization_bounds: Bounds = None,
                 operational_bounds: Bounds = None):
        dimension = 4
        default_init_bounds = Bounds(
            low=np.full(dimension, self._LOW),
            high=np.full(dimension, self._HIGH),
            mode=BoundModeEnum.INITIALIZATION,
            qtype=QuantizationTypeEnum.INTEGER
        )
        default_oper_bounds = Bounds(
            low=np.full(dimension, self._LOW),
            high=np.full(dimension, self._HIGH),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.INTEGER
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_init_bounds,
            operational_bounds=operational_bounds or default_oper_bounds
        )
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Gear Train function at point x.
        
        Parameters
        ----------
        x : np.ndarray
            Input vector of shape (4,). Should be integer-valued.
        Returns
        -------
        float
            The absolute deviation from the target ratio.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        # Assume input is already integer and within bounds (enforced by decorator)
        ratio = (x[0] * x[1]) / (x[2] * x[3])
        result = np.abs(ratio - self._R_TARGET)
        return float(result)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Gear Train function for a batch of points.
        
        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_points, 4). Should be integer-valued.
        Returns
        -------
        np.ndarray
            The absolute deviation from the target ratio for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        ratio = (X[:, 0] * X[:, 1]) / (X[:, 2] * X[:, 3])
        result = np.abs(ratio - self._R_TARGET)
        return result

    @staticmethod
    def get_global_minimum() -> tuple:
        """
        Get the global minimum of the function.

        Returns
        -------
        tuple
            (global_min_point, global_min_value)
        """
        global_min_point = np.array([16, 19, 43, 49])
        global_min_value = 1.643428e-6
        return global_min_point, global_min_value 