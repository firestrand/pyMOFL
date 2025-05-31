"""
Tripod function implementation.

The Tripod function is a 2D benchmark function with discontinuities along 
the coordinate axes, creating steep ridges and flat valleys.

References
----------
.. [1] Molga, M., & Smutnicki, C. (2005). "Test Functions for Optimization Needs."
       Technical report, Technical University of Gdańsk.
.. [2] Clerc, M. (2012). "Standard PSO 2007/2011 Benchmark Documentation."
       Technical report, see also Zambrano-Bigiarini et al. (2013).
"""

import numpy as np
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction

class TripodFunction(OptimizationFunction):
    """
    Tripod benchmark function.

    The function is defined as:
        f(x1,x2) = p(x2)(1+p(x1))
                 + |x1 + 50·p(x2)(1-2·p(x1))|
                 + |x2 + 50(1-2·p(x2))|
    where p(z) = 1 if z ≥ 0, and 0 if z < 0.

    Global minimum: f(0, -50) = 0

    Parameters
    ----------
    initialization_bounds : Bounds, optional
        Bounds for initialization. If None, defaults to [-100, 100]^2.
    operational_bounds : Bounds, optional
        Bounds for operation. If None, defaults to [-100, 100]^2.

    References
    ----------
    .. [1] Molga, M., & Smutnicki, C. (2005). "Test Functions for Optimization Needs."
           Technical report, Technical University of Gdańsk.
    .. [2] Clerc, M. (2012). "Standard PSO 2007/2011 Benchmark Documentation."
           Technical report, see also Zambrano-Bigiarini et al. (2013).
    """
    def __init__(self,
                 initialization_bounds: Bounds = None,
                 operational_bounds: Bounds = None):
        dimension = 2
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
        Evaluate the Tripod function at a single point.

        Parameters
        ----------
        x : np.ndarray
            Input vector of shape (2,).

        Returns
        -------
        float
            Function value at x.
        """
        x = self._validate_input(x)
        p = np.array(x >= 0, dtype=float)
        term1 = p[1] * (1 + p[0])
        term2 = np.abs(x[0] + 50 * p[1] * (1 - 2 * p[0]))
        term3 = np.abs(x[1] + 50 * (1 - 2 * p[1]))
        return float(term1 + term2 + term3)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch evaluation of the Tripod function.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_points, 2).

        Returns
        -------
        np.ndarray
            Function values of shape (n_points,).
        """
        X = self._validate_batch_input(X)
        p = np.array(X >= 0, dtype=float)
        term1 = p[:, 1] * (1 + p[:, 0])
        term2 = np.abs(X[:, 0] + 50 * p[:, 1] * (1 - 2 * p[:, 0]))
        term3 = np.abs(X[:, 1] + 50 * (1 - 2 * p[:, 1]))
        return term1 + term2 + term3

    @property
    def bounds(self) -> np.ndarray:
        """
        Legacy-compatible property for test and composite compatibility.
        Returns the operational bounds as a (dimension, 2) array [[low, high], ...].
        Prefer using .operational_bounds for new code.
        """
        return np.stack([self.operational_bounds.low, self.operational_bounds.high], axis=1)

    @staticmethod
    def get_global_minimum(dimension: int = 2) -> tuple:
        """
        Get the global minimum of the function.

        Parameters
        ----------
        dimension : int, optional
            The dimension of the function. Defaults to 2, as Tripod is always 2D.

        Returns
        -------
        tuple
            (global_min_point, global_min_value)
        """
        if dimension != 2:
            raise ValueError("Tripod function is only defined for 2 dimensions")
        global_min_point = np.array([0.0, -50.0])
        global_min_value = 0.0
        return global_min_point, global_min_value 