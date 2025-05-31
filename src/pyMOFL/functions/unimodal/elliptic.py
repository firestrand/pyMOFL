"""
High Conditioned Elliptic function implementation.

The Elliptic function is a unimodal benchmark function with high conditioning.
It is continuous, convex, and differentiable.

References
----------
.. [1] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
       "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
       optimization". Nanyang Technological University, Singapore, Tech. Rep.
.. [2] Hansen, N., Müller, S. D., & Koumoutsakos, P. (2003). 
       "Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES)". 
       Evolutionary Computation, 11(1), 1-18.
"""

import numpy as np
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.base import register_function

@register_function("HighConditionedElliptic")
class HighConditionedElliptic(OptimizationFunction):
    r"""
    Core Elliptic function used inside CEC-2005 F3.

    The function is defined as:
        f(x) = Σ_{i=1..D} (condition)^{(i-1)/(D-1)} · x_i²
    where condition is typically 1e6.

    *Unimodal, convex, continuous, differentiable, extremely ill-conditioned.*

    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    initialization_bounds : Bounds, optional
        Bounds for random initialization. If None, defaults to [-100, 100]^d.
    operational_bounds : Bounds, optional
        Bounds for domain enforcement. If None, defaults to [-100, 100]^d.
    condition : float, optional
        Conditioning parameter (default: 1e6).

    References
    ----------
    .. [1] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
           "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
           optimization". Nanyang Technological University, Singapore, Tech. Rep.
    .. [2] Hansen, N., Müller, S. D., & Koumoutsakos, P. (2003). 
           "Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES)". 
           Evolutionary Computation, 11(1), 1-18.
    """
    def __init__(self, dimension: int,
                 initialization_bounds: Bounds = None,
                 operational_bounds: Bounds = None,
                 condition: float = 1e6):
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
        self.condition = condition
        if dimension > 1:
            idx = np.arange(dimension, dtype=np.float64)
            self._weights = self.condition ** (idx / (dimension - 1))
        else:
            self._weights = np.ones(1, dtype=np.float64)

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the High Conditioned Elliptic function at a single point.

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
        return float(np.dot(self._weights, x ** 2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch evaluation of the High Conditioned Elliptic function.

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
        return np.sum(self._weights * X * X, axis=1)
    
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