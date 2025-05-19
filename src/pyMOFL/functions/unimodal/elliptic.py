"""
High Conditioned Elliptic function implementation.

The Elliptic function is a unimodal benchmark function with high conditioning.
It is continuous, convex, and differentiable.

References:
    .. [1] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
           "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
           optimization". Nanyang Technological University, Singapore, Tech. Rep.
    .. [2] Hansen, N., Müller, S. D., & Koumoutsakos, P. (2003). 
           "Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES)". 
           Evolutionary Computation, 11(1), 1-18.
"""

import numpy as np
from ...base import OptimizationFunction


class HighConditionedElliptic(OptimizationFunction):
    r"""
    Core Elliptic function used inside CEC-2005 F3.

        f(x) = Σ_{i=1..D} (10⁶)^{(i-1)/(D-1)} · x_i²

    *Unimodal, convex, continuous, differentiable, extremely ill-conditioned.*

    Parameters
    ----------
    dimension : int
        Problem dimensionality *D*.
    bounds : (D, 2) ndarray | None
        Search domain.  CEC uses ``[-100, 100]`` for every dim.
    condition : float
        Condition number (default: 10⁶). Higher values make the problem more difficult.

    Notes
    -----
    • No shift, rotation, or bias is applied here.  
    • For the full CEC-2005 F3, wrap this core with shift/rotation/bias
      utilities (or use the decorator pattern).
    """

    # ---------- construction -------------------------------------------------
    def __init__(self, dimension: int, bounds: np.ndarray = None, condition: float = 1e6):
        super().__init__(dimension, bounds)
        
        # Store condition number
        self.condition = condition

        # Pre-compute the condition^(i/(D-1)) weights once
        if dimension > 1:
            idx = np.arange(dimension, dtype=np.float64)
            self._weights = self.condition ** (idx / (dimension - 1))
        else:                       # D == 1 → condition number = 1
            self._weights = np.ones(1, dtype=np.float64)

    def evaluate(self, x: np.ndarray) -> float:
        x = self._validate_input(x)
        return float(np.dot(self._weights, x ** 2))     # x² weighted sum

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        return np.sum(self._weights * X * X, axis=1)
    
    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        """
        Get the global minimum of the function.
        
        Args:
            dimension (int): The dimension of the function.
            
        Returns:
            tuple: A tuple containing the global minimum point and the function value at that point.
        """
        global_min_point = np.zeros(dimension)
        global_min_value = 0.0
        return global_min_point, global_min_value