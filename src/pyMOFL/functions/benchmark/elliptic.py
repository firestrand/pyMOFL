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

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("HighConditionedElliptic")
@register("elliptic")
@register("high_conditioned_elliptic")
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
    condition : float, optional
        Conditioning parameter (default: 1e6).
    initialization_bounds : Bounds, optional
        Bounds for random initialization. If None, defaults to [-100, 100]^d.
    operational_bounds : Bounds, optional
        Bounds for domain enforcement. If None, defaults to [-100, 100]^d.

    References
    ----------
    .. [1] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
           "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
           optimization". Nanyang Technological University, Singapore, Tech. Rep.
    .. [2] Hansen, N., Müller, S. D., & Koumoutsakos, P. (2003).
           "Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES)".
           Evolutionary Computation, 11(1), 1-18.
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        condition: float = 1e6,
        **kwargs,
    ):
        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.full(dimension, -100.0),
                high=np.full(dimension, 100.0),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, -100.0),
                high=np.full(dimension, 100.0),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
            **kwargs,
        )
        self.condition = condition
        if dimension > 1:
            idx = np.arange(dimension, dtype=np.float64)
            self._weights = self.condition ** (idx / (dimension - 1))
        else:
            self._weights = np.ones(1, dtype=np.float64)

    def evaluate(self, x: np.ndarray) -> float:
        """Compute the elliptic function value."""
        x = self._validate_input(x)
        return float(np.dot(self._weights, x**2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute elliptic function for batch."""
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
