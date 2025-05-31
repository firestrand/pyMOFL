"""
Step function implementation (De Jong's biased variant).

A 10-dimensional discontinuous surface inherited from De Jong's original test set,
with an additive bias introduced by Clerc. The function applies a step operation
(floor of x+0.5) to each coordinate before squaring and summing.

References
----------
.. [1] De Jong, K.A. (1975). "An Analysis of the Behavior of a Class of
       Genetic Adaptive Systems." PhD thesis, University of Michigan.
.. [2] Clerc, M. (2012). "Standard Particle Swarm Optimisation 2007/2011 –
       Benchmark Suite and Descriptions", Technical Note.
"""

import numpy as np
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction

class StepFunction(OptimizationFunction):
    """
    Step function (De Jong's Step function).

    The function is defined as:
        f(x) = sum_{i=1}^{10} [step(x_i)]^2
    where step(x_i) = floor(x_i + 0.5)

    Global minimum: f = 0 at any point with all coordinates in the interval [-0.5, 0.5)

    Parameters
    ----------
    initialization_bounds : Bounds, optional
        Bounds for initialization. If None, defaults to [-100, 100]^10.
    operational_bounds : Bounds, optional
        Bounds for operation. If None, defaults to [-100, 100]^10.

    References
    ----------
    .. [1] De Jong, K.A. (1975). "An Analysis of the Behavior of a Class of
           Genetic Adaptive Systems." PhD thesis, University of Michigan.
    .. [2] Clerc, M. (2012). "Standard Particle Swarm Optimisation 2007/2011 –
           Benchmark Suite and Descriptions", Technical Note.
    """
    def __init__(self,
                 initialization_bounds: Bounds = None,
                 operational_bounds: Bounds = None):
        dimension = 10
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
        Evaluate the Step function at a single point.

        Parameters
        ----------
        x : np.ndarray
            Input vector of shape (10,).

        Returns
        -------
        float
            Function value at x.
        """
        x = self._validate_input(x)
        stepped = np.floor(x + 0.5)
        return float(np.sum(stepped ** 2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch evaluation of the Step function.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_points, 10).

        Returns
        -------
        np.ndarray
            Function values of shape (n_points,).
        """
        X = self._validate_batch_input(X)
        stepped = np.floor(X + 0.5)
        return np.sum(stepped ** 2, axis=1)

    @property
    def bounds(self) -> np.ndarray:
        """
        Legacy-compatible property for test and composite compatibility.
        Returns the operational bounds as a (dimension, 2) array [[low, high], ...].
        Prefer using .operational_bounds for new code.
        """
        return np.stack([self.operational_bounds.low, self.operational_bounds.high], axis=1) 