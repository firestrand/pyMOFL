"""
Rana function implementation.

The Rana function is a complex multimodal function defined by pairwise interactions
between consecutive variables.

References
----------
.. [1] Rana, S. (1998). "The Distribution of Evaluation Effort in Evolutionary Algorithms."
       PhD thesis, The University of Reading.
.. [2] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions for global
       optimization problems". International Journal of Mathematical Modelling and Numerical
       Optimisation, 4(2), 150-194.
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Rana")
@register("rana")
class RanaFunction(OptimizationFunction):
    """
    Rana function.

    f(x) = sum_{i=0}^{D-2} [x_i * sin(sqrt(|x_{i+1} - x_i + 1|)) * cos(sqrt(|x_{i+1} + x_i + 1|))
           + (x_{i+1} + 1) * cos(sqrt(|x_{i+1} - x_i + 1|)) * sin(sqrt(|x_{i+1} + x_i + 1|))]

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Multimodal
    Domain: [-500, 500]^D
    Global minimum: dimension-dependent
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, -500.0),
            high=np.full(dimension, 500.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

    def _compute(self, x_i: np.ndarray, x_next: np.ndarray) -> np.ndarray:
        """Compute pairwise Rana terms."""
        t1 = np.sqrt(np.abs(x_next - x_i + 1.0))
        t2 = np.sqrt(np.abs(x_next + x_i + 1.0))
        return x_i * np.sin(t1) * np.cos(t2) + (x_next + 1.0) * np.cos(t1) * np.sin(t2)

    def evaluate(self, x: np.ndarray) -> float:
        """Compute the Rana function value."""
        x = self._validate_input(x)
        result = np.sum(self._compute(x[:-1], x[1:]))
        return float(result)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Rana function for a batch of points."""
        X = self._validate_batch_input(X)
        x_i = X[:, :-1]
        x_next = X[:, 1:]
        terms = self._compute(x_i, x_next)
        return np.sum(terms, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Rana function.

        The global minimum is dimension-dependent and not analytically known.
        We provide a numerically evaluated reference point.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        # For D=2, approximately at (-500, -500) area
        # The exact optimum is problem-specific; return a reference evaluation
        x_opt = np.full(self.dimension, -500.0)
        return x_opt, float(self.evaluate(x_opt))
