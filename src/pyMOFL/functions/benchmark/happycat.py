"""
HappyCat and HGBat function implementations.

These structurally similar CEC 2014 functions both combine L2 norms with
linear sums, sharing the same domain, optimum location, and properties.

- HappyCat: ||x|²-D|^(1/4) + (0.5|x|²+Σxᵢ)/D + 0.5
- HGBat:    |(|x|²)²-(Σxᵢ)²|^(1/2) + (0.5|x|²+Σxᵢ)/D + 0.5

References
----------
.. [1] Beyer, H.G. & Finck, S. (2012). "HappyCat — A Simple Function Class
       Where Well-Known Direct Search Algorithms Do Fail."
.. [2] Toz, M. (2014). "HGBat benchmark function." (referenced in CEC 2014
       technical report)
.. [3] Liang, J.J., et al. (2014). "Problem definitions and evaluation criteria
       for the CEC 2014 special session on single objective real-parameter
       numerical optimization."
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("HappyCat")
@register("happycat")
class HappyCatFunction(OptimizationFunction):
    """
    HappyCat function.

    f(x) = ||x|²-D|^(1/4) + (0.5·|x|² + Σxᵢ)/D + 0.5

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Multimodal
    Domain: [-100, 100]^D
    Global minimum: f(-1, ..., -1) = 0
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, -100.0),
            high=np.full(dimension, 100.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Compute HappyCat: ||x|²-D|^(1/4) + (0.5|x|²+Σxᵢ)/D + 0.5."""
        x = self._validate_input(x)
        r2 = np.sum(x**2)
        sum_x = np.sum(x)
        return float(
            np.abs(r2 - self.dimension) ** 0.25 + (0.5 * r2 + sum_x) / self.dimension + 0.5
        )

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute HappyCat for a batch of points."""
        X = self._validate_batch_input(X)
        r2 = np.sum(X**2, axis=1)
        sum_x = np.sum(X, axis=1)
        return np.abs(r2 - self.dimension) ** 0.25 + (0.5 * r2 + sum_x) / self.dimension + 0.5

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        """Get the global minimum of the HappyCat function.

        Parameters
        ----------
        dimension : int
            The dimension of the function.

        Returns
        -------
        tuple
            (global_min_point, global_min_value)
        """
        return np.full(dimension, -1.0), 0.0


@register("HGBat")
@register("hgbat")
class HGBatFunction(OptimizationFunction):
    """
    HGBat function.

    Structurally similar to HappyCat but uses a different power term involving
    the difference of squared norms.

    f(x) = |(|x|²)² - (Σxᵢ)²|^(1/2) + (0.5·|x|² + Σxᵢ)/D + 0.5

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Multimodal
    Domain: [-100, 100]^D
    Global minimum: f(-1, ..., -1) = 0
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, -100.0),
            high=np.full(dimension, 100.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Compute HGBat: |(|x|²)²-(Σxᵢ)²|^(1/2) + (0.5|x|²+Σxᵢ)/D + 0.5."""
        x = self._validate_input(x)
        r2 = np.sum(x**2)
        sum_x = np.sum(x)
        return float(np.abs(r2**2 - sum_x**2) ** 0.5 + (0.5 * r2 + sum_x) / self.dimension + 0.5)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute HGBat for a batch of points."""
        X = self._validate_batch_input(X)
        r2 = np.sum(X**2, axis=1)
        sum_x = np.sum(X, axis=1)
        return np.abs(r2**2 - sum_x**2) ** 0.5 + (0.5 * r2 + sum_x) / self.dimension + 0.5

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        """Get the global minimum of the HGBat function.

        Parameters
        ----------
        dimension : int
            The dimension of the function.

        Returns
        -------
        tuple
            (global_min_point, global_min_value)
        """
        return np.full(dimension, -1.0), 0.0
