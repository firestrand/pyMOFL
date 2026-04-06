"""
Cola function implementation.

The Cola function models a layout optimization problem for 10 cities with known
inter-city distances. It has 17 decision variables representing the coordinates
of cities 2-10 (city 1 is fixed at the origin, city 2 has only its x-coordinate
free with y=0).

References
----------
.. [1] Rao, S. (2009). "Engineering Optimization: Theory and Practice", 4th ed.
.. [2] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions
       for global optimization problems". IJMMNO, 4(2), 150-194.
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register

# Upper-triangular inter-city distance matrix for 10 cities (row i, col j>i)
_D_MATRIX = [
    [0],
    [1.27, 0],
    [1.69, 1.43, 0],
    [2.04, 2.35, 2.43, 0],
    [3.09, 3.18, 3.26, 2.85, 0],
    [3.20, 3.22, 3.27, 2.88, 1.55, 0],
    [2.86, 2.56, 2.58, 2.59, 3.12, 3.06, 0],
    [3.17, 3.18, 3.18, 3.12, 3.45, 3.44, 2.77, 0],
    [3.21, 3.18, 3.18, 3.17, 3.45, 3.44, 2.77, 0.50, 0],
    [2.38, 2.31, 2.42, 1.94, 2.85, 2.81, 2.56, 2.91, 2.97, 0],
]


def _build_full_distance_matrix() -> np.ndarray:
    """Build the full 10x10 symmetric distance matrix from the upper-triangular data."""
    n = 10
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(len(_D_MATRIX[i])):
            d[i][j] = _D_MATRIX[i][j]
            d[j][i] = _D_MATRIX[i][j]
    return d


_FULL_D = _build_full_distance_matrix()


@register("Cola")
@register("cola")
class ColaFunction(OptimizationFunction):
    """
    Cola function.

    Models the layout of 10 cities with known inter-city distances.
    The decision vector x encodes coordinates of cities 2-10:
    - City 1: fixed at (0, 0)
    - City 2: (x[0], 0)  -- only x-coordinate is free
    - City k (k=3..10): (x[2k-5], x[2k-4])

    f(x) = sum_{i<j} (r_ij - d_ij)^2

    Properties: Continuous, Non-Separable, Non-Scalable, Multimodal
    Domain: [-4, 4]^17
    Global minimum: f* ~ 11.7464

    Parameters
    ----------
    dimension : int
        Must be 17.
    """

    def __init__(
        self,
        dimension: int = 17,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 17:
            raise ValueError("Cola function requires dimension=17")

        default_bounds = Bounds(
            low=np.full(dimension, -4.0),
            high=np.full(dimension, 4.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

    def _reconstruct_coordinates(self, x: np.ndarray) -> np.ndarray:
        """Reconstruct 10x2 city coordinates from the 17-dimensional decision vector."""
        coords = np.zeros((10, 2))
        # City 1: (0, 0) -- already zeros
        # City 2: (x[0], 0)
        coords[1, 0] = x[0]
        # Cities 3-10: (x[2k-5], x[2k-4]) for k=3..10
        for k in range(3, 11):
            coords[k - 1, 0] = x[2 * k - 5]
            coords[k - 1, 1] = x[2 * k - 4]
        return coords

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Cola function."""
        x = self._validate_input(x)
        coords = self._reconstruct_coordinates(x)

        result = 0.0
        for i in range(10):
            for j in range(i + 1, 10):
                diff = coords[i] - coords[j]
                r_ij = np.sqrt(np.sum(diff**2))
                result += (r_ij - _FULL_D[i, j]) ** 2
        return float(result)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the Cola function for a batch of points."""
        X = self._validate_batch_input(X)
        return np.array([self.evaluate(x) for x in X])

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get approximate global minimum.

        The global minimum is approximately 11.7464 from the literature.
        The exact optimal coordinates depend on the specific distance
        reconstruction encoding. We return the origin as a placeholder point
        along with the known approximate optimal value.

        Returns
        -------
        tuple[np.ndarray, float]
            (approximate_point, approximate_optimal_value)
        """
        # The exact optimum point is not standardized in the literature.
        # We return zeros as a placeholder; the known f* ~ 11.7464.
        return np.zeros(self.dimension), 11.7464
