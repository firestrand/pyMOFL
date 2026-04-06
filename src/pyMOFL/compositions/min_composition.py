"""
Min composition of multiple optimization functions.

Returns the element-wise minimum of multiple optimization functions.
Useful for landscapes with multiple, disconnected basins.

Formula: f(x) = min(f1(x), f2(x), ..., fk(x))
"""

from __future__ import annotations

import numpy as np

from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction


class MinComposition(OptimizationFunction):
    """
    Min composition of multiple optimization functions.

    Returns the element-wise minimum of all component function outputs.

    Parameters
    ----------
    dimension : int
        Dimensionality of the search space.
    components : list[OptimizationFunction]
        Component functions. All components must have the same dimension.
    initialization_bounds : Bounds or None
        Bounds for initialization.
    operational_bounds : Bounds or None
        Bounds for operation.
    """

    def __init__(
        self,
        *,
        dimension: int,
        components: list[OptimizationFunction],
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
    ) -> None:
        if not components:
            raise ValueError("At least one component must be provided")

        for i, comp in enumerate(components):
            if comp.dimension != dimension:
                raise ValueError(
                    f"Component {i} dimension mismatch: {comp.dimension} != {dimension}"
                )

        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )

        self.components = components

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the min composition at point x."""
        x = self._validate_input(x)
        values = [comp.evaluate(x) for comp in self.components]
        return float(np.min(values))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the min composition for a batch of points."""
        X = self._validate_batch_input(X)
        n = X.shape[0]

        # Collect results for all components
        all_results = np.empty((len(self.components), n), dtype=np.float64)
        for i, comp in enumerate(self.components):
            all_results[i] = comp.evaluate_batch(X)

        # Return element-wise min along component axis
        return np.min(all_results, axis=0)
