"""
Hybrid function implementation.

This module provides a class for creating hybrid functions by combining
multiple base functions, where each function operates on a different subset of dimensions.
"""

import numpy as np

from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction


class HybridFunction(OptimizationFunction):
    """
    A hybrid function that combines multiple base functions by partitioning the input vector.

    The hybrid function divides the input vector into subsets of dimensions and applies
    different component functions to each subset.

    Parameters
    ----------
    components : List[OptimizationFunction]
        The component functions.
    partitions : List[Tuple[int, int]]
        The start and end indices for each partition.
    weights : List[float], optional
        The weights for each component. If None, all components are weighted equally.
    initialization_bounds : Bounds, optional
        Bounds for initialization. If None, constructs bounds from component bounds.
    operational_bounds : Bounds, optional
        Bounds for operation. If None, constructs bounds from component bounds.
    """

    def __init__(
        self,
        components: list[OptimizationFunction],
        partitions: list[tuple[int, int]],
        weights: list[float] | None = None,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        normalize_weights: bool = True,
    ):
        """
        Initialize the hybrid function.

        Args:
            components (List[OptimizationFunction]): The component functions.
            partitions (List[Tuple[int, int]]): The start and end indices for each partition.
            weights (List[float], optional): The weights for each component.
                                           If None, all components are weighted equally.
            initialization_bounds (Bounds, optional): Bounds for initialization.
                                                   If None, constructs bounds from component bounds.
            operational_bounds (Bounds, optional): Bounds for operation.
                                                   If None, constructs bounds from component bounds.
            normalize_weights (bool): If True, normalize weights to sum to 1.
                                     If False, use weights as-is (e.g. for plain summation).
        """
        # Check if the number of components matches the number of partitions
        if len(components) != len(partitions):
            raise ValueError("The number of components must match the number of partitions")

        # Calculate the total dimension
        total_dimension = 0
        for start, end in partitions:
            if start < 0 or end < start:
                raise ValueError(f"Invalid partition: ({start}, {end})")
            total_dimension = max(total_dimension, end)

        # Construct bounds from component bounds if not provided
        if initialization_bounds is None or operational_bounds is None:
            # Build per-variable bounds arrays
            init_lows = []
            init_highs = []
            oper_lows = []
            oper_highs = []
            for comp, (_start, _end) in zip(components, partitions, strict=False):
                # For each variable in the partition, use the corresponding component's bounds
                assert comp.initialization_bounds is not None
                assert comp.operational_bounds is not None
                for i in range(comp.dimension):
                    init_lows.append(comp.initialization_bounds.low[i])
                    init_highs.append(comp.initialization_bounds.high[i])
                    oper_lows.append(comp.operational_bounds.low[i])
                    oper_highs.append(comp.operational_bounds.high[i])
            assert components[0].initialization_bounds is not None
            assert components[0].operational_bounds is not None
            init_bounds = Bounds(
                low=np.array(init_lows),
                high=np.array(init_highs),
                mode=components[0].initialization_bounds.mode,
                qtype=components[0].initialization_bounds.qtype,
            )
            oper_bounds = Bounds(
                low=np.array(oper_lows),
                high=np.array(oper_highs),
                mode=components[0].operational_bounds.mode,
                qtype=components[0].operational_bounds.qtype,
            )
            if initialization_bounds is None:
                initialization_bounds = init_bounds
            if operational_bounds is None:
                operational_bounds = oper_bounds

        # Initialize with the total dimension
        super().__init__(
            dimension=total_dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )

        # Store the components and partitions
        self.components = components
        self.partitions = partitions

        # Set weights
        if weights is None:
            if normalize_weights:
                self.weights = np.ones(len(components)) / len(components)
            else:
                self.weights = np.ones(len(components))
        else:
            self.weights = np.asarray(weights)
            if len(self.weights) != len(components):
                raise ValueError("The number of weights must match the number of components")
            # Normalize weights
            if normalize_weights and np.sum(self.weights) > 0:
                self.weights = self.weights / np.sum(self.weights)

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the hybrid function at point x.

        Parameters
        ----------
        x : np.ndarray
            A point in the search space.

        Returns
        -------
        float
            The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)

        # Evaluate each component on its partition
        values = np.zeros(len(self.components))

        for i, (component, (start, end)) in enumerate(
            zip(self.components, self.partitions, strict=False)
        ):
            # Extract the subset of dimensions for this component
            x_subset = x[start:end]

            # Check if the subset has the correct dimension for the component
            if x_subset.shape[0] != component.dimension:
                # If not, pad or truncate to match the component's dimension
                if x_subset.shape[0] < component.dimension:
                    # Pad with zeros
                    x_subset = np.pad(x_subset, (0, component.dimension - x_subset.shape[0]))
                else:
                    # Truncate
                    x_subset = x_subset[: component.dimension]

            # Evaluate the component function
            values[i] = component.evaluate(x_subset)

        # Compute the weighted sum
        return float(np.dot(self.weights, values))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch evaluation of the hybrid function.

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
        n_points = X.shape[0]
        results = np.zeros(n_points)
        for idx in range(n_points):
            results[idx] = self.evaluate(X[idx])
        return results
