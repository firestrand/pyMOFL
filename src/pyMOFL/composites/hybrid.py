"""
Hybrid function implementation.

This module provides a class for creating hybrid functions by combining
multiple base functions, where each function operates on a different subset of dimensions.
"""

import numpy as np
from typing import List, Optional, Tuple
from ..base import OptimizationFunction


class HybridFunction(OptimizationFunction):
    """
    A hybrid function that combines multiple base functions by partitioning the input vector.
    
    The hybrid function divides the input vector into subsets of dimensions and applies
    different component functions to each subset.
    
    Attributes:
        components (List[OptimizationFunction]): The component functions.
        partitions (List[Tuple[int, int]]): The start and end indices for each partition.
        weights (np.ndarray): The weights for each component.
        dimension (int): The total dimensionality of the function.
    """
    
    def __init__(self, components: List[OptimizationFunction],
                 partitions: List[Tuple[int, int]],
                 weights: Optional[List[float]] = None,
                 bounds: Optional[np.ndarray] = None):
        """
        Initialize the hybrid function.
        
        Args:
            components (List[OptimizationFunction]): The component functions.
            partitions (List[Tuple[int, int]]): The start and end indices for each partition.
            weights (List[float], optional): The weights for each component.
                                           If None, all components are weighted equally.
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          If None, constructs bounds from component bounds.
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
        
        # Initialize with the total dimension
        super().__init__(total_dimension)
        
        # Store the components and partitions
        self.components = components
        self.partitions = partitions
        
        # Set weights
        if weights is None:
            self.weights = np.ones(len(components)) / len(components)
        else:
            self.weights = np.asarray(weights)
            if len(self.weights) != len(components):
                raise ValueError("The number of weights must match the number of components")
            # Normalize weights
            if np.sum(self.weights) > 0:
                self.weights = self.weights / np.sum(self.weights)
        
        # Construct bounds from component bounds if not provided
        if bounds is None:
            self._bounds = np.zeros((total_dimension, 2))
            for i, (comp, (start, end)) in enumerate(zip(components, partitions)):
                comp_bounds = comp.bounds
                for j, dim in enumerate(range(start, end)):
                    if j < comp_bounds.shape[0]:
                        self._bounds[dim] = comp_bounds[j]
                    else:
                        # Use default bounds if the component doesn't have enough dimensions
                        self._bounds[dim] = np.array([-100, 100])
        else:
            self._bounds = np.asarray(bounds)
            if self._bounds.shape != (total_dimension, 2):
                raise ValueError(f"Expected bounds shape ({total_dimension}, 2), got {self._bounds.shape}")
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the hybrid function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Ensure x is a numpy array
        x = np.asarray(x)
        
        # Check if the input has the correct dimension
        if x.shape[0] != self.dimension:
            raise ValueError(f"Expected input dimension {self.dimension}, got {x.shape[0]}")
        
        # Evaluate each component on its partition
        values = np.zeros(len(self.components))
        
        for i, (component, (start, end)) in enumerate(zip(self.components, self.partitions)):
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
                    x_subset = x_subset[:component.dimension]
            
            # Evaluate the component function
            values[i] = component.evaluate(x_subset)
        
        # Compute the weighted sum
        return float(np.dot(self.weights, values))
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the hybrid function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Ensure X is a numpy array
        X = np.asarray(X)
        
        # Check if the input has the correct shape
        if X.shape[1] != self.dimension:
            raise ValueError(f"Expected input dimension {self.dimension}, got {X.shape[1]}")
        
        # Initialize the result array
        result = np.zeros(X.shape[0])
        
        # Evaluate each point
        for i in range(X.shape[0]):
            result[i] = self.evaluate(X[i])
        
        return result 