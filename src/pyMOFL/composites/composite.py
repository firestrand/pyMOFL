"""
Composite function implementation.

This module provides a class for creating composite functions by combining
multiple base functions with weighted sums.
"""

import numpy as np
from typing import List, Optional
from ..base import OptimizationFunction


class CompositeFunction(OptimizationFunction):
    """
    A composite function that combines multiple base functions with weighted sums.
    
    The composite function evaluates each component function and combines their values
    using a weighted sum, where the weights are computed based on the distance to each
    component's optimum.
    
    Attributes:
        components (List[OptimizationFunction]): The component functions.
        sigmas (np.ndarray): The sigma values for weight computation.
        lambdas (np.ndarray): The scaling factors for each component.
        biases (np.ndarray): The bias values for each component.
        dimension (int): The dimensionality of the function.
    """
    
    def __init__(self, components: List[OptimizationFunction],
                 sigmas: List[float], lambdas: List[float],
                 biases: List[float], bounds: Optional[np.ndarray] = None):
        """
        Initialize the composite function.
        
        Args:
            components (List[OptimizationFunction]): The component functions.
            sigmas (List[float]): The sigma values for weight computation.
            lambdas (List[float]): The scaling factors for each component.
            biases (List[float]): The bias values for each component.
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          If None, uses the bounds of the first component.
        """
        # Check if all components have the same dimension
        dimensions = [comp.dimension for comp in components]
        if len(set(dimensions)) != 1:
            raise ValueError("All component functions must have the same dimension")
        
        # Initialize with the dimension of the components
        dimension = dimensions[0]
        
        # If bounds are not provided, use the bounds of the first component
        if bounds is None:
            bounds = components[0].bounds
        
        super().__init__(dimension, bounds)
        
        # Store the components and parameters
        self.components = components
        self.sigmas = np.asarray(sigmas)
        self.lambdas = np.asarray(lambdas)
        self.biases = np.asarray(biases)
        
        # Check if the number of components matches the number of parameters
        if len(components) != len(sigmas) or len(components) != len(lambdas) or len(components) != len(biases):
            raise ValueError("The number of components must match the number of parameters")
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the composite function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Evaluate each component and compute the weights
        weights = np.zeros(len(self.components))
        values = np.zeros(len(self.components))
        
        for i, component in enumerate(self.components):
            # Evaluate the component function
            f_val = component.evaluate(x)
            values[i] = self.lambdas[i] * f_val + self.biases[i]
            
            # Compute the weight based on the distance to the component's optimum
            # If the component has a 'shift' attribute, use it as the optimum
            if hasattr(component, 'shift'):
                dist2 = np.sum((x - component.shift)**2)
            else:
                # Otherwise, assume the optimum is at the origin
                dist2 = np.sum(x**2)
            
            # Compute the weight using an exponential function
            weights[i] = np.exp(-dist2 / (2 * (self.sigmas[i]**2) * self.dimension))
        
        # Normalize the weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # If all weights are zero, use the component with the minimum bias
            weights[np.argmin(self.biases)] = 1.0
        
        # Compute the weighted sum
        return float(np.dot(weights, values)) 