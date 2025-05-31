"""
Composite function implementation.

This module provides a class for creating composite functions by combining
multiple base functions with weighted sums.
"""

import numpy as np
from typing import List, Optional
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.bounds import Bounds


class CompositeFunction(OptimizationFunction):
    """
    A composite function that combines multiple base functions with weighted sums.
    
    The composite function evaluates each component function and combines their values
    using a weighted sum, where the weights are computed based on the distance to each
    component's optimum.
    
    Parameters
    ----------
    components : List[OptimizationFunction]
        The component functions.
    sigmas : List[float]
        The sigma values for weight computation.
    lambdas : List[float]
        The scaling factors for each component.
    biases : List[float]
        The bias values for each component.
    initialization_bounds : Bounds, optional
        Bounds for initialization. If None, uses the bounds of the first component.
    operational_bounds : Bounds, optional
        Bounds for operation. If None, uses the bounds of the first component.
    """
    
    def __init__(self,
                 components: List[OptimizationFunction],
                 sigmas: List[float],
                 lambdas: List[float],
                 biases: List[float],
                 initialization_bounds: Optional[Bounds] = None,
                 operational_bounds: Optional[Bounds] = None):
        """
        Initialize the composite function.
        
        Args:
            components (List[OptimizationFunction]): The component functions.
            sigmas (List[float]): The sigma values for weight computation.
            lambdas (List[float]): The scaling factors for each component.
            biases (List[float]): The bias values for each component.
            initialization_bounds (Bounds, optional): Bounds for initialization.
                                                    If None, uses the bounds of the first component.
            operational_bounds (Bounds, optional): Bounds for operation.
                                                    If None, uses the bounds of the first component.
        """
        # Check if all components have the same dimension
        dimensions = [comp.dimension for comp in components]
        if len(set(dimensions)) != 1:
            raise ValueError("All component functions must have the same dimension")
        
        # Initialize with the dimension of the components
        dimension = dimensions[0]
        
        # Use bounds from the first component if not provided
        if initialization_bounds is None:
            initialization_bounds = components[0].initialization_bounds
        if operational_bounds is None:
            operational_bounds = components[0].operational_bounds
        
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds
        )
        
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

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch evaluation of the composite function.

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