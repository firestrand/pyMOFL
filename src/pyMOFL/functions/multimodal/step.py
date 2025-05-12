"""
Step function implementation (De Jong's biased variant).

A 10-dimensional discontinuous surface inherited from De Jong's original test set,
with an additive bias introduced by Clerc. The function applies a step operation
(floor of x+0.5) to each coordinate before squaring and summing.

References:
    .. [1] De Jong, K.A. (1975). "An Analysis of the Behavior of a Class of
           Genetic Adaptive Systems." PhD thesis, University of Michigan.
    .. [2] Clerc, M. (2012). "Standard Particle Swarm Optimisation 2007/2011 –
           Benchmark Suite and Descriptions", Technical Note.
"""

import numpy as np
from ...base import OptimizationFunction


class StepFunction(OptimizationFunction):
    """
    Step function (De Jong's Step function).
    
    Mathematical definition:
    f(x) = sum_{i=1}^{10} [step(x_i)]^2
    
    where step(x_i) = floor(x_i + 0.5)
    
    Global minimum: f = 0 at any point with all coordinates in the interval [-0.5, 0.5)
    
    Attributes:
        dimension (int): Always 10 for the Step function.
        bounds (np.ndarray): Default bounds are [-100, 100] for each dimension.
    
    Properties:
        - Discontinuous
        - Non-separable
        - Large flat plateaus
        - Many ties for global optimum
    
    References:
        .. [1] De Jong, K.A. (1975). "An Analysis of the Behavior of a Class of
               Genetic Adaptive Systems." PhD thesis, University of Michigan.
        .. [2] Clerc, M. (2012). "Standard Particle Swarm Optimisation 2007/2011 –
               Benchmark Suite and Descriptions", Technical Note.
    
    Note:
        To add a bias to the function, use the BiasedFunction decorator from the decorators module.
        The standard SPSO benchmark uses a bias of 30.0.
    """
    
    def __init__(self, bounds: np.ndarray = None):
        """
        Initialize the Step function.
        
        Args:
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          Defaults to [-100, 100] for each dimension.
        """
        # Step function is always 10D
        dimension = 10
        
        # Set default bounds to [-100, 100] for each dimension
        if bounds is None:
            bounds = np.array([[-100, 100]] * dimension)
        
        super().__init__(dimension, bounds)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Step function at point x.
        
        Args:
            x (np.ndarray): A 10D point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Apply step function: floor(x + 0.5)
        stepped = np.floor(x + 0.5)
        
        # Square and sum
        result = np.sum(stepped ** 2)
        
        return float(result)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Step function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of 10D points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        # Apply step function: floor(x + 0.5)
        stepped = np.floor(X + 0.5)
        
        # Square and sum along each row
        result = np.sum(stepped ** 2, axis=1)
        
        return result 