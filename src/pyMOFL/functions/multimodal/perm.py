"""
Perm (0,d,β) function implementation.

This module implements the Perm (0,d,β) function, a multimodal optimization problem
featuring a nested summation that creates a ridge-valley structure. This function
is part of the SPSO benchmark suite as function ID 20.

References:
    .. [1] Surjanovic, S., & Bingham, D. (2013). *Virtual Library of Simulation Experiments: 
           Test Functions and Datasets*. Simon Fraser University.
           https://www.sfu.ca/~ssurjano/permdb.html
"""

import numpy as np
from ...base import OptimizationFunction


class PermFunction(OptimizationFunction):
    """
    Perm (0,d,β) function with d=5, β=0.5 (SPSO ID-20).
    
    This multimodal function features a nested summation and a pronounced ridge-valley 
    structure. Although SPSO classifies this as discrete (the global optimum is at integer 
    coordinates), the analytic form is continuous. Following SPSO convention, this 
    implementation rounds particle positions to the nearest integer before evaluation.
    
    Mathematical definition:
    
    f(x) = sum_{i=1}^{5} { sum_{j=1}^{5} [(j + β)·(x_j/j)^i - 1] }^2
    
    Global minimum: f = 0 at x = (1, 2, 3, 4, 5)
    
    Attributes:
        dimension (int): Always 5.
        bounds (np.ndarray): Default bounds are [-5, 5] for each dimension.
        beta (float): The β parameter, default is 0.5 for SPSO.
    
    Properties:
        - Continuous but rugged landscape
        - Non-separable
        - Steep ridges near the optimum
        - Highly ill-conditioned valley leading to global minimum
    
    References:
        .. [1] Surjanovic, S., & Bingham, D. (2013). *Virtual Library of 
               Simulation Experiments: Test Functions and Datasets*.
               Simon Fraser University.
               https://www.sfu.ca/~ssurjano/permdb.html
               
    Note:
        To add a bias to the function, use the BiasedFunction decorator from the decorators module.
    """
    
    def __init__(self, beta: float = 0.5, bounds: np.ndarray = None):
        """
        Initialize the Perm function.
        
        Args:
            beta (float, optional): The β parameter. Defaults to 0.5 for SPSO.
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          Defaults to [-5, 5] for each dimension.
        """
        # Perm function is always 5D for SPSO
        dimension = 5
        
        # Set default bounds to [-5, 5] for each dimension
        if bounds is None:
            bounds = np.array([[-5, 5]] * dimension)
        
        super().__init__(dimension, bounds)
        self.beta = beta
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Perm function at point x.
        
        Args:
            x (np.ndarray): A 5D point.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Optional SPSO rounding step to emulate 'discrete' specification
        z = np.rint(x).astype(float)
        
        # Initialize result
        result = 0.0
        
        # Calculate using the exact formula
        for i in range(1, self.dimension + 1):
            inner_sum = 0.0
            for j in range(1, self.dimension + 1):
                # (j+beta)*((x_j/j)^i - 1)
                term = (j + self.beta) * (np.power(z[j-1] / j, i) - 1)
                inner_sum += term
            result += inner_sum ** 2
        
        return float(result)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Perm function for a batch of points.
        
        Args:
            X (np.ndarray): A batch of 5D points.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        # Optional SPSO rounding step to emulate 'discrete' specification
        Z = np.rint(X).astype(float)
        
        # Compute results for each point individually to ensure correctness
        result = np.zeros(len(Z))
        
        for idx, z in enumerate(Z):
            result_val = 0.0
            for i in range(1, self.dimension + 1):
                inner_sum = 0.0
                for j in range(1, self.dimension + 1):
                    # (j+beta)*((x_j/j)^i - 1)
                    term = (j + self.beta) * (np.power(z[j-1] / j, i) - 1)
                    inner_sum += term
                result_val += inner_sum ** 2
            result[idx] = result_val
        
        return result
    
    @staticmethod
    def get_global_minimum(dimension: int = 5) -> tuple:
        """
        Get the global minimum of the function.
        
        Args:
            dimension (int, optional): The dimension of the function.
                                      Should be 5 for the standard Perm function.
            
        Returns:
            tuple: A tuple containing the global minimum point and the function value at that point.
        """
        if dimension != 5:
            raise ValueError("Perm function is only defined for 5 dimensions in this implementation")
            
        # The global minimum is at (1, 2, 3, 4, 5)
        global_min_point = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        global_min_value = 0.0
        return global_min_point, global_min_value 