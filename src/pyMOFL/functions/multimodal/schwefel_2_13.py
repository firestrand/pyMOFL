"""
Schwefel's Problem 2.13 function implementation.

Schwefel's Problem 2.13 is a multimodal benchmark function with trigonometric
patterns, making it challenging for optimization algorithms.

References:
    .. [1] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
           "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
           optimization". Nanyang Technological University, Singapore, Tech. Rep.
    .. [2] Schwefel, H.-P. (1981). "Numerical optimization of computer models".
           John Wiley & Sons, Inc.
"""

import numpy as np
from ...base import OptimizationFunction


class SchwefelFunction213(OptimizationFunction):
    """
    Schwefel's Problem 2.13 function: f(x) = sum((A_i - B_i(x))^2)
    
    Where:
    A_i = sum(a_ij * sin(alpha_j) + b_ij * cos(alpha_j))
    B_i(x) = sum(a_ij * sin(x_j) + b_ij * cos(x_j))
    
    This function is multimodal and non-separable.
    
    Global minimum: f(alpha) = 0, where alpha is the predefined optimum point.
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Default bounds are [-π, π] for each dimension.
        a (np.ndarray): D×D coefficient matrix for sine terms.
        b (np.ndarray): D×D coefficient matrix for cosine terms.
        alpha (np.ndarray): Global optimum point.
        A (np.ndarray): Precomputed A_i values.
        
    References:
        .. [1] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
               "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
               optimization". Nanyang Technological University, Singapore, Tech. Rep.
        .. [2] Schwefel, H.-P. (1981). "Numerical optimization of computer models".
               John Wiley & Sons, Inc.
    """
    
    def __init__(self, dimension: int, bias: float = 0.0, bounds: np.ndarray = None, 
                 a: np.ndarray = None, b: np.ndarray = None, alpha: np.ndarray = None):
        """
        Initialize the Schwefel's Problem 2.13 function.
        
        Args:
            dimension (int): The dimensionality of the function.
            bias (float, optional): Bias term added to the function value. Defaults to 0.0.
            bounds (np.ndarray, optional): Bounds for each dimension. 
                                          Defaults to [-π, π] for each dimension.
            a (np.ndarray, optional): D×D coefficient matrix for sine terms.
                                     If None, random integers in [-100, 100] are used.
            b (np.ndarray, optional): D×D coefficient matrix for cosine terms.
                                     If None, random integers in [-100, 100] are used.
            alpha (np.ndarray, optional): Global optimum point.
                                         If None, random values in [-π, π] are used.
        """
        # Set default bounds to [-π, π] for each dimension
        if bounds is None:
            bounds = np.array([[-np.pi, np.pi]] * dimension)
        
        super().__init__(dimension, bias, bounds)
        
        # Initialize coefficient matrices if not provided
        if a is None:
            # Random integers in [-100, 100]
            a = np.random.randint(-100, 101, (dimension, dimension))
        
        if b is None:
            # Random integers in [-100, 100]
            b = np.random.randint(-100, 101, (dimension, dimension))
        
        # Initialize optimum point if not provided
        if alpha is None:
            alpha = np.random.uniform(-np.pi, np.pi, dimension)
        
        self.a = a
        self.b = b
        self.alpha = alpha
        
        # Precompute A_i values
        self.A = np.zeros(dimension)
        for i in range(dimension):
            for j in range(dimension):
                self.A[i] += a[i, j] * np.sin(alpha[j]) + b[i, j] * np.cos(alpha[j])
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Schwefel's Problem 2.13 function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        result = 0.0
        for i in range(self.dimension):
            # Calculate B_i(x)
            B_i = 0.0
            for j in range(self.dimension):
                B_i += self.a[i, j] * np.sin(x[j]) + self.b[i, j] * np.cos(x[j])
            
            # Add squared difference to result
            result += (self.A[i] - B_i)**2
        
        return float(result + self.bias)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Schwefel's Problem 2.13 function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        n_points, n_dims = X.shape
        results = np.zeros(n_points)
        
        # For each point in the batch
        for p in range(n_points):
            x = X[p]
            result = 0.0
            
            for i in range(n_dims):
                # Calculate B_i(x)
                B_i = 0.0
                for j in range(n_dims):
                    B_i += self.a[i, j] * np.sin(x[j]) + self.b[i, j] * np.cos(x[j])
                
                # Add squared difference to result
                result += (self.A[i] - B_i)**2
            
            results[p] = result
        
        return results + self.bias
    
    def get_global_minimum(self) -> tuple:
        """
        Get the global minimum of the function.
        
        Returns:
            tuple: A tuple containing the global minimum point and the function value at that point.
        """
        return self.alpha, self.bias


# Alias for backward compatibility
SchwefelProblem213Function = SchwefelFunction213 