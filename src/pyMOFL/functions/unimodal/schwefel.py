"""
Schwefel function implementations.

This module provides implementations of various Schwefel problems used as benchmark functions.
They are named according to their original numbering in Schwefel's work.

References:
    .. [1] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
           "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
           optimization". Nanyang Technological University, Singapore, Tech. Rep.
    .. [2] Schwefel, H.-P. (1981). "Numerical optimization of computer models".
           John Wiley & Sons, Inc.
"""

import numpy as np
from ...base import OptimizationFunction


class SchwefelFunction12(OptimizationFunction):
    """
    Schwefel's Problem 1.2 function: f(x) = sum(sum(x_j)^2) for j=1 to i, for i=1 to D
    
    This function belongs to the family of unimodal functions but is non-separable.
    It is continuous, convex, and differentiable.
    
    Global minimum: f(0, 0, ..., 0) = 0
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Default bounds are [-100, 100] for each dimension.
        
    References:
        .. [1] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
               "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
               optimization". Nanyang Technological University, Singapore, Tech. Rep.
        .. [2] Schwefel, H.-P. (1981). "Numerical optimization of computer models".
               John Wiley & Sons, Inc.
               
    Note:
        To add a bias to the function, use the BiasedFunction decorator from the decorators module.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None):
        """
        Initialize the Schwefel's Problem 1.2 function.
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension. 
                                          Defaults to [-100, 100] for each dimension.
        """
        super().__init__(dimension, bounds)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Schwefel's Problem 1.2 function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Compute the function value
        prefix = np.cumsum(x, dtype=np.float64)  # prefix[k] = Σ_{j≤k} x[j]
        return float(np.dot(prefix, prefix))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Schwefel's Problem 1.2 function on a batch of points.
        
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
            # Calculate the cumulative sums for this point
            cumulative_sums = np.zeros(n_dims)
            for i in range(n_dims):
                cumulative_sums[i] = np.sum(X[p, :i+1])
            
            # Square and sum these cumulative sums
            results[p] = np.sum(cumulative_sums**2)
        
        return results
    
    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        """
        Get the global minimum of the function.
        
        Args:
            dimension (int): The dimension of the function.
            
        Returns:
            tuple: A tuple containing the global minimum point and the function value at that point.
        """
        global_min_point = np.zeros(dimension)
        global_min_value = 0.0
        return global_min_point, global_min_value


class SchwefelFunction26(OptimizationFunction):
    """
    Schwefel's Problem 2.6 function: f(x) = max{|A_i·x - B_i|}, i=1,...,D
    
    Where A is a D×D matrix and B = A·o for some vector o.
    
    This function is unimodal and non-separable, with linear constraints.
    
    Global minimum: f(o) = 0, where o is the predefined optimum point.
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Default bounds are [-100, 100] for each dimension.
        A (np.ndarray): D×D matrix used in the function.
        B (np.ndarray): Vector B = A·o, where o is the global optimum.
        optimum_point (np.ndarray): The global optimum point.
        
    References:
        .. [1] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
               "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
               optimization". Nanyang Technological University, Singapore, Tech. Rep.
        .. [2] Schwefel, H.-P. (1981). "Numerical optimization of computer models".
               John Wiley & Sons, Inc.
               
    Note:
        To add a bias to the function, use the BiasedFunction decorator from the decorators module.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, 
                 A: np.ndarray = None, optimum_point: np.ndarray = None):
        """
        Initialize the Schwefel's Problem 2.6 function.
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension. 
                                          Defaults to [-100, 100] for each dimension.
            A (np.ndarray, optional): D×D matrix used in the function. If None, a random matrix is generated.
            optimum_point (np.ndarray, optional): The global optimum point. If None, a zero vector is used.
        """
        super().__init__(dimension, bounds)
        
        # Initialize A if not provided
        if A is None:
            # Generate a random matrix with elements in [-500, 500]
            # with a non-zero determinant
            while True:
                A = np.random.uniform(-500, 500, (dimension, dimension))
                if np.abs(np.linalg.det(A)) > 1e-10:
                    break
        
        # Initialize optimum point if not provided
        if optimum_point is None:
            optimum_point = np.zeros(dimension)
        
        self.A = A
        self.optimum_point = optimum_point
        
        # Calculate B = A·o
        self.B = np.dot(A, optimum_point)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Schwefel's Problem 2.6 function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Calculate A·x
        Ax = np.dot(self.A, x)
        
        # Calculate |A·x - B|
        abs_diff = np.abs(Ax - self.B)
        
        # Return max value
        return float(np.max(abs_diff))
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Schwefel's Problem 2.6 function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        n_points = X.shape[0]
        results = np.zeros(n_points)
        
        # For each point in the batch
        for p in range(n_points):
            # Calculate A·x
            Ax = np.dot(self.A, X[p])
            
            # Calculate |A·x - B|
            abs_diff = np.abs(Ax - self.B)
            
            # Get max value
            results[p] = np.max(abs_diff)
        
        return results
    
    def get_global_minimum(self) -> tuple:
        """
        Get the global minimum of the function.
        
        Returns:
            tuple: A tuple containing the global minimum point and the function value at that point.
        """
        return self.optimum_point, 0.0


# Aliases for backward compatibility
SchwefelProblem12Function = SchwefelFunction12
SchwefelProblem26Function = SchwefelFunction26