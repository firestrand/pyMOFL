"""
Rotated function decorator implementation.

This module provides a decorator that applies a rotation transformation to a base optimization function.
"""

import numpy as np
from ..base import OptimizationFunction
from ..utils.rotation import generate_rotation_matrix


class RotatedFunction(OptimizationFunction):
    """
    A decorator that applies a rotation transformation to a base optimization function.
    
    The rotation transformation rotates the search space around the origin.
    
    Attributes:
        base (OptimizationFunction): The base optimization function to be rotated.
        rotation_matrix (np.ndarray): The rotation matrix to be applied to the input.
        dimension (int): The dimensionality of the function (inherited from the base function).
    """
    
    def __init__(self, base_func: OptimizationFunction, rotation_matrix: np.ndarray = None):
        """
        Initialize the rotated function decorator.
        
        Args:
            base_func (OptimizationFunction): The base optimization function to be rotated.
            rotation_matrix (np.ndarray, optional): The rotation matrix to be applied to the input.
                                                  If None, a random orthogonal matrix is generated.
        """
        self.base = base_func
        self.dimension = base_func.dimension
        
        # If no rotation matrix is provided, generate a random orthogonal matrix
        if rotation_matrix is None:
            self.rotation_matrix = generate_rotation_matrix(self.dimension)
        else:
            # Ensure the rotation matrix is a numpy array with the correct shape
            self.rotation_matrix = np.asarray(rotation_matrix)
            if self.rotation_matrix.shape != (self.dimension, self.dimension):
                raise ValueError(f"Expected rotation matrix shape ({self.dimension}, {self.dimension}), "
                                f"got {self.rotation_matrix.shape}")
        
        # Use the same bounds as the base function
        self._bounds = base_func.bounds.copy()
    
    @property
    def bounds(self) -> np.ndarray:
        """
        Get the search space bounds for the function.
        
        Returns:
            np.ndarray: A 2D array of shape (dimension, 2) with lower and upper bounds.
        """
        return self._bounds
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the rotated function at point x.
        
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
        
        # Apply the rotation transformation and evaluate the base function
        rotated_x = self.rotation_matrix.dot(x)
        return self.base.evaluate(rotated_x)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the rotated function on a batch of points.
        
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
        
        # Apply the rotation transformation and evaluate the base function
        rotated_X = np.dot(X, self.rotation_matrix.T)
        return self.base.evaluate_batch(rotated_X) 