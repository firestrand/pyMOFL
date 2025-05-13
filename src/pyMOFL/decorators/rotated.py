"""
Rotated function decorator implementation.

This module provides a decorator that applies a rotation transformation to a base optimization function.
The rotation follows the CEC convention where rotation is applied as M*x (matrix times vector).
"""

import numpy as np
from ..base import OptimizationFunction
from ..utils.rotation import generate_rotation_matrix


class RotatedFunction(OptimizationFunction):
    """
    A decorator that applies a rotation transformation to a base optimization function.
    
    The rotation transformation rotates the search space around the origin.
    The rotation is applied following the CEC benchmark convention, where the
    transformation is M*x (matrix times vector) rather than x*M. This is the standard
    in scientific computing and most optimization benchmarks.
    
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
        
        Following the CEC convention, rotation is applied as M*x (matrix times vector).
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Apply the rotation transformation as M*x (CEC convention)
        rotated_x = np.dot(self.rotation_matrix, x)
        
        # Evaluate the base function
        return self.base.evaluate(rotated_x)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the rotated function on a batch of points.
        
        Following the CEC convention, rotation is applied as M*x (matrix times vector).
        For batch evaluation, we need to apply rotation to each vector individually.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        # Apply rotation to each vector individually with M*x
        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            # Apply rotation to each point
            rotated_x = np.dot(self.rotation_matrix, X[i])
            # Evaluate the base function
            result[i] = self.base.evaluate(rotated_x)
        
        return result 