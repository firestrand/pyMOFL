"""
Shift then rotate function decorator implementation.

This module provides a decorator that applies both shift and rotation transformations
to a base optimization function in the correct order for CEC benchmarks: first shift, then rotate.
This ensures the transformation sequence matches the original C implementation.
"""

import numpy as np
from ..base import OptimizationFunction
from ..utils.rotation import generate_rotation_matrix


class ShiftThenRotateFunction(OptimizationFunction):
    """
    A decorator that applies shift followed by rotation to a base optimization function.
    
    This decorator handles the exact transformation sequence required by CEC benchmarks:
    1. Shift: z = x - shift_vector
    2. Rotate: z = rotation_matrix * z
    
    This matches the C implementation sequence which cannot be achieved by nesting
    shift and rotate decorators separately due to the order inversion.
    
    Attributes:
        base (OptimizationFunction): The base optimization function to be transformed.
        shift_vector (np.ndarray): The shift vector to be applied to the input.
        rotation_matrix (np.ndarray): The rotation matrix to be applied to the shifted input.
        dimension (int): The dimensionality of the function (inherited from the base function).
    """
    
    def __init__(self, base_func: OptimizationFunction, shift_vector: np.ndarray = None, 
                 rotation_matrix: np.ndarray = None):
        """
        Initialize the shift-then-rotate function decorator.
        
        Args:
            base_func (OptimizationFunction): The base optimization function to be transformed.
            shift_vector (np.ndarray, optional): The shift vector to be applied to the input.
                                               If None, a vector of zeros is used (no shift).
            rotation_matrix (np.ndarray, optional): The rotation matrix to be applied to the input.
                                                  If None, an identity matrix is used (no rotation).
        """
        self.base = base_func
        self.dimension = base_func.dimension
        
        # Handle shift vector
        if shift_vector is None:
            self.shift_vector = np.zeros(self.dimension)
        else:
            # Ensure the shift is a numpy array with the correct dimension
            self.shift_vector = np.asarray(shift_vector)
            if self.shift_vector.shape[0] != self.dimension:
                raise ValueError(f"Expected shift dimension {self.dimension}, got {self.shift_vector.shape[0]}")
        
        # Handle rotation matrix
        if rotation_matrix is None:
            # Use identity matrix (no rotation)
            self.rotation_matrix = np.eye(self.dimension)
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
        Evaluate the transformed function at point x.
        
        The transformations are applied in the CEC sequence:
        1. Shift: z = x - shift_vector
        2. Rotate: z = rotation_matrix * z
        3. Evaluate base function on z
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # 1. Apply shift transformation (parity with C implementation)
        z = x - self.shift_vector
        
        # 2. Apply rotation transformation: z = M*z (CEC convention)
        z = np.dot(self.rotation_matrix, z)
        
        # 3. Evaluate the base function on the transformed point
        return self.base.evaluate(z)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the transformed function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        # Apply transformations to each point and evaluate
        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            # 1. Apply shift
            z = X[i] - self.shift_vector
            
            # 2. Apply rotation
            z = np.dot(self.rotation_matrix, z)
            
            # 3. Evaluate the base function
            result[i] = self.base.evaluate(z)
        
        return result 