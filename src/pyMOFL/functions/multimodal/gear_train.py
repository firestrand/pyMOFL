"""
Gear Train function implementation.

This module implements the Gear Train function, a discrete optimization problem where
the goal is to choose the number of teeth on a compound gear train to achieve a target
speed-reduction ratio. This engineering problem was first published by Sandgren (1990).

References:
    .. [1] Sandgren, E. (1990). "Nonlinear Integer and Discrete Programming
           in Mechanical Design Optimization." *Journal of Mechanical Design*,
           112(2), 223-229.
"""

import numpy as np
from ...base import OptimizationFunction


class GearTrainFunction(OptimizationFunction):
    """
    Gear Train function (SPSO ID-18).
    
    This discrete optimization problem aims to find the number of teeth on four gears
    in a compound gear train to match a target ratio of approximately 1:6.931 as
    closely as possible.
    
    Mathematical definition:
    f(z₁, z₂, z₃, z₄) = | (z₁·z₂)/(z₃·z₄) - Rₜₐᵣgₑₜ |
    
    where Rₜₐᵣgₑₜ = 1/6.931 ≈ 0.144279, and 12 ≤ zᵢ ≤ 60 (integers).
    
    Global minimum: f = 1.643428e-6 at (z₁, z₂, z₃, z₄) = (16, 19, 43, 49)
    Note: Multiple symmetry-equivalent designs achieve the same error.
    
    Attributes:
        dimension (int): Always 4 (four gear tooth counts).
        bounds (np.ndarray): Fixed bounds are [12, 60] for each dimension.
    
    Properties:
        - Discrete variables (integers)
        - Multimodal
        - Plateau-like (many ties in function value)
    
    References:
        .. [1] Sandgren, E. (1990). "Nonlinear Integer and Discrete Programming
               in Mechanical Design Optimization." *Journal of Mechanical Design*,
               112(2), 223-229.
    
    Note:
        To add a bias to the function, use the BiasedFunction decorator from the decorators module.
    """
    
    # Constants
    _LOW = 12    # Minimum number of teeth
    _HIGH = 60   # Maximum number of teeth
    _R_TARGET = 1.0 / 6.931  # Target gear ratio
    
    def __init__(self, bounds: np.ndarray = None):
        """
        Initialize the Gear Train function.
        
        Args:
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          Defaults to [12, 60] for each dimension.
        """
        # Gear Train function is always 4D
        dimension = 4
        
        # Set default bounds to [12, 60] for each dimension
        if bounds is None:
            bounds = np.array([[self._LOW, self._HIGH]] * dimension)
        
        super().__init__(dimension, bounds)
    
    def _clip_round(self, z: np.ndarray) -> np.ndarray:
        """
        Clamp values to [12, 60] and round to nearest integer.
        
        Args:
            z (np.ndarray): Input array of gear tooth counts.
            
        Returns:
            np.ndarray: Clipped and rounded integer values.
        """
        z = np.clip(z, self._LOW, self._HIGH)
        return np.rint(z).astype(int)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Gear Train function at point x.
        
        Args:
            x (np.ndarray): A 4D point representing gear tooth counts.
            
        Returns:
            float: The absolute deviation from the target ratio.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Convert to integers within bounds
        z = self._clip_round(x)
        
        # Calculate gear ratio
        ratio = (z[0] * z[1]) / (z[2] * z[3])
        
        # Calculate absolute deviation from target ratio
        result = np.abs(ratio - self._R_TARGET)
        
        return float(result)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Gear Train function for a batch of points.
        
        Args:
            X (np.ndarray): A batch of 4D points.
            
        Returns:
            np.ndarray: The absolute deviation from the target ratio for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        # Convert to integers within bounds
        z = self._clip_round(X)
        
        # Calculate gear ratio for each point
        ratio = (z[:, 0] * z[:, 1]) / (z[:, 2] * z[:, 3])
        
        # Calculate absolute deviation from target ratio
        result = np.abs(ratio - self._R_TARGET)
        
        return result 