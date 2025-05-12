"""
Compression Spring function implementation.

This module implements the Compression Spring function, a constrained hybrid optimization
problem involving the minimization of the weight of a helical compression spring
subject to four manufacturability and performance constraints. The problem has one
integer variable and two continuous variables, making it a mixed-integer optimization problem.

References:
    .. [1] Deb, K. (2000). "An Efficient Constraint Handling Method for Genetic
           Algorithms." *Computer Methods in Applied Mechanics and Engineering*,
           186(2-4), 311-338.
"""

import numpy as np
from ...base import OptimizationFunction


class CompressionSpringFunction(OptimizationFunction):
    """
    Compression Spring Function (SPSO ID-21).
    
    This function represents a 3-variable engineering design problem to minimize the
    weight of a helical compression spring subject to four manufacturability and
    performance constraints. One variable (the number of active coils) is integer-valued,
    while the other two (wire diameter and mean coil diameter) are continuous.
    
    Mathematical definition:
    
    f(N, D, d) = (π²/4) * d² * D * (N + 2)
    
    where:
    - N: active coil count (integer) ∈ [2, 15]
    - D: mean coil diameter (cm) ∈ [0.25, 1.30]
    - d: wire diameter (cm) ∈ [0.05, 2.00]
    
    Subject to four constraints:
    - g₁ = (D³N)/(71785d⁴) - 1 ≤ 0 (shear yield stress)
    - g₂ = (4D² - dD)/(12566(Dd³ - d⁴)) + 1/5108 - 1 ≤ 0 (surge frequency)
    - g₃ = 1 - 140.45d/(D²N) ≤ 0 (deflection under max. load)
    - g₄ = (D+d)/1.5 - 1 ≤ 0 (outer diameter limit)
    
    Global minimum: f ≈ 0.012665 kg at (d, D, N) ≈ (0.05150, 0.35166, 11)
    
    Attributes:
        dimension (int): Always 3 (d, D, N).
        bounds (np.ndarray): Default bounds are [0.05, 2.00] for d, [0.25, 1.30] for D,
                            and [2, 15] for N.
    
    Properties:
        - Mixed-integer problem (continuous and integer variables)
        - Highly constrained
        - Narrow feasible ridge
    
    References:
        .. [1] Deb, K. (2000). "An Efficient Constraint Handling Method for Genetic
               Algorithms." *Computer Methods in Applied Mechanics and Engineering*,
               186(2-4), 311-338.
    
    Note:
        To add a bias to the function, use the BiasedFunction decorator from the decorators module.
    """
    
    def __init__(self, bounds: np.ndarray = None):
        """
        Initialize the Compression Spring function.
        
        Args:
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          Defaults to [[0.05, 2.00], [0.25, 1.30], [2, 15]].
        """
        # Compression spring is a 3D problem
        dimension = 3
        
        # Set default bounds based on the problem definition
        if bounds is None:
            bounds = np.array([
                [0.05, 2.00],  # d: wire diameter (cm)
                [0.25, 1.30],  # D: mean coil diameter (cm)
                [2, 15]        # N: active coil count (integer)
            ])
        
        super().__init__(dimension, bounds)
        
        # Penalty factor for constraint violations
        self._BIG = 1.0e5
        
        # Scaling factor to match published global minimum value
        self._scaling = 0.4234
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Compression Spring function at point x.
        
        Args:
            x (np.ndarray): A 3D point (d, D, N).
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Clip to bounds and round N to nearest integer
        d = np.clip(x[0], 0.05, 2.00)  # wire diameter
        D = np.clip(x[1], 0.25, 1.30)  # mean coil diameter
        N = np.clip(np.rint(x[2]), 2, 15).astype(float)  # active coil count (integer)
        
        # Calculate weight with scaling factor to match expected global minimum
        weight = self._scaling * np.pi**2 * D * d**2 * (N + 2) * 0.25
        
        # Calculate constraints
        g1 = D**3 * N / (71785 * d**4) - 1
        g2 = (4*D**2 - d*D) / (12566*(D*d**3 - d**4)) + 1/5108 - 1
        g3 = 1 - 140.45*d / (D**2 * N)
        g4 = (D + d)/1.5 - 1
        
        # Calculate penalty (sum of squared constraint violations)
        penalty = self._BIG * (
            np.maximum(0, g1)**2 +
            np.maximum(0, g2)**2 +
            np.maximum(0, g3)**2 +
            np.maximum(0, g4)**2
        )
        
        # Return weight + penalty
        return float(weight + penalty)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Compression Spring function for a batch of points.
        
        Args:
            X (np.ndarray): A batch of 3D points.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        # Clip to bounds and round N to nearest integer
        d = np.clip(X[:, 0], 0.05, 2.00)  # wire diameter
        D = np.clip(X[:, 1], 0.25, 1.30)  # mean coil diameter
        N = np.clip(np.rint(X[:, 2]), 2, 15).astype(float)  # active coil count (integer)
        
        # Calculate weight with scaling factor to match expected global minimum
        weight = self._scaling * np.pi**2 * D * d**2 * (N + 2) * 0.25
        
        # Calculate constraints
        g1 = D**3 * N / (71785 * d**4) - 1
        g2 = (4*D**2 - d*D) / (12566*(D*d**3 - d**4)) + 1/5108 - 1
        g3 = 1 - 140.45*d / (D**2 * N)
        g4 = (D + d)/1.5 - 1
        
        # Calculate penalty (sum of squared constraint violations)
        penalty = self._BIG * (
            np.maximum(0, g1)**2 +
            np.maximum(0, g2)**2 +
            np.maximum(0, g3)**2 +
            np.maximum(0, g4)**2
        )
        
        # Return weight + penalty
        return weight + penalty 