"""
Boundary adjusted shift function decorator implementation.

This module provides a decorator that applies a shift transformation with boundary
adjustments. This is specifically used for functions like Schwefel's Problem 2.6
where certain indices of the shift vector need to be set to boundary values.
"""

import numpy as np
from pyMOFL.core.function import OptimizationFunction


class BoundaryAdjustedShiftFunction(OptimizationFunction):
    """
    A decorator that applies a shift transformation with boundary adjustments.
    
    This decorator modifies the shift vector by setting certain indices to boundary
    values before applying the shift. This is used for Schwefel's Problem 2.6 where:
    - First quarter of indices are set to -100.0
    - Last quarter of indices are set to 100.0  
    - Then computes B = A * adjusted_shift_vector for matrix-based functions
    
    The transformation applies: input_transformed = input - adjusted_shift_vector
    
    Attributes:
        base (OptimizationFunction): The base optimization function to decorate.
        original_shift (np.ndarray): The original shift vector before boundary adjustment.
        adjusted_shift (np.ndarray): The shift vector after boundary adjustments.
        dimension (int): The dimensionality inherited from the base function.
        bounds (np.ndarray): The bounds inherited from the base function.
    """
    
    def __init__(self, base_function: OptimizationFunction, shift_vector: np.ndarray):
        """
        Initialize the boundary adjusted shift decorator.
        
        Args:
            base_function: The base optimization function to apply shift to.
            shift_vector: The original shift vector to be boundary-adjusted.
        """
        self.base = base_function
        self.dimension = base_function.dimension
        self.constraint_penalty = base_function.constraint_penalty
        self.original_shift = np.array(shift_vector[:self.dimension])
        
        # Apply boundary adjustments following C implementation
        self.adjusted_shift = self._apply_boundary_adjustments(self.original_shift)
        
        # If base function has set_B_vector method (for matrix functions), compute B = A * adjusted_shift
        if hasattr(self.base, 'set_B_vector') and hasattr(self.base, 'A_matrix'):
            B_vector = np.dot(self.base.A_matrix, self.adjusted_shift)
            self.base.set_B_vector(B_vector)
    
    def _apply_boundary_adjustments(self, shift_vector: np.ndarray) -> np.ndarray:
        """
        Apply boundary adjustments to shift vector following CEC 2005 C implementation.
        
        From the C code:
        if (nreal%4==0)
            index = nreal/4;
        else
            index = nreal/4 + 1;
        for (i=0; i<index; i++)
            o[0][i] = -100.0;
        index = (3*nreal)/4 - 1;
        for (i=index; i<nreal; i++)
            o[0][i] = 100.0;
        
        Args:
            shift_vector: Original shift vector
            
        Returns:
            Boundary-adjusted shift vector
        """
        adjusted = shift_vector.copy()
        nreal = len(shift_vector)
        
        # First quarter adjustment  
        if nreal % 4 == 0:
            first_quarter_end = nreal // 4
        else:
            first_quarter_end = nreal // 4 + 1
            
        # Set first quarter to -100.0
        adjusted[:first_quarter_end] = -100.0
        
        # Last quarter adjustment
        last_quarter_start = (3 * nreal) // 4 - 1
        
        # Set last quarter to 100.0
        adjusted[last_quarter_start:] = 100.0
        
        return adjusted
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the function with boundary adjusted shift applied.
        
        Args:
            x: Input vector to evaluate.
            
        Returns:
            Function value after applying boundary adjusted shift.
        """
        # Apply shift: f(x - adjusted_shift) 
        shifted_x = x - self.adjusted_shift
        
        return self.base.evaluate(shifted_x)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the function for a batch of inputs with boundary adjusted shift.
        
        Args:
            X: Batch of input vectors to evaluate.
            
        Returns:
            Array of function values after applying boundary adjusted shift.
        """
        # Apply shift to all inputs
        shifted_X = X - self.adjusted_shift[np.newaxis, :]
        
        return self.base.evaluate_batch(shifted_X)

    def violations(self, x):
        return self.base.violations(x - self.adjusted_shift)

    @property
    def initialization_bounds(self):
        return self.base.initialization_bounds

    @property
    def operational_bounds(self):
        return self.base.operational_bounds 