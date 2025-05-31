"""
BoundaryAdjustedShift: Input-transforming decorator for applying a boundary-adjusted shift transformation to an OptimizationFunction.

Inherits from InputTransformingFunction. Subclasses should only implement _apply and _apply_batch, never override evaluate or evaluate_batch.

Usage:
    base = SphereFunction(...)
    f = BoundaryAdjustedShift(base_function=base, shift_vector=...)
    value = f(x)
"""

import numpy as np
from pyMOFL.core.composable_function import InputTransformingFunction

class BoundaryAdjustedShift(InputTransformingFunction):
    """
    A decorator that applies a shift transformation with boundary adjustments.
    
    This decorator modifies the shift vector by setting certain indices to boundary
    values before applying the shift. This is used for Schwefel's Problem 2.6 where:
    - First quarter of indices are set to -100.0
    - Last quarter of indices are set to 100.0  
    - Then computes B = A * adjusted_shift_vector for matrix-based functions
    
    The transformation applies: input_transformed = input - adjusted_shift_vector
    """
    def __init__(self, base_function=None, dimension=None, shift_vector=None, initialization_bounds=None, operational_bounds=None):
        if shift_vector is None:
            raise ValueError("BoundaryAdjustedShift requires a shift_vector.")
        if base_function is not None:
            dim = base_function.dimension
        else:
            dim = dimension
        self.original_shift = np.array(shift_vector[:dim])
        self.adjusted_shift = self._apply_boundary_adjustments(self.original_shift)
        # If base function has set_B_vector method (for matrix functions), compute B = A * adjusted_shift
        if base_function is not None and hasattr(base_function, 'set_B_vector') and hasattr(base_function, 'A_matrix'):
            B_vector = np.dot(base_function.A_matrix, self.adjusted_shift)
            base_function.set_B_vector(B_vector)
        super().__init__(base_function=base_function, dimension=dimension, initialization_bounds=initialization_bounds, operational_bounds=operational_bounds)

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
        adjusted[:first_quarter_end] = -100.0
        # Last quarter adjustment
        last_quarter_start = (3 * nreal) // 4 - 1
        adjusted[last_quarter_start:] = 100.0
        return adjusted

    def _apply(self, x):
        return x - self.adjusted_shift

    def _apply_batch(self, X):
        return X - self.adjusted_shift[np.newaxis, :]
    