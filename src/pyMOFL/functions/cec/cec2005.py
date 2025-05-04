"""
CEC 2005 benchmark functions implementation.

This module provides implementations of the benchmark functions from the Special Session on
Real-Parameter Optimization at the 2005 IEEE Congress on Evolutionary Computation (CEC 2005).

The CEC 2005 benchmark suite consists of 25 functions designed to test various aspects of
optimization algorithms, including:
- Unimodal functions (F01-F05)
- Basic multimodal functions (F06-F12)
- Expanded multimodal functions (F13-F14)
- Hybrid composition functions (F15-F25)

References:
    .. [1] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
           "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
           optimization". Nanyang Technological University, Singapore, Tech. Rep.
    .. [2] Hansen, N., Auger, A., Ros, R., Finck, S., & Pošík, P. (2010). "Comparing results of 31 algorithms
           from the black-box optimization benchmarking BBOB-2009". In Proceedings of the 12th annual
           conference companion on Genetic and evolutionary computation (pp. 1689-1696).
"""

import numpy as np
import os
from ...base import OptimizationFunction
from ...decorators.shifted import ShiftedFunction
from ...decorators.rotated import RotatedFunction
from ..unimodal.sphere import SphereFunction


class CEC2005Function(OptimizationFunction):
    """
    Base class for CEC 2005 benchmark functions.
    
    This class provides common functionality for all CEC 2005 functions, including:
    - Loading of shift vectors and rotation matrices
    - Handling of function-specific parameters
    - Common evaluation interface
    - Shift and rotation transformations
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Bounds for each dimension.
        function_number (int): The function number in the CEC 2005 suite (1-25).
        optimum_value (float): The global optimum value of the function.
        shift_vector (np.ndarray): The shift vector for the function.
        rotation_matrix (np.ndarray): The rotation matrix for the function.
        use_rotation (bool): Whether to apply rotation transformation.
    """
    
    def __init__(self, dimension: int, function_number: int, bounds: np.ndarray = None, 
                 data_dir: str = None):
        """
        Initialize the CEC 2005 function.
        
        Args:
            dimension (int): The dimensionality of the function.
            function_number (int): The function number in the CEC 2005 suite (1-25).
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          If None, uses the default bounds for the function.
            data_dir (str, optional): Directory containing the CEC 2005 data files.
                                     If None, random values will be used.
        """
        # Set default bounds if not provided
        if bounds is None:
            # Most CEC 2005 functions use [-100, 100] bounds
            bounds = np.array([[-100, 100]] * dimension)
        
        super().__init__(dimension, bounds)
        
        self.function_number = function_number
        self.optimum_value = 0.0  # Most CEC 2005 functions have optimum value = 0
        self.data_dir = data_dir
        
        # Initialize shift vector and rotation matrix
        self.shift_vector = np.zeros(dimension)
        self.rotation_matrix = np.eye(dimension)
        
        # Load function-specific parameters
        self._load_function_parameters()
    
    def _load_function_parameters(self):
        """
        Load function-specific parameters.
        
        This method loads the shift vector and rotation matrix from data files
        if available, or generates random values otherwise.
        
        Derived classes should override this method to load additional
        function-specific parameters.
        """
        # Load shift vector
        if self.data_dir and os.path.exists(self.data_dir):
            shift_file = os.path.join(self.data_dir, f"f{self.function_number:02d}_o.dat")
            if os.path.exists(shift_file):
                try:
                    self.shift_vector = np.loadtxt(shift_file)[:self.dimension]
                except Exception as e:
                    print(f"Error loading shift vector: {e}")
                    # Fall back to random values
                    self.shift_vector = np.random.uniform(-80, 80, self.dimension)
            else:
                # Fall back to random values
                self.shift_vector = np.random.uniform(-80, 80, self.dimension)
        else:
            # Generate random shift vector
            self.shift_vector = np.random.uniform(-80, 80, self.dimension)
        
        # Load rotation matrix if needed
        if self.use_rotation:
            if self.data_dir and os.path.exists(self.data_dir):
                rotation_file = os.path.join(self.data_dir, f"f{self.function_number:02d}_m.dat")
                if os.path.exists(rotation_file):
                    try:
                        full_matrix = np.loadtxt(rotation_file)
                        # Extract the appropriate sized matrix
                        self.rotation_matrix = full_matrix[:self.dimension, :self.dimension]
                    except Exception as e:
                        print(f"Error loading rotation matrix: {e}")
                        # Fall back to identity matrix
                        self.rotation_matrix = np.eye(self.dimension)
                else:
                    # Fall back to identity matrix
                    self.rotation_matrix = np.eye(self.dimension)
            else:
                # Generate random orthogonal matrix
                from ...utils.rotation import generate_rotation_matrix
                self.rotation_matrix = generate_rotation_matrix(self.dimension)
    
    def _transform_input(self, x: np.ndarray) -> np.ndarray:
        """
        Apply transformations to input.
        
        This method applies shift and rotation transformations to the input.
        
        Args:
            x (np.ndarray): Input vector.
            
        Returns:
            np.ndarray: Transformed vector.
        """
        # Apply shift
        z = self._apply_shift(x)
        
        # Apply rotation if needed
        if self.use_rotation:
            z = self._apply_rotation(z)
        
        return z
    
    def _transform_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Apply transformations to a batch of inputs.
        
        This method applies shift and rotation transformations to a batch of inputs.
        
        Args:
            X (np.ndarray): Batch of input vectors.
            
        Returns:
            np.ndarray: Transformed batch of vectors.
        """
        # Apply shift
        Z = X - self.shift_vector
        
        # Apply rotation if needed
        if self.use_rotation:
            Z = np.dot(Z, self.rotation_matrix.T)
        
        return Z
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # This is a placeholder that should be overridden by derived classes
        raise NotImplementedError("This method should be implemented by derived classes")


# Implementation of F01: Shifted Sphere Function
class F01(CEC2005Function):
    """
    F01: Shifted Sphere Function from CEC 2005.
    
    f(x) = sum((x - o)^2) + f_bias
    
    Global optimum: f(o) = f_bias = -450
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Bounds for each dimension, default is [-100, 100].
        shift_vector (np.ndarray): The shift vector o.
        base_function (OptimizationFunction): The base sphere function.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, data_dir: str = None):
        """
        Initialize the F01 function (Shifted Sphere).
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          Defaults to [-100, 100] for each dimension.
            data_dir (str, optional): Directory containing the CEC 2005 data files.
        """
        super().__init__(dimension, function_number=1, bounds=bounds, use_rotation=False, data_dir=data_dir)
        self.optimum_value = -450.0
        
        # Create the base sphere function
        self.base_function = SphereFunction(dimension, bounds)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the F01 function (Shifted Sphere) at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Apply shift transformation
        z = self._transform_input(x)
        
        # Compute function value using the base function formula
        return float(np.sum(z**2)) + self.optimum_value
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the F01 function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        # Apply transformations
        Z = self._transform_batch(X)
        
        # Compute function values using vectorized operations
        return np.sum(Z**2, axis=1) + self.optimum_value


# Implementation of F02: Shifted Schwefel's Problem 1.2
class F02(CEC2005Function):
    """
    F02: Shifted Schwefel's Problem 1.2 from CEC 2005.
    
    f(x) = sum(( sum(x_j - o_j) for j=1 to i )^2) + f_bias
    
    Global optimum: f(o) = f_bias = -450
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Bounds for each dimension, default is [-100, 100].
        shift_vector (np.ndarray): The shift vector o.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, data_dir: str = None):
        """
        Initialize the F02 function (Shifted Schwefel's Problem 1.2).
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          Defaults to [-100, 100] for each dimension.
            data_dir (str, optional): Directory containing the CEC 2005 data files.
        """
        super().__init__(dimension, function_number=2, bounds=bounds, use_rotation=False, data_dir=data_dir)
        self.optimum_value = -450.0
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the F02 function (Shifted Schwefel's Problem 1.2) at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Apply shift transformation
        z = self._transform_input(x)
        
        # Compute function value
        result = 0.0
        for i in range(self.dimension):
            result += np.sum(z[:i+1])**2
        
        return float(result) + self.optimum_value
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the F02 function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        # Apply transformations
        Z = self._transform_batch(X)
        
        # Compute function values
        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            z = Z[i]
            value = 0.0
            for j in range(self.dimension):
                value += np.sum(z[:j+1])**2
            result[i] = value
        
        return result + self.optimum_value


# More CEC 2005 functions can be added here following the same pattern


# Factory function to create CEC 2005 functions
def create_cec2005_function(function_number: int, dimension: int, bounds: np.ndarray = None, 
                           data_dir: str = None) -> CEC2005Function:
    """
    Create a CEC 2005 benchmark function.
    
    Args:
        function_number (int): The function number in the CEC 2005 suite (1-25).
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray, optional): Bounds for each dimension.
                                      If None, uses the default bounds for the function.
        data_dir (str, optional): Directory containing the CEC 2005 data files.
    
    Returns:
        CEC2005Function: The requested CEC 2005 function.
        
    Raises:
        ValueError: If the function number is not valid or the function is not implemented.
    """
    if function_number < 1 or function_number > 25:
        raise ValueError(f"Invalid function number: {function_number}. Must be between 1 and 25.")
    
    # Map function numbers to their implementations
    function_map = {
        1: F01,
        2: F02,
        # Add more functions as they are implemented
    }
    
    if function_number not in function_map:
        raise ValueError(f"Function F{function_number:02d} is not implemented yet.")
    
    # Create and return the requested function
    return function_map[function_number](dimension, bounds, data_dir) 