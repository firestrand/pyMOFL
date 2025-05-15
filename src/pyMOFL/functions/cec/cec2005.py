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

import os
import glob
import logging
import warnings
import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional, Any

# Import base functions from unimodal module
from ..unimodal.schwefel import SchwefelFunction12
from ..unimodal.sphere import SphereFunction
from ..unimodal.elliptic import HighConditionedElliptic
from ..unimodal.rosenbrock import RosenbrockFunction

from ..multimodal.ackley import AckleyFunction
from ..multimodal.griewank import GriewankFunction

# Import decorators
from ...decorators.shifted import ShiftedFunction
from ...decorators.biased import BiasedFunction
from ...decorators.noise import NoiseDecorator
from ...decorators.rotated import RotatedFunction
from ...decorators.shift_then_rotate import ShiftThenRotateFunction

# Base imports
from ...base import OptimizationFunction

# Load global bias values for CEC 2005 functions
# These are defined in the function specifications
CEC_2005_BIAS = {
    1: -450.0,   # F1: Shifted Sphere
    2: -450.0,   # F2: Shifted Schwefel 1.2
    3: -450.0,   # F3: Shifted Rotated High Conditioned Elliptic
    4: -450.0,   # F4: Shifted Schwefel 1.2 with Noise
    5: -310.0,   # F5: Schwefel 2.6 with Global Optimum on Bounds
    6: 390.0,    # F6: Shifted Rosenbrock
    7: -180.0,   # F7: Shifted Rotated Griewank without Bounds
    8: -140.0,   # F8: Shifted Rotated Ackley with Global Optimum on Bounds
    9: -330.0,   # F9: Shifted Rastrigin
    10: -330.0,  # F10: Shifted Rotated Rastrigin
    11: 90.0,    # F11: Shifted Rotated Weierstrass
    12: -460.0,  # F12: Schwefel Problem 2.13
    13: -130.0,  # F13: Shifted Expanded Griewank plus Rosenbrock
    14: -300.0,  # F14: Shifted Rotated Expanded Scaffer F6
    15: 120.0,   # F15: Hybrid Composition Function
    16: 120.0,   # F16: Rotated Version of Hybrid Composition Function F15
    17: 120.0,   # F17: F16 with Noise in Fitness
    18: 10.0,    # F18: Rotated Hybrid Composition Function
    19: 10.0,    # F19: Rotated Hybrid Composition Function with narrow basin for global optimum
    20: 10.0,    # F20: Rotated Hybrid Composition Function with the global optimum on the bounds
    21: 360.0,   # F21: Rotated Hybrid Composition Function
    22: 360.0,   # F22: Rotated Hybrid Composition Function with high condition number matrix
    23: 360.0,   # F23: Non-Continuous Rotated Hybrid Composition Function
    24: 260.0,   # F24: Rotated Hybrid Composition Function
    25: 260.0    # F25: Rotated Hybrid Composition Function without bounds
}


class CEC2005Function(OptimizationFunction):
    """
    Base class for CEC 2005 benchmark functions.
    
    This class provides common functionality for CEC 2005 benchmark functions,
    including loading data files, applying transformations, and dispatching evaluation.
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Bounds for each dimension.
        function_number (int): The function number in the CEC 2005 suite (1-25).
        shift_vector (np.ndarray): The shift vector for the function.
        rotation_matrix (np.ndarray): The rotation matrix for the function (if applicable).
        data_dir (str): Directory containing the CEC 2005 data files.
        metadata (dict): Dictionary containing metadata about the function.
        bias (float): The bias value (function value at global optimum).
    """
    
    def __init__(self, dimension: int, function_number: int, bounds: np.ndarray = None,
                 data_dir: str = None, use_rotation: bool = False):
        """
        Initialize a CEC 2005 benchmark function.
        
        Args:
            dimension (int): The dimensionality of the function.
            function_number (int): The function number in the CEC 2005 suite (1-25).
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          If None, uses the default bounds for the function.
            data_dir (str, optional): Directory containing the CEC 2005 data files.
                                     If None, uses the default directory in the package.
            use_rotation (bool, optional): Whether to use rotation for this function.
                                          Default is False.
        """
        # Set default bounds if not provided
        if bounds is None:
            # Most CEC 2005 functions use [-100, 100] bounds
            bounds = np.array([[-100, 100]] * dimension)
        
        super().__init__(dimension, bounds)
        
        self.function_number = function_number
        self.bias = 0.0  # Most CEC 2005 functions have bias value = 0
        self.use_rotation = use_rotation
        
        # Initialize metadata with default values
        self.metadata = {
            "name": f"F{function_number:02d}",
            "is_shifted": False,
            "is_rotated": use_rotation,
            "is_biased": False,
            "is_unimodal": False,
            "is_multimodal": False,
            "is_hybrid": False,
            "is_separable": True,
            "has_noise": False,
            "global_optimum_on_bounds": False,
        }
        
        # Set default data directory if not provided
        if data_dir is None:
            # Use the package's constants directory
            import os.path
            package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.data_dir = os.path.join(package_dir, 'constants', 'cec', '2005')
        else:
            self.data_dir = data_dir
        
        # Initialize arrays
        self.shift_vector = np.zeros(dimension)
        self.rotation_matrix = np.eye(dimension)
        
        # Load function parameters from data files
        self._load_function_parameters()
    
    def _load_function_parameters(self):
        """
        Load function-specific parameters.
        
        This method loads the shift vector and rotation matrix from data files
        according to the standardized CEC constant naming convention.
        
        The method expects a meta_2005.json manifest file in the constants directory,
        and will fail critically if it's missing.
        
        For CEC2005, the file structure follows the naming convention:
        - shift_D{MAX_DIM}.txt: Contains shift vector values for up to MAX_DIM dimensions
          (typically shift_D50.txt containing 50 values, from which we take the first D)
        - rot_D{DIM}.txt: Contains rotation matrix for the specified dimension
        
        The shift and rotation files are located in function-specific subdirectories:
        constants/cec/2005/f{FUNC_NUM:02d}/
        """
        import os
        import json
        
        if not self.data_dir or not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"CEC2005 data directory not found: {self.data_dir}")
        
        # Load manifest file
        manifest_path = os.path.join(self.data_dir, "meta_2005.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"CEC2005 manifest file not found: {manifest_path}")
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load CEC2005 manifest file: {e}")
        
        # Get function information from manifest
        func_key = f"f{self.function_number:02d}"
        
        # Check if function exists in the manifest
        if "functions" not in manifest or func_key not in manifest["functions"]:
            raise ValueError(f"Function {func_key} not found in manifest")
        
        func_info = manifest["functions"][func_key]
        
        # Load shift vector - try to find the appropriate file
        # CEC files typically contain data for larger dimensions (e.g., D50)
        # and we take the first D values for our dimension
        shift_file = None
        
        # First try to get the shift file path from the manifest
        if "files" in func_info and "shift" in func_info["files"]:
            potential_file = os.path.join(self.data_dir, func_info["files"]["shift"])
            if os.path.exists(potential_file):
                shift_file = potential_file
        
        # If not found in manifest, try the standard naming convention
        if shift_file is None:
            for max_dim in [50, 30, 10]:  # Try different dimension files, starting with largest
                potential_file = os.path.join(self.data_dir, func_key, f"shift_D{max_dim}.txt")
                if os.path.exists(potential_file):
                    shift_file = potential_file
                    break
        
        if shift_file is None:
            raise FileNotFoundError(f"Shift vector file not found for function {func_key}")
        
        try:
            # Load the full shift vector and take only the first D values
            full_shift_vector = np.loadtxt(shift_file)
            if len(full_shift_vector) < self.dimension:
                raise ValueError(f"Shift vector file contains {len(full_shift_vector)} values, "
                                f"but {self.dimension} are needed")
            
            self.shift_vector = full_shift_vector[:self.dimension]
        except Exception as e:
            raise RuntimeError(f"Failed to load shift vector: {e}")
        
        # Load rotation matrix if needed - usually dimension specific
        if self.use_rotation:
            rot_file = None
            
            # First try to get the dimension-specific rotation file path from the manifest
            dim_specific_key = f"rot{self.dimension}"
            if "files" in func_info and dim_specific_key in func_info["files"]:
                potential_file = os.path.join(self.data_dir, func_info["files"][dim_specific_key])
                if os.path.exists(potential_file):
                    rot_file = potential_file
            
            # Then try the generic "rotation" key if dimension-specific wasn't found
            if rot_file is None and "files" in func_info and "rotation" in func_info["files"]:
                potential_file = os.path.join(self.data_dir, func_info["files"]["rotation"])
                if os.path.exists(potential_file):
                    rot_file = potential_file
            
            # If not found in manifest, try the standard naming convention
            if rot_file is None:
                # First try exact dimension
                potential_file = os.path.join(self.data_dir, func_key, f"rot_D{self.dimension}.txt")
                if os.path.exists(potential_file):
                    rot_file = potential_file
                else:
                    # Then try to find any rotation matrix, preferring larger dimensions
                    for dim in [50, 30, 10, 2]:  # Try different dimensions, starting with largest
                        potential_file = os.path.join(self.data_dir, func_key, f"rot_D{dim}.txt")
                        if os.path.exists(potential_file):
                            rot_file = potential_file
                            break
            
            if rot_file is None:
                raise FileNotFoundError(f"Rotation matrix file not found for function {func_key} dimension {self.dimension}")
            
            try:
                # Load the full rotation matrix
                full_rotation_matrix = np.loadtxt(rot_file)
                
                # Get appropriate sized submatrix if needed
                if full_rotation_matrix.shape[0] >= self.dimension and full_rotation_matrix.shape[1] >= self.dimension:
                    self.rotation_matrix = full_rotation_matrix[:self.dimension, :self.dimension].T # Transpose to match CEC convention
                else:
                    raise ValueError(f"Rotation matrix file contains {full_rotation_matrix.shape} matrix, "
                                    f"but ({self.dimension}, {self.dimension}) is needed")
            except Exception as e:
                raise RuntimeError(f"Failed to load rotation matrix: {e}")
        
        # Load function bias from manifest or use the global bias map
        if "bias" in func_info:
            self.bias = func_info["bias"]
        else:
            # Use the global bias map if available
            self.bias = CEC_2005_BIAS.get(self.function_number, 0.0)
            
        self.metadata["is_biased"] = self.bias != 0.0
    
    @property
    def optimum_value(self) -> float:
        """
        Get the function value at the global optimum. (Alias for bias).
        
        Returns:
            float: Function value at the global optimum.
        """
        return self.bias

    @optimum_value.setter
    def optimum_value(self, value: float):
        """
        Set the function value at the global optimum. (Alias for bias).
        
        This is provided for backward compatibility.
        
        Args:
            value (float): Function value at the global optimum.
        """
        self.bias = value
        self.metadata["is_biased"] = self.bias != 0.0
    
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
        z = x - self.shift_vector
        
        # Apply rotation if needed
        if self.use_rotation:
            z = np.dot(self.rotation_matrix, z)
        
        return z
    
    def _transform_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Apply transformations to a batch of inputs.
        
        This method applies shift and rotation transformations to a batch of inputs.
        
        Args:
            X (np.ndarray): Batch of input vectors, shape (N, dimension).
            
        Returns:
            np.ndarray: Batch of transformed vectors.
        """
        # Apply shift
        Z = X - self.shift_vector
        
        # Apply rotation if needed
        if self.use_rotation:
            # For matrices, we need to rotate each vector individually
            Z_rotated = np.empty_like(Z)
            for i in range(Z.shape[0]):
                Z_rotated[i] = np.dot(self.rotation_matrix, Z[i])
            Z = Z_rotated
        
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
    
    @property
    def shift(self) -> np.ndarray:
        """
        Get the shift vector.
        
        Returns:
            np.ndarray: The shift vector.
        """
        return self.shift_vector
    
    @property
    def rotation(self) -> np.ndarray:
        """
        Get the rotation matrix.
        
        Returns:
            np.ndarray: The rotation matrix.
        """
        return self.rotation_matrix
    
    def get_metadata(self) -> dict:
        """
        Get metadata about the function.
        
        Returns:
            dict: Dictionary containing metadata about the function.
        """
        return self.metadata
    
    def __call__(self, x: np.ndarray) -> float:
        """
        Make the function callable.
        
        Args:
            x (np.ndarray): Input vector.
            
        Returns:
            float: Function value.
        """
        return self.evaluate(x)


# Implementation of F01: Shifted Sphere Function
class F01(CEC2005Function):
    """
    F01: Shifted Sphere Function from CEC 2005.
    
    f(x) = sum(z_i^2) + bias
    where z = x - o
    
    Global optimum: f(o) = bias = -450
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Bounds for each dimension, default is [-100, 100].
        shift_vector (np.ndarray): The shift vector o.
        bias (float): The bias value (-450.0).
        base_func (OptimizationFunction): The composed function using decorators.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, data_dir: str = None):
        """
        Initialize the F01 function (Shifted Sphere Function).
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          If None, uses the default bounds [-100, 100].
            data_dir (str, optional): Directory containing the CEC 2005 data files.
                                     If None, uses the default directory in the package.
        """
        super().__init__(dimension, 1, bounds, data_dir)
        
        # Define function-specific metadata
        self.metadata.update({
            "name": "F01 - Shifted Sphere Function",
            "is_shifted": True,
            "is_biased": True,
            "is_unimodal": True,
            "is_multimodal": False,
            "is_separable": True
        })
        
        # Set bias value from the CEC_2005_BIAS mapping constant
        self.bias = CEC_2005_BIAS[1]
        
        # Create base function using decorators according to CEC 2005 specifications
        # 1. Create base Sphere function
        func = SphereFunction(dimension, bounds)
        
        # 2. Apply shift transformation using the loaded shift vector
        shifted_func = ShiftedFunction(func, self.shift_vector)
        
        # 3. Apply bias transformation
        self.base_func = BiasedFunction(shifted_func, self.bias)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the function at point x.
        
        Args:
            x (np.ndarray): Input vector of dimension D.
            
        Returns:
            float: Function value at x.
        """
        x = self._validate_input(x)
        
        # Use the composed function with decorators
        return self.base_func.evaluate(x)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the function at multiple points.
        
        Args:
            X (np.ndarray): Input matrix of shape (N, D).
            
        Returns:
            np.ndarray: Function values at each point, shape (N,).
        """
        X = self._validate_batch_input(X)
        
        # Use the composed function with decorators
        return self.base_func.evaluate_batch(X)


# Implementation of F02: Shifted Schwefel's Problem 1.2
class F02(CEC2005Function):
    """
    F02: Shifted Schwefel's Problem 1.2 from CEC 2005.
    
    f(x) = sum( (sum(z_j, j=1..i))^2 ) + bias
    where z = x - o
    
    Global optimum: f(o) = bias = -450
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Bounds for each dimension, default is [-100, 100].
        shift_vector (np.ndarray): The shift vector o.
        bias (float): The bias value (-450.0).
        base_func (OptimizationFunction): The composed function using decorators.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, data_dir: str = None):
        """
        Initialize the F02 function (Shifted Schwefel's Problem 1.2).
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          If None, uses the default bounds [-100, 100].
            data_dir (str, optional): Directory containing the CEC 2005 data files.
                                     If None, uses the default directory in the package.
        """
        super().__init__(dimension, 2, bounds, data_dir)
        
        # Define function-specific metadata
        self.metadata.update({
            "name": "F02 - Shifted Schwefel's Problem 1.2",
            "is_shifted": True,
            "is_biased": True,
            "is_unimodal": True,
            "is_multimodal": False,
            "is_separable": False
        })
        
        # Set bias value from the CEC_2005_BIAS mapping constant
        self.bias = CEC_2005_BIAS[2]
        
        # Create base function using decorators according to CEC 2005 specifications
        # 1. Create base Schwefel 1.2 function
        func = SchwefelFunction12(dimension)
        
        # 2. Apply shift transformation using the loaded shift vector
        shifted_func = ShiftedFunction(func, self.shift_vector)
        
        # 3. Apply bias transformation
        self.base_func = BiasedFunction(shifted_func, self.bias)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the function at point x.
        
        Args:
            x (np.ndarray): Input vector of dimension D.
            
        Returns:
            float: Function value at x.
        """
        x = self._validate_input(x)
        
        # Use the composed function with decorators
        return self.base_func.evaluate(x)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the function at multiple points.
        
        Args:
            X (np.ndarray): Input matrix of shape (N, D).
            
        Returns:
            np.ndarray: Function values at each point, shape (N,).
        """
        X = self._validate_batch_input(X)
        
        # Use the composed function with decorators
        return self.base_func.evaluate_batch(X)


# Implementation of F03: Shifted Rotated High Conditioned Elliptic Function
class F03(CEC2005Function):
    """
    F03: Shifted Rotated High Conditioned Elliptic Function from CEC 2005.
    
    f(x) = sum((10^6)^((i-1)/(D-1)) * z_i^2) + bias
    where z = M * (x - o), M is an orthogonal matrix
    
    Global optimum: f(o) = bias = -450
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Bounds for each dimension, default is [-100, 100].
        shift_vector (np.ndarray): The shift vector o.
        rotation_matrix (np.ndarray): The rotation matrix M.
        bias (float): The bias value (-450.0).
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, data_dir: str = None):
        """
        Initialize the F03 function (Shifted Rotated High Conditioned Elliptic Function).
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          If None, uses the default bounds [-100, 100].
            data_dir (str, optional): Directory containing the CEC 2005 data files.
                                     If None, uses the default directory in the package.
        """
        super().__init__(dimension, 3, bounds, data_dir, use_rotation=True)
        
        # Define function-specific metadata
        self.metadata.update({
            "name": "F03 - Shifted Rotated High Conditioned Elliptic Function",
            "is_shifted": True,
            "is_rotated": True,
            "is_biased": True,
            "is_unimodal": True,
            "is_multimodal": False,
            "is_separable": False
        })
        
        # Set bias value from the CEC_2005_BIAS mapping constant
        self.bias = CEC_2005_BIAS[3]

        # Start with the base EllipticFunction
        func = HighConditionedElliptic(dimension)

        # Apply combined shift-then-rotate transformation matching the CEC C code sequence
        shifted_rotated_func = ShiftThenRotateFunction(func, self.shift_vector, self.rotation_matrix)

        # Apply BiasedFunction decorator to add the bias
        self.base_func = BiasedFunction(shifted_rotated_func, self.bias)

    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the function at point x.
        
        Args:
            x (np.ndarray): Input vector of dimension D.
            
        Returns:
            float: Function value at x.
        """
        x = self._validate_input(x)

        # Use the composed function with decorators
        return self.base_func.evaluate(x)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the function at multiple points.
        
        Args:
            X (np.ndarray): Input matrix of shape (N, D).
            
        Returns:
            np.ndarray: Function values at each point, shape (N,).
        """
        X = self._validate_batch_input(X)

        # Use the composed function with decorators
        return self.base_func.evaluate_batch(X)


# Implementation of F04: Shifted Schwefel's Problem 1.2 with Noise in Fitness
class F04(CEC2005Function):
    """
    F04: Shifted Schwefel's Problem 1.2 with Noise in Fitness from CEC 2005.
    
    f(x) = (sum( (sum(z_j, j=1..i))^2 )) * (1 + 0.4|N(0,1)|) + bias
    where z = x - o, N(0,1) is a Gaussian random variable
    
    Global optimum: f(o) = bias = -450
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Bounds for each dimension, default is [-100, 100].
        shift_vector (np.ndarray): The shift vector o.
        bias (float): The bias value (-450.0).
        base_func (OptimizationFunction): The composed function using decorators.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, data_dir: str = None):
        """
        Initialize the F04 function (Shifted Schwefel's Problem 1.2 with Noise).
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          If None, uses the default bounds [-100, 100].
            data_dir (str, optional): Directory containing the CEC 2005 data files.
                                     If None, uses the default directory in the package.
        """
        super().__init__(dimension, 4, bounds, data_dir)
        
        # Define function-specific metadata
        self.metadata.update({
            "name": "F04 - Shifted Schwefel's Problem 1.2 with Noise in Fitness",
            "is_shifted": True,
            "is_biased": True,
            "is_unimodal": True,
            "is_multimodal": False,
            "is_separable": False,
            "has_noise": True
        })
        
        # Set bias value from the CEC_2005_BIAS mapping constant
        self.bias = CEC_2005_BIAS[4]
        
        # Create base function using decorators according to CEC 2005 specifications
        # 1. Create base Schwefel 1.2 function
        func = SchwefelFunction12(dimension, bounds)
        
        # 2. Apply shift transformation using the loaded shift vector
        shifted_func = ShiftedFunction(func, self.shift_vector)
        
        # 3. Apply noise transformation with level 0.4
        # The CEC noise is defined as 1 + 0.4|N(0,1)| which matches the NoiseDecorator implementation
        noisy_shifted_func = NoiseDecorator(shifted_func, noise_type='gaussian', noise_level=0.4)
        
        # 4. Apply bias transformation
        self.base_func = BiasedFunction(noisy_shifted_func, self.bias)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the function at point x.
        
        Args:
            x (np.ndarray): Input vector of dimension D.
            
        Returns:
            float: Function value at x.
        """
        x = self._validate_input(x)
        
        # Use the composed function with decorators
        return self.base_func.evaluate(x)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the function at multiple points.
        
        Args:
            X (np.ndarray): Input matrix of shape (N, D).
            
        Returns:
            np.ndarray: Function values at each point, shape (N,).
        """
        X = self._validate_batch_input(X)
        
        # Use the composed function with decorators
        return self.base_func.evaluate_batch(X)


# Implementation of F05: Schwefel's Problem 2.6 with Global Optimum on Bounds
class F05(CEC2005Function):
    """
    F05: Schwefel's Problem 2.6 with Global Optimum on Bounds.
    
    f(x) = max(|A_i*x - B_i|) + bias
    where B_i = A_i*o, and o is the shifted global optimum.
    
    Global optimum: f(o) = bias = -310
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Bounds for each dimension, default is [-100, 100].
        shift_vector (np.ndarray): The shift vector o.
        A_matrix (np.ndarray): A matrix of size dimension x dimension.
        B_vector (np.ndarray): B vector of size dimension, calculated as B = A*o.
        bias (float): The bias value (-310.0).
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, data_dir: str = None):
        """
        Initialize the F05 function (Schwefel's Problem 2.6 with Global Optimum on Bounds).
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          If None, uses the default bounds [-100, 100].
            data_dir (str, optional): Directory containing the CEC 2005 data files.
                                     If None, uses the default directory in the package.
        """
        # Initialize parent class
        super().__init__(dimension, 5, bounds, data_dir)
        
        # Define function-specific metadata
        self.metadata.update({
            "name": "F05 - Schwefel's Problem 2.6 with Global Optimum on Bounds",
            "is_shifted": True,
            "is_biased": True,
            "is_unimodal": True, 
            "is_multimodal": False,
            "is_separable": False
        })
        
        # Set bias value from Schwefel's problem (negative to make global minimum)
        self.bias = CEC_2005_BIAS[5]
        
        # Try to load data from f5_data_dump
        import os
        import numpy as np
        from pathlib import Path
        
        # Determine the project root directory
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        f5_dump_dir = os.path.join(project_root, "utility_scripts", "f5_data_dump")
        
        # Try to load shift vector from dump
        shift_file = os.path.join(f5_dump_dir, f"shift_vector_D{dimension}.txt")
        if os.path.exists(shift_file):
            try:
                # Load the shift vector directly
                self.shift_vector = np.loadtxt(shift_file)
                print(f"Loaded shift vector from f5_data_dump: {shift_file}")
            except Exception as e:
                print(f"Failed to load shift vector from f5_data_dump, using default. Error: {e}")
        
        # Adjust shift vector to bounds (needed regardless of source)
        self._adjust_shift_vector_to_bounds()
        
        # Try to load A matrix from dump
        a_file = os.path.join(f5_dump_dir, f"A_matrix_D{dimension}.txt")
        if os.path.exists(a_file):
            try:
                # Load the A matrix directly
                self.A_matrix = np.loadtxt(a_file)
                print(f"Loaded A matrix from f5_data_dump: {a_file}")
            except Exception as e:
                print(f"Failed to load A matrix from f5_data_dump, using default. Error: {e}")
                # Fall back to the standard loading method
                self.A_matrix = self._load_A_matrix_fallback()
        else:
            # Fall back to the standard loading method
            self.A_matrix = self._load_A_matrix_fallback()
        
        # Calculate B vector as B = A*o
        self.B_vector = np.dot(self.A_matrix, self.shift_vector)
    
    def _adjust_shift_vector_to_bounds(self):
        """
        Adjust the shift vector to match the original implementation.
        
        Based on the CEC 2005 paper and original code:
        - Quarter of the variables are set to the lower bound (-100)
        - Quarter of the variables are set to the upper bound (100)
        """
        # First quarter set to lower bound (-100)
        index = self.dimension // 4
        if index < 1:  # Handle edge case for very small dimensions
            index = 1
        for i in range(index):
            self.shift_vector[i] = -100.0
        
        # Last quarter set to upper bound (100)
        index = (3 * self.dimension) // 4 - 1
        if index < 0:  # Handle edge case for very small dimensions
            index = 0
        for i in range(index, self.dimension):
            self.shift_vector[i] = 100.0
    
    def _load_A_matrix_fallback(self):
        """
        Fallback method to load the A matrix if it cannot be loaded from f5_data_dump.
        
        This method tries to load the A matrix from:
        1. The shift_D50.txt file (original implementation)
        2. Generate a random matrix as fallback
        
        Returns:
            np.ndarray: The A matrix of size dimension x dimension.
        """
        import os
        import numpy as np
        
        # Try original implementation as fallback
        potential_paths = [
            os.path.join(self.data_dir, "f05", "shift_D50.txt"),  # Standard path in our repo
            os.path.join(self.data_dir, "schwefel_206_data.txt")  # Original CEC path
        ]
        
        shift_file = None
        for path in potential_paths:
            if os.path.exists(path):
                shift_file = path
                break
        
        if shift_file is None:
            print(f"Warning: Could not find data file for F05, using random matrix instead.")
            return self._generate_A_matrix()
        
        try:
            # Read the file line by line, matching the original C implementation
            with open(shift_file, 'r') as f:
                lines = f.readlines()
            
            # First read the shift vectors (nfunc lines)
            num_funcs = min(len(lines), 25)  # CEC 2005 has 25 functions max
            
            # Then read the A matrix (dimension lines after the shift vectors)
            A_matrix = np.zeros((self.dimension, self.dimension))
            for i in range(min(self.dimension, len(lines) - num_funcs)):
                # Parse the line for the A matrix row
                line = lines[num_funcs + i]
                values = np.array([float(x) for x in line.strip().split()])
                
                # Fill the A matrix row
                A_matrix[i, :min(self.dimension, len(values))] = values[:self.dimension]
            
            return A_matrix
                
        except Exception as e:
            print(f"Warning: Could not load A matrix from file, using random matrix instead. Error: {e}")
            return self._generate_A_matrix()
    
    def _generate_A_matrix(self):
        """
        Generate A matrix with random integer values and non-zero determinant.
        This is used as a fallback if the data file cannot be loaded.
        """
        while True:
            A = np.random.randint(-100, 101, size=(self.dimension, self.dimension))
            if np.linalg.det(A) != 0:
                return A
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the function at point x.
        
        Direct implementation based on the original C code:
        f(x) = max(|A_i*x - B_i|) + bias
        where B_i = A_i*o
        
        Args:
            x (np.ndarray): Input vector of dimension D.
            
        Returns:
            float: Function value at x.
        """
        x = self._validate_input(x)
        
        # Calculate A*x
        Ax = np.dot(self.A_matrix, x)
        
        # Calculate |A*x - B| for each row
        abs_diff = np.abs(Ax - self.B_vector)
        
        # Get maximum value as per original C code
        max_diff = np.max(abs_diff)
        
        # Return max value + bias
        return float(max_diff) + self.bias
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the function at multiple points.
        
        Args:
            X (np.ndarray): Input matrix of shape (N, D).
            
        Returns:
            np.ndarray: Function values at each point, shape (N,).
        """
        X = self._validate_batch_input(X)
        results = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            # Calculate A*x for each input point
            Ax = np.dot(self.A_matrix, X[i])
            
            # Calculate |A*x - B| for each row
            abs_diff = np.abs(Ax - self.B_vector)
            
            # Get maximum value
            results[i] = np.max(abs_diff)
        
        # Add bias to all results
        return results + self.bias


# Implementation of F06: Shifted Rosenbrock's Function
class F06(CEC2005Function):
    """
    F06: Shifted Rosenbrock's Function from CEC 2005.
    
    f(x) = sum(100(z_i^2 - z_{i+1})^2 + (z_i - 1)^2) + bias
    where z = x - o + 1
    
    For Rosenbrock function, the standard optimum is at [1,1,...,1], 
    but in CEC2005 it is shifted to be at the specific shift vector.
    
    Global optimum: f(o) = bias = 390
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Bounds for each dimension, default is [-100, 100].
        shift_vector (np.ndarray): The shift vector o.
        bias (float): The bias value (390.0).
        base_func (OptimizationFunction): The composed function using decorators.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, data_dir: str = None):
        """
        Initialize the F06 function (Shifted Rosenbrock's Function).
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          If None, uses the default bounds [-100, 100].
            data_dir (str, optional): Directory containing the CEC 2005 data files.
                                     If None, uses the default directory in the package.
        """
        super().__init__(dimension, 6, bounds, data_dir)
        
        # Define function-specific metadata
        self.metadata.update({
            "name": "F06 - Shifted Rosenbrock's Function",
            "is_shifted": True,
            "is_biased": True,
            "is_unimodal": False,  # Rosenbrock is actually multimodal for D > 3
            "is_multimodal": True,
            "is_separable": False
        })
        
        # Set bias value from the CEC_2005_BIAS mapping constant
        self.bias = CEC_2005_BIAS[6]
        
        # Create base function using decorators according to CEC 2005 specifications
        # 1. Create base Rosenbrock function
        from ..unimodal import RosenbrockFunction
        func = RosenbrockFunction(dimension, bounds)
        
        # 2. First apply a shift of -1 to all dimensions 
        # This handles the fact that Rosenbrock's natural optimum is at [1,1,...,1]
        # and we need to move it to [0,0,...,0] before applying the CEC shift
        ones_vector = np.ones(dimension)
        centered_func = ShiftedFunction(func, -ones_vector)
        
        # 3. Then apply the actual CEC shift to move the optimum to the required position
        shifted_func = ShiftedFunction(centered_func, self.shift_vector)
        
        # 4. Apply bias transformation
        self.base_func = BiasedFunction(shifted_func, self.bias)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the function at point x.
        
        Args:
            x (np.ndarray): Input vector of dimension D.
            
        Returns:
            float: Function value at x.
        """
        x = self._validate_input(x)
        
        # Use the composed function with decorators
        return self.base_func.evaluate(x)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the function at multiple points.
        
        Args:
            X (np.ndarray): Input matrix of shape (N, D).
            
        Returns:
            np.ndarray: Function values at each point, shape (N,).
        """
        X = self._validate_batch_input(X)
        
        # Use the composed function with decorators
        return self.base_func.evaluate_batch(X)


# Implementation of F07: Shifted Rotated Griewank's Function without Bounds
class F07(CEC2005Function):
    """
    F07: Shifted Rotated Griewank's Function without Bounds from CEC 2005.
    
    f(x) = sum(z_i^2/4000) - prod(cos(z_i/sqrt(i+1))) + 1 + bias
    where z = M * (x - o), M is a rotation matrix
    
    Global optimum: f(o) = bias = -180
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): No bounds, but initialization is in [0, 600].
        shift_vector (np.ndarray): The shift vector o.
        rotation_matrix (np.ndarray): The rotation matrix M.
        bias (float): The bias value (-180.0).
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, data_dir: str = None):
        """
        Initialize the F07 function.
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Traditionally unbounded, but initialization in [0, 600].
            data_dir (str, optional): Directory containing the CEC 2005 data files.
        """
        # For F07, there are no bounds, but initialization is in [0, 600]
        if bounds is None:
            # Using much larger bounds as the function is unbounded in the original specification
            bounds = np.array([[-1000, 1000]] * dimension)

        super().__init__(dimension, function_number=7, bounds=bounds, data_dir=data_dir, use_rotation=True)
        
        # Define function-specific metadata
        self.metadata.update({
            "name": "F07 - Shifted Rotated Griewank's Function without Bounds",
            "is_shifted": True,
            "is_rotated": True,
            "is_biased": True,
            "is_unimodal": False,
            "is_multimodal": True,
            "is_separable": False
        })
        
        # Set bias value from the CEC_2005_BIAS mapping constant
        self.bias = CEC_2005_BIAS[7]

        # Create GriewankCore function - this is the base implementation without shift/rotate/bias
        func = GriewankFunction(dimension)

        # Apply combined shift-then-rotate transformation matching the CEC C code sequence
        shifted_rotated_func = ShiftThenRotateFunction(func, self.shift_vector, self.rotation_matrix)

        # Apply BiasedFunction decorator to add the bias
        self.base_func = BiasedFunction(shifted_rotated_func, self.bias)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the function at point x.
        
        Args:
            x (np.ndarray): Input vector of dimension D.
            
        Returns:
            float: Function value at x.
        """
        x = self._validate_input(x)
        
        # Use the composed function with decorators
        return self.base_func.evaluate(x)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the function at multiple points.
        
        Args:
            X (np.ndarray): Input matrix of shape (N, D).
            
        Returns:
            np.ndarray: Function values at each point, shape (N,).
        """
        X = self._validate_batch_input(X)
        
        # Use the composed function with decorators
        return self.base_func.evaluate_batch(X)


# Implementation of F08: Shifted Rotated Ackley's Function with Global Optimum on Bounds
class F08(CEC2005Function):
    """
    F08: Shifted Rotated Ackley's Function with Global Optimum on Bounds from CEC 2005.
    
    f(x) = -20*exp(-0.2*sqrt(sum(z_i^2)/D)) - exp(sum(cos(2π*z_i))/D) + 20 + e + bias
    where z = M * (x - o), M is a rotation matrix
    
    Global optimum: f(o) = bias = -140
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Bounds for each dimension, default is [-32, 32].
        shift_vector (np.ndarray): The shift vector o.
        rotation_matrix (np.ndarray): The rotation matrix M.
        bias (float): The bias value (-140.0).
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, data_dir: str = None):
        """
        Initialize the F08 function.
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          Defaults to [-32, 32] for each dimension.
            data_dir (str, optional): Directory containing the CEC 2005 data files.
        """
        # Ackley's function has different bounds: [-32, 32]
        if bounds is None:
            bounds = np.array([[-32, 32]] * dimension)

        super().__init__(dimension, function_number=8, bounds=bounds, data_dir=data_dir, use_rotation=True)
        
        # Set bias value from the CEC_2005_BIAS mapping constant
        self.bias = CEC_2005_BIAS[8]
        
        # This is copied from the CEC C code initialization of the shift vector
        # For F08, global optimum is partially on bounds
        # Set even indexed elements to the lower bound (-32)
        for i in range(0, self.dimension, 2):
            if i < self.dimension:
                self.shift_vector[i] = -32
        
        # Define function-specific metadata
        self.metadata.update({
            "name": "F08 - Shifted Rotated Ackley's Function with Global Optimum on Bounds",
            "is_shifted": True,
            "is_rotated": True,
            "is_biased": True,
            "is_unimodal": False,
            "is_multimodal": True,
            "is_separable": False,
            "global_optimum_on_bounds": True
        })
        
        # Create AckleyFunction - this is the base implementation without shift/rotate/bias
        func = AckleyFunction(dimension)

        # Apply combined shift-then-rotate transformation matching the CEC C code sequence
        shifted_rotated_func = ShiftThenRotateFunction(func, self.shift_vector, self.rotation_matrix)

        # Apply BiasedFunction decorator to add the bias
        self.base_func = BiasedFunction(shifted_rotated_func, self.bias)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the function at point x.
        
        Args:
            x (np.ndarray): Input vector of dimension D.
            
        Returns:
            float: Function value at x.
        """
        x = self._validate_input(x)
        
        # Use the composed function with decorators
        return self.base_func.evaluate(x)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the function at multiple points.
        
        Args:
            X (np.ndarray): Input matrix of shape (N, D).
            
        Returns:
            np.ndarray: Function values at each point, shape (N,).
        """
        X = self._validate_batch_input(X)
        
        # Use the composed function with decorators
        return self.base_func.evaluate_batch(X)


# Implementation of F09: Shifted Rastrigin's Function
class F09(CEC2005Function):
    """
    F09: Shifted Rastrigin's Function from CEC 2005.
    
    f(x) = sum(z_i^2 - 10*cos(2π*z_i) + 10) + bias
    where z = x - o
    
    Global optimum: f(o) = bias = -330
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Bounds for each dimension, default is [-5, 5].
        shift_vector (np.ndarray): The shift vector o.
        bias (float): The bias value (-330.0).
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, data_dir: str = None):
        """
        Initialize the F09 function.
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          Defaults to [-5, 5] for each dimension.
            data_dir (str, optional): Directory containing the CEC 2005 data files.
        """
        # Rastrigin's function has different bounds: [-5, 5]
        if bounds is None:
            bounds = np.array([[-5, 5]] * dimension)
        
        self.use_rotation = False
        super().__init__(dimension, function_number=9, bounds=bounds, data_dir=data_dir)
        self.bias = -330.0
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the F09 function at point x.
        
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
        return float(np.sum(z**2 - 10.0 * np.cos(2 * np.pi * z) + 10.0)) + self.bias
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the F09 function on a batch of points.
        
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
        return np.sum(Z**2 - 10.0 * np.cos(2 * np.pi * Z) + 10.0, axis=1) + self.bias


# Implementation of F10: Shifted Rotated Rastrigin's Function
class F10(CEC2005Function):
    """
    F10: Shifted Rotated Rastrigin's Function from CEC 2005.
    
    f(x) = sum(z_i^2 - 10*cos(2π*z_i) + 10) + bias
    where z = (x - o) * M, M is a rotation matrix
    
    Global optimum: f(o) = bias = -330
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Bounds for each dimension, default is [-5, 5].
        shift_vector (np.ndarray): The shift vector o.
        rotation_matrix (np.ndarray): The rotation matrix M.
        bias (float): The bias value (-330.0).
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, data_dir: str = None):
        """
        Initialize the F10 function.
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          Defaults to [-5, 5] for each dimension.
            data_dir (str, optional): Directory containing the CEC 2005 data files.
        """
        # Rastrigin's function has different bounds: [-5, 5]
        if bounds is None:
            bounds = np.array([[-5, 5]] * dimension)
        
        super().__init__(dimension, function_number=10, bounds=bounds, data_dir=data_dir, use_rotation=True)
        self.bias = CEC_2005_BIAS[10]
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the F10 function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Apply transformations (shift and rotation)
        z = self._transform_input(x)
        
        # Compute function value
        return float(np.sum(z**2 - 10.0 * np.cos(2 * np.pi * z) + 10.0)) + self.bias
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the F10 function on a batch of points.
        
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
        return np.sum(Z**2 - 10.0 * np.cos(2 * np.pi * Z) + 10.0, axis=1) + self.bias


# Implementation of F11: Shifted Rotated Weierstrass Function
class F11(CEC2005Function):
    """
    F11: Shifted Rotated Weierstrass Function from CEC 2005.
    
    f(x) = sum(sum(a^k * cos(2π * b^k * (z_i + 0.5)))) - D*sum(a^k * cos(2π * b^k * 0.5)) + bias
    f(x) = sum(sum(a^k * cos(2π * b^k * (z_i + 0.5)))) - D*sum(a^k * cos(2π * b^k * 0.5)) + f_bias
    where z = (x - o) * M, a = 0.5, b = 3, k_max = 20
    
    Global optimum: f(o) = f_bias = 90
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Bounds for each dimension, default is [-0.5, 0.5].
        shift_vector (np.ndarray): The shift vector o.
        rotation_matrix (np.ndarray): The rotation matrix M.
        a (float): Constant a = 0.5.
        b (float): Constant b = 3.
        k_max (int): Maximum value for k, k_max = 20.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, data_dir: str = None):
        """
        Initialize the F11 function.
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          Defaults to [-0.5, 0.5] for each dimension.
            data_dir (str, optional): Directory containing the CEC 2005 data files.
        """
        # Weierstrass function has different bounds: [-0.5, 0.5]
        if bounds is None:
            bounds = np.array([[-0.5, 0.5]] * dimension)
        
        super().__init__(dimension, function_number=11, bounds=bounds, data_dir=data_dir, use_rotation=True)
        self.bias = CEC_2005_BIAS[11]
        
        # Set constants
        self.a = 0.5
        self.b = 3.0
        self.k_max = 20
        
        # Precompute powers and cosine terms for second sum
        self.a_k = np.power(self.a, range(self.k_max + 1))
        self.b_k = np.power(self.b, range(self.k_max + 1))
        self.cos_term = np.sum(self.a_k * np.cos(2 * np.pi * self.b_k * 0.5))
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the F11 function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Apply transformations (shift and rotation)
        z = self._transform_input(x)
        
        # First sum (over dimensions)
        sum_i = 0.0
        for i in range(self.dimension):
            # Second sum (over k)
            sum_k = 0.0
            for k in range(self.k_max + 1):
                sum_k += self.a_k[k] * np.cos(2 * np.pi * self.b_k[k] * (z[i] + 0.5))
            sum_i += sum_k
        
        # Subtract the constant term
        result = sum_i - self.dimension * self.cos_term
        
        return float(result) + self.bias
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the F11 function on a batch of points.
        
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
            
            # First sum (over dimensions)
            sum_i = 0.0
            for j in range(self.dimension):
                # Second sum (over k)
                sum_k = 0.0
                for k in range(self.k_max + 1):
                    sum_k += self.a_k[k] * np.cos(2 * np.pi * self.b_k[k] * (z[j] + 0.5))
                sum_i += sum_k
            
            # Subtract the constant term
            result[i] = sum_i - self.dimension * self.cos_term
        
        return result + self.bias


# Implementation of F12: Schwefel's Problem 2.13
class F12(CEC2005Function):
    """
    F12: Schwefel's Problem 2.13 from CEC 2005.
    
    f(x) = sum((A_i - B_i(x))^2) + f_bias
    where A_i = sum(a_ij*sin(α_j) + b_ij*cos(α_j)) for j=1 to D
    and B_i(x) = sum(a_ij*sin(x_j) + b_ij*cos(x_j)) for j=1 to D
    
    Global optimum: f(o) = f_bias = -460
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Bounds for each dimension, default is [-π, π].
        alpha (np.ndarray): The α vector (used as shift vector in other functions).
        a (np.ndarray): Matrix a.
        b (np.ndarray): Matrix b.
        A (np.ndarray): Vector A.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, data_dir: str = None):
        """
        Initialize the F12 function.
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          Defaults to [-π, π] for each dimension.
            data_dir (str, optional): Directory containing the CEC 2005 data files.
        """
        # Schwefel's Problem 2.13 has different bounds: [-π, π]
        if bounds is None:
            bounds = np.array([[-np.pi, np.pi]] * dimension)
        
        super().__init__(dimension, function_number=12, bounds=bounds, data_dir=data_dir)
        self.bias = CEC_2005_BIAS[12]
        
        # Define function-specific metadata
        self.metadata.update({
            "name": "F12 - Schwefel's Problem 2.13",
            "is_shifted": True,  # Uses alpha as shift vector
            "is_biased": True,
            "is_unimodal": False,
            "is_multimodal": True,
            "is_separable": False  # Not separable due to matrix operations
        })
        
        # Initialize matrices and vectors
        self.alpha = None
        self.a = None
        self.b = None
        self.A = None
        self._initialize_matrices()
    
    def _load_function_parameters(self):
        """
        Load function-specific parameters.
        
        F12 doesn't use the standard shift/rotation pattern, so we override this method.
        This method also loads and stores the specific file paths for alpha, A, and B matrices.
        """
        # Load manifest file to get the base data directory
        import os
        import json
        
        if not self.data_dir or not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"CEC2005 data directory not found: {self.data_dir}")
        
        # Load manifest file
        manifest_path = os.path.join(self.data_dir, "meta_2005.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"CEC2005 manifest file not found: {manifest_path}")
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load CEC2005 manifest file: {e}")
        
        # Get function information from manifest
        func_key = f"f{self.function_number:02d}"
        
        if "functions" not in manifest or func_key not in manifest["functions"]:
            raise ValueError(f"Function {func_key} not found in manifest")
        
        func_info = manifest["functions"][func_key]
        
        # Load bias value from manifest or use the global bias map
        self.bias = CEC_2005_BIAS.get(self.function_number, 0.0)
        self.metadata["is_biased"] = self.bias != 0.0

        # Store file paths for alpha, A, and B matrices
        files_info = func_info.get("files", {})
        if not all(k in files_info for k in ["alpha", "A", "B"]):
            raise FileNotFoundError(
                f"Missing alpha, A, or B file paths in manifest for {func_key}. "
                f"Ensure 'alpha', 'A', and 'B' keys exist under 'files' for {func_key} in meta_2005.json."
            )

        self._alpha_vector_path = os.path.join(self.data_dir, files_info["alpha"])
        self._a_matrix_path = os.path.join(self.data_dir, files_info["A"])
        self._b_matrix_path = os.path.join(self.data_dir, files_info["B"])

        # Ensure these files exist before proceeding to _initialize_matrices
        for path, name in [
            (self._alpha_vector_path, "Alpha vector"),
            (self._a_matrix_path, "A matrix"),
            (self._b_matrix_path, "B matrix"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} data file not found: {path}")
    
    def _initialize_matrices(self):
        """
        Initialize the matrices for F12 from data files specified in the manifest.
        
        This loads alpha vector, A matrix, and B matrix from their respective _D50.txt
        files and slices them to the current function dimension.
        """
        import numpy as np
        
        dim = self.dimension
        
        try:
            # Load full 50-dimensional data
            # Assumes alpha_D50.txt loads as 1D array (50,)
            # Assumes A_D50.txt and B_D50.txt load as 2D arrays (50,50)
            alpha_full = np.loadtxt(self._alpha_vector_path)
            a_full = np.loadtxt(self._a_matrix_path)
            b_full = np.loadtxt(self._b_matrix_path)

            # Basic validation of loaded shapes (assuming D50 files)
            # More robust checks could compare against a known MAX_DIM if available
            if alpha_full.ndim != 1 or alpha_full.shape[0] < dim:
                raise ValueError(f"Loaded alpha data from {self._alpha_vector_path} has unexpected shape {alpha_full.shape} for dimension {dim}.")
            if a_full.ndim != 2 or a_full.shape[0] < dim or a_full.shape[1] < dim:
                raise ValueError(f"Loaded A matrix data from {self._a_matrix_path} has unexpected shape {a_full.shape} for dimension {dim}.")
            if b_full.ndim != 2 or b_full.shape[0] < dim or b_full.shape[1] < dim:
                raise ValueError(f"Loaded B matrix data from {self._b_matrix_path} has unexpected shape {b_full.shape} for dimension {dim}.")

            # Slice to current dimension
            self.alpha = alpha_full[:dim]
            self.a = a_full[:dim, :dim]
            self.b = b_full[:dim, :dim]
            
            # Use alpha as the shift vector (as per existing logic for F12)
            self.shift_vector = self.alpha.copy()
            
            # Compute A vector (dependent on self.alpha, self.a, self.b)
            self._compute_A_vector()
            
        except FileNotFoundError: # Should be caught by _load_function_parameters, but as safety
            raise 
        except Exception as e:
            # Catch other potential errors during loadtxt or slicing
            raise RuntimeError(f"Error processing F12 data files: {e}")
    
    def _compute_A_vector(self):
        """Compute the A vector based on alpha, a, and b matrices."""
        self.A = np.zeros(self.dimension)
        for i in range(self.dimension):
            sum_val = 0.0
            for j in range(self.dimension):
                sum_val += (self.a[i, j] * np.sin(self.alpha[j]) + 
                           self.b[i, j] * np.cos(self.alpha[j]))
            self.A[i] = sum_val
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the F12 function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Compute B(x) vector
        B = np.zeros(self.dimension)
        for i in range(self.dimension):
            sum_val = 0.0
            for j in range(self.dimension):
                sum_val += (self.a[i, j] * np.sin(x[j]) + 
                           self.b[i, j] * np.cos(x[j]))
            B[i] = sum_val
        
        # Compute sum of squared differences
        result = np.sum((self.A - B)**2)
        
        return float(result) + self.bias
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the F12 function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        # Compute function values
        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            x = X[i]
            
            # Compute B(x) vector
            B = np.zeros(self.dimension)
            for j in range(self.dimension):
                sum_val = 0.0
                for k in range(self.dimension):
                    sum_val += (self.a[j, k] * np.sin(x[k]) + 
                              self.b[j, k] * np.cos(x[k]))
                B[j] = sum_val
            
            # Compute sum of squared differences
            result[i] = np.sum((self.A - B)**2)
        
        return result + self.bias


# Implementation of F13: Shifted Expanded Griewank's plus Rosenbrock's Function
class F13(CEC2005Function):
    """
    F13: Shifted Expanded Griewank's plus Rosenbrock's Function from CEC 2005.
    
    This is an expanded function F8F2, where:
    F8: Griewank's Function
    F2: Rosenbrock's Function
    
    The expansion pattern is:
    F(x1,x2,...,xD) = F8(F2(x1,x2)) + F8(F2(x2,x3)) + ... + F8(F2(xD-1,xD)) + F8(F2(xD,x1))
    
    Global optimum: f(o) = f_bias = -130
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Bounds for each dimension, default is [-5, 5].
        shift_vector (np.ndarray): The shift vector o.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, data_dir: str = None):
        """
        Initialize the F13 function.
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          Defaults to [-5, 5] for each dimension.
            data_dir (str, optional): Directory containing the CEC 2005 data files.
        """
        # Different bounds: [-5, 5]
        if bounds is None:
            bounds = np.array([[-5, 5]] * dimension)
        
        self.use_rotation = False
        super().__init__(dimension, function_number=13, bounds=bounds, data_dir=data_dir)
        self.bias = -130.0
    
    def _transform_input(self, x: np.ndarray) -> np.ndarray:
        """
        Apply transformations to input.
        
        For this expanded function, we need to shift by o and add 1.
        
        Args:
            x (np.ndarray): Input vector.
            
        Returns:
            np.ndarray: Transformed vector.
        """
        # Shift by o and add 1 (z = x - o + 1)
        return x - self.shift_vector + 1
    
    def _transform_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Apply transformations to a batch of inputs.
        
        For this expanded function, we need to shift by o and add 1.
        
        Args:
            X (np.ndarray): Batch of input vectors.
            
        Returns:
            np.ndarray: Transformed batch of vectors.
        """
        # Shift by o and add 1 (Z = X - o + 1)
        return X - self.shift_vector + 1
    
    def _rosenbrock(self, x1: float, x2: float) -> float:
        """
        Compute the 2-D Rosenbrock function F2(x1, x2).
        
        Args:
            x1 (float): First component.
            x2 (float): Second component.
            
        Returns:
            float: F2(x1, x2) value.
        """
        return 100.0 * (x1**2 - x2)**2 + (x1 - 1.0)**2
    
    def _griewank(self, y: float) -> float:
        """
        Compute the 1-D Griewank function F8(y).
        
        Args:
            y (float): Input value.
            
        Returns:
            float: F8(y) value.
        """
        return y**2 / 4000.0 - np.cos(y) + 1.0
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the F13 function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Apply transformation
        z = self._transform_input(x)
        
        # Compute expanded function value
        result = 0.0
        for i in range(self.dimension - 1):
            y = self._rosenbrock(z[i], z[i+1])
            result += self._griewank(y)
        
        # Add the wrap-around term
        y = self._rosenbrock(z[-1], z[0])
        result += self._griewank(y)
        
        return float(result) + self.bias
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the F13 function on a batch of points.
        
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
            for j in range(self.dimension - 1):
                y = self._rosenbrock(z[j], z[j+1])
                value += self._griewank(y)
            
            # Add the wrap-around term
            y = self._rosenbrock(z[-1], z[0])
            value += self._griewank(y)
            
            result[i] = value
        
        return result + self.bias


# Implementation of F14: Shifted Rotated Expanded Scaffer's F6 Function
class F14(CEC2005Function):
    """
    F14: Shifted Rotated Expanded Scaffer's F6 Function from CEC 2005.
    
    The base 2-D Scaffer's F6 function is:
    F(x,y) = 0.5 + (sin^2(sqrt(x^2+y^2)) - 0.5) / (1 + 0.001*(x^2+y^2))^2
    
    The expanded form is:
    F(x1,x2,...,xD) = F(x1,x2) + F(x2,x3) + ... + F(xD-1,xD) + F(xD,x1)
    
    Global optimum: f(o) = f_bias = -300
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Bounds for each dimension, default is [-100, 100].
        shift_vector (np.ndarray): The shift vector o.
        rotation_matrix (np.ndarray): The rotation matrix M.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, data_dir: str = None):
        """
        Initialize the F14 function.
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          Defaults to [-100, 100] for each dimension.
            data_dir (str, optional): Directory containing the CEC 2005 data files.
        """
        self.use_rotation = True
        super().__init__(dimension, function_number=14, bounds=bounds, data_dir=data_dir)
        self.bias = -300.0
    
    def _scaffer_f6(self, x: float, y: float) -> float:
        """
        Compute the 2-D Scaffer's F6 function.
        
        Args:
            x (float): First component.
            y (float): Second component.
            
        Returns:
            float: Scaffer's F6 value.
        """
        num = np.sin(np.sqrt(x**2 + y**2))**2 - 0.5
        denom = (1.0 + 0.001 * (x**2 + y**2))**2
        return 0.5 + num / denom
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the F14 function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Apply transformations (shift and rotation)
        z = self._transform_input(x)
        
        # Compute expanded function value
        result = 0.0
        for i in range(self.dimension - 1):
            result += self._scaffer_f6(z[i], z[i+1])
        
        # Add the wrap-around term
        result += self._scaffer_f6(z[-1], z[0])
        
        return float(result) + self.bias
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the F14 function on a batch of points.
        
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
            for j in range(self.dimension - 1):
                value += self._scaffer_f6(z[j], z[j+1])
            
            # Add the wrap-around term
            value += self._scaffer_f6(z[-1], z[0])
            
            result[i] = value
        
        return result + self.bias


# Implementation of F15: Hybrid Composition Function
class F15(CEC2005Function):
    """
    F15: Hybrid Composition Function from CEC 2005.
    
    This is a composition of 10 different functions with different properties.
    
    The composition formula is:
    F(x) = sum(w_i * [f'_i(((x-o_i)/λ_i)) + bias_i]) + f_bias
    
    where:
    - w_i are weight values for each function
    - f'_i are normalized basic functions
    - o_i are shifted optima
    - λ_i are used to stretch/compress functions
    - bias_i define heights of local optima
    
    Global optimum: f(o_1) = f_bias = 120
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Bounds for each dimension, default is [-5, 5].
        shift_vectors (np.ndarray): Matrix of shift vectors for each function.
        lambda_values (np.ndarray): Stretch/compress coefficients.
        sigma_values (np.ndarray): Coverage range parameters.
        bias_values (np.ndarray): Heights of function optima.
        C (float): Normalization constant.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, data_dir: str = None):
        """
        Initialize the F15 function.
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          Defaults to [-5, 5] for each dimension.
            data_dir (str, optional): Directory containing the CEC 2005 data files.
        """
        # Hybrid composition function has different bounds: [-5, 5]
        if bounds is None:
            bounds = np.array([[-5, 5]] * dimension)
        
        # We'll handle rotation separately for each component function
        self.use_rotation = False
        
        # Initialize parent class - note that we use the base class init
        # and handle some of the attributes (like shift_vector) differently
        super().__init__(dimension, function_number=15, bounds=bounds, data_dir=data_dir)
        self.bias = 120.0
        
        # Number of basic functions
        self.n_func = 10
        
        # Initialize matrices for all component functions
        self.shift_vectors = np.zeros((self.n_func, self.dimension))
        self.rotation_matrices = [np.eye(self.dimension) for _ in range(self.n_func)]
        
        # Set parameters for the composition
        self.lambda_values = np.array([1, 1, 10, 10, 5/60, 5/60, 5/32, 5/32, 5/100, 5/100])
        self.sigma_values = np.ones(self.n_func)  # All sigma = 1 for F15
        self.bias_values = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
        self.C = 2000  # Normalization constant
        
        # Load component functions' parameters
        self._load_composition_parameters()
    
    def _load_function_parameters(self):
        """
        Load function-specific parameters.
        
        For the hybrid composition function, we handle shift vectors differently,
        so we override the parent class method.
        """
        # For F15, we don't need to load a global shift vector
        # We'll load the component functions' shift vectors in _load_composition_parameters
        pass
    
    def _load_composition_parameters(self):
        """
        Load parameters for the component functions of the composition.
        
        This includes shift vectors and rotation matrices for each component function.
        """
        # Load shift vectors for all component functions
        if self.data_dir and os.path.exists(self.data_dir):
            shift_file = os.path.join(self.data_dir, f"f{self.function_number:02d}_o.dat")
            
            if os.path.exists(shift_file):
                try:
                    # Load the full matrix of shift vectors
                    full_matrix = np.loadtxt(shift_file)
                    # Extract the appropriate sized matrix
                    self.shift_vectors = full_matrix[:self.n_func, :self.dimension]
                except Exception as e:
                    print(f"Error loading shift vectors: {e}")
                    # Fall back to random values
                    self._generate_random_shifts()
            else:
                # Fall back to random values
                self._generate_random_shifts()
        else:
            # Generate random shift vectors
            self._generate_random_shifts()
        
        # For F15, all rotation matrices are identity matrices
        # (already initialized in constructor)
    
    def _generate_random_shifts(self):
        """Generate random shift vectors for all component functions."""
        for i in range(self.n_func):
            self.shift_vectors[i] = np.random.uniform(-5, 5, self.dimension)
    
    def _calculate_weights(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate weight values for each component function.
        
        Args:
            x (np.ndarray): The input vector.
            
        Returns:
            np.ndarray: Weight values for each component function.
        """
        weights = np.zeros(self.n_func)
        
        # Calculate raw weights based on distance to each function's optimum
        for i in range(self.n_func):
            # Calculate squared Euclidean distance to optimum
            dist_sq = np.sum((x - self.shift_vectors[i])**2)
            
            # Calculate weight using Gaussian-like formula
            weights[i] = np.exp(-dist_sq / (2 * self.dimension * self.sigma_values[i]**2))
        
        # Find maximum weight
        max_weight = np.max(weights)
        max_idx = np.argmax(weights)
        
        # Adjust weights
        for i in range(self.n_func):
            if i != max_idx:
                weights[i] *= (1.0 - max_weight**10)
        
        # Handle special case - if all weights are zero
        sum_weights = np.sum(weights)
        if sum_weights < 1e-6:
            weights = np.ones(self.n_func) / self.n_func
        else:
            # Normalize weights
            weights /= sum_weights
        
        return weights
    
    def _evaluate_basic_function(self, idx: int, z: np.ndarray) -> float:
        """
        Evaluate a single basic function component.
        
        Args:
            idx (int): Index of the basic function (0-9).
            z (np.ndarray): Transformed input vector.
            
        Returns:
            float: The function value.
        """
        if idx % 2 == 0:  # Functions 1, 3, 5, 7, 9 (zero-indexed as 0, 2, 4, 6, 8)
            # Rastrigin's Function
            return np.sum(z**2 - 10.0 * np.cos(2 * np.pi * z) + 10.0)
        
        elif idx % 2 == 1 and idx < 4:  # Functions 2, 4 (zero-indexed as 1, 3)
            # Weierstrass Function
            a = 0.5
            b = 3.0
            k_max = 20
            
            # Precompute powers and cosine term
            a_k = np.power(a, range(k_max + 1))
            b_k = np.power(b, range(k_max + 1))
            cos_term = np.sum(a_k * np.cos(2 * np.pi * b_k * 0.5))
            
            # Calculate function value
            sum_i = 0.0
            for i in range(self.dimension):
                sum_k = 0.0
                for k in range(k_max + 1):
                    sum_k += a_k[k] * np.cos(2 * np.pi * b_k[k] * (z[i] + 0.5))
                sum_i += sum_k
            
            return sum_i - self.dimension * cos_term
        
        elif idx in [5, 7]:  # Functions 6, 8 (zero-indexed as 5, 7)
            # Griewank's Function
            sum_term = np.sum(z**2) / 4000.0
            
            prod_term = 1.0
            for i in range(self.dimension):
                prod_term *= np.cos(z[i] / np.sqrt(i + 1))
            
            return sum_term - prod_term + 1.0
        
        elif idx in [9]:  # Function 10 (zero-indexed as 9)
            # Sphere Function
            return np.sum(z**2)
        
        else:  # Functions 5, 9 (zero-indexed as 4, 8)
            # Ackley's Function
            sum_sq = np.sum(z**2) / self.dimension
            sum_cos = np.sum(np.cos(2 * np.pi * z)) / self.dimension
            
            term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq))
            term2 = -np.exp(sum_cos)
            
            return term1 + term2 + 20.0 + np.e
    
    def _normalize_function(self, idx: int, raw_value: float) -> float:
        """
        Normalize the raw function value to a similar height.
        
        Args:
            idx (int): Index of the basic function.
            raw_value (float): The raw function value.
            
        Returns:
            float: Normalized function value.
        """
        # Create fixed test point to estimate f_max
        test_point = np.ones(self.dimension) * 5.0
        test_point = test_point / self.lambda_values[idx]
        
        # Rotate test point if needed (not for F15 as all rotation matrices are identity)
        # test_point = np.dot(test_point, self.rotation_matrices[idx])
        
        # Evaluate function at test point to estimate f_max
        f_max = abs(self._evaluate_basic_function(idx, test_point))
        
        # Avoid division by zero
        if f_max < 1e-10:
            return self.C * raw_value
        else:
            return self.C * raw_value / f_max
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the F15 function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Calculate weights
        weights = self._calculate_weights(x)
        
        # Evaluate each component function and combine
        result = 0.0
        for i in range(self.n_func):
            # Transform input for current function
            z = (x - self.shift_vectors[i]) / self.lambda_values[i]
            
            # Note: For F15, all rotation matrices are identity, so no rotation is applied
            
            # Evaluate basic function
            raw_value = self._evaluate_basic_function(i, z)
            
            # Normalize and add to result with weight
            normalized_value = self._normalize_function(i, raw_value)
            result += weights[i] * (normalized_value + self.bias_values[i])
        
        return float(result) + self.bias
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the F15 function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        # Compute function values
        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            x = X[i]
            
            # Calculate weights
            weights = self._calculate_weights(x)
            
            # Evaluate each component function and combine
            value = 0.0
            for j in range(self.n_func):
                # Transform input for current function
                z = (x - self.shift_vectors[j]) / self.lambda_values[j]
                
                # Evaluate basic function
                raw_value = self._evaluate_basic_function(j, z)
                
                # Normalize and add to result with weight
                normalized_value = self._normalize_function(j, raw_value)
                value += weights[j] * (normalized_value + self.bias_values[j])
            
            result[i] = value
        
        return result + self.bias


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
        3: F03,
        4: F04,
        5: F05,
        6: F06,
        7: F07,
        8: F08,
        9: F09,
        10: F10,
        11: F11,
        12: F12,
        13: F13,
        14: F14,
        15: F15,
        # Functions F16-F25 will be implemented in the future
    }
    
    if function_number not in function_map:
        raise ValueError(f"Function F{function_number:02d} is not implemented yet.")
    
    # Create and return the requested function
    return function_map[function_number](dimension, bounds, data_dir) 