"""
Utility module for handling CEC benchmark function data.

This module provides functionality for loading, processing, and managing data files
required by CEC benchmark functions (shift vectors, rotation matrices, etc.).
"""
import os
import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class CECDataHandler:
    """
    Handler for CEC benchmark function data files.
    
    This class centralizes the loading and processing of data files used by CEC benchmark
    functions, such as shift vectors, rotation matrices, and function-specific matrices.
    
    Attributes:
        base_dir (str): Base directory for CEC data files.
        suite (str): CEC benchmark suite (e.g., "2005").
        manifest (dict): Loaded manifest data with file information.
    """
    
    def __init__(self, base_dir: str, suite: str):
        """
        Initialize the CEC data handler.
        
        Args:
            base_dir (str): Base directory for CEC data files.
            suite (str): CEC benchmark suite (e.g., "2005").
        """
        self.base_dir = base_dir
        self.suite = suite
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict:
        """
        Load the manifest file for the specified CEC suite.
        
        Returns:
            Dict: The loaded manifest data.
            
        Raises:
            FileNotFoundError: If the manifest file doesn't exist.
            RuntimeError: If there's an error loading the manifest file.
        """
        manifest_path = os.path.join(self.base_dir, f"meta_{self.suite}.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"CEC{self.suite} manifest file not found: {manifest_path}")
        
        try:
            with open(manifest_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load CEC{self.suite} manifest file: {e}")
    
    def get_function_info(self, func_number: int) -> Dict:
        """
        Get function information from the manifest.
        
        Args:
            func_number (int): Function number in the CEC suite.
            
        Returns:
            Dict: Function information from the manifest.
            
        Raises:
            ValueError: If the function is not found in the manifest.
        """
        func_key = f"f{func_number:02d}"
        
        if "functions" not in self.manifest or func_key not in self.manifest["functions"]:
            raise ValueError(f"Function {func_key} not found in manifest")
        
        return self.manifest["functions"][func_key]

    def _get_file_path(self, func_number: int, file_category: str, dimension: Optional[int] = None) -> str:
        """
        Get the file path for a specific function data file from the manifest.
        
        Args:
            func_number (int): Function number in the CEC suite.
            file_category (str): Category/type of file (e.g., "shift").
            dimension (int, optional): Specific dimension version to look for.
            
        Returns:
            str: The full path to the data file.
            
        Raises:
            FileNotFoundError: If the file is not found in the manifest.
        """
        func_info = self.get_function_info(func_number)
        func_key = f"f{func_number:02d}"
        
        # If dimension is specified, try dimension-specific key first
        if dimension is not None:
            dim_specific_key = f"{file_category}{dimension}"
            if "files" in func_info and dim_specific_key in func_info["files"]:
                file_path = os.path.join(self.base_dir, func_info["files"][dim_specific_key])
                if os.path.exists(file_path):
                    return file_path
        
        # Try generic key
        if "files" in func_info and file_category in func_info["files"]:
            file_path = os.path.join(self.base_dir, func_info["files"][file_category])
            if os.path.exists(file_path):
                return file_path
        
        # Provide a clear error message
        if dimension is not None:
            raise FileNotFoundError(
                f"{file_category.capitalize()} file not found for function {func_key} with dimension {dimension}"
            )
        else:
            raise FileNotFoundError(
                f"{file_category.capitalize()} file not found for function {func_key}"
            )
    
    def load_vector(self, func_number: int, dimension: int, file_category: str) -> np.ndarray:
        """
        Load a vector (e.g., shift vector) for a specific function and dimension.
        
        Args:
            func_number (int): Function number in the CEC suite.
            dimension (int): Required dimension for the vector.
            file_category (str): Category/type of file (e.g., "shift").
        
        Returns:
            np.ndarray: Loaded vector sliced to the required dimension.
            
        Raises:
            FileNotFoundError: If the vector file is not found.
            ValueError: If the loaded vector is smaller than the required dimension.
        """
        file_path = self._get_file_path(func_number, file_category)
        
        try:
            # Load the full vector
            full_vector = np.loadtxt(file_path)
            
            # Validate and slice if needed
            if len(full_vector) < dimension:
                raise ValueError(
                    f"{file_category.capitalize()} vector file contains {len(full_vector)} values, "
                    f"but {dimension} are needed"
                )
            
            # Return sliced vector
            return full_vector[:dimension]
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            raise RuntimeError(f"Failed to load {file_category} vector: {e}")
    
    def load_matrix(self, func_number: int, dimension: int, file_category: str, 
                   transpose: bool = False) -> np.ndarray:
        """
        Load a matrix (e.g., rotation matrix) for a specific function and dimension.
        
        Args:
            func_number (int): Function number in the CEC suite.
            dimension (int): Required dimension for the matrix.
            file_category (str): Category/type of file (e.g., "rotation").
            transpose (bool, optional): Whether to transpose the matrix after loading.
                                       Default is False.
        
        Returns:
            np.ndarray: Loaded matrix sliced to the required dimension.
            
        Raises:
            FileNotFoundError: If the matrix file is not found.
            ValueError: If the loaded matrix is smaller than the required dimension.
        """
        # Try dimension-specific file first
        try:
            file_path = self._get_file_path(func_number, file_category, dimension)
        except FileNotFoundError:
            # If dimension-specific file not found, try generic file
            file_path = self._get_file_path(func_number, file_category)
        
        try:
            # Load the full matrix
            full_matrix = np.loadtxt(file_path)
            
            # Validate matrix shape
            if full_matrix.ndim != 2:
                raise ValueError(
                    f"{file_category.capitalize()} matrix file contains data with unexpected dimensions: {full_matrix.ndim}"
                )
            
            # Validate matrix size
            if (full_matrix.shape[0] < dimension or full_matrix.shape[1] < dimension):
                raise ValueError(
                    f"{file_category.capitalize()} matrix file contains {full_matrix.shape} matrix, "
                    f"but ({dimension}, {dimension}) is needed"
                )
            
            # Slice to required dimension and transpose if needed
            result = full_matrix[:dimension, :dimension]
            if transpose:
                result = result.T
            
            return result
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            raise RuntimeError(f"Failed to load {file_category} matrix: {e}")
    
    def load_schwefel_213_data(self, dimension: int, func_number: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the specific data files for Schwefel's Problem 2.13.
        
        This function loads the A and B matrices and alpha vector required for 
        Schwefel's Problem 2.13, which appears as different function numbers in 
        different CEC benchmark suites.
        
        Args:
            dimension (int): Required dimension for the vectors and matrices.
            func_number (int, optional): Function number in the CEC suite. 
                                        Default is 12 (F12 in CEC2005).
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of (A, B, alpha) matrices/vectors.
            
        Raises:
            FileNotFoundError: If any required file is not found.
            ValueError: If any loaded data is smaller than the required dimension.
        """
        try:
            # Get file paths from manifest
            alpha_path = self._get_file_path(func_number, "alpha")
            a_path = self._get_file_path(func_number, "A")
            b_path = self._get_file_path(func_number, "B")
            
            # Load data files
            alpha_full = np.loadtxt(alpha_path)
            a_full = np.loadtxt(a_path)
            b_full = np.loadtxt(b_path)
            
            # Validate shapes
            if alpha_full.ndim != 1 or alpha_full.shape[0] < dimension:
                raise ValueError(
                    f"Alpha vector has unexpected shape {alpha_full.shape} for dimension {dimension}"
                )
            if a_full.ndim != 2 or a_full.shape[0] < dimension or a_full.shape[1] < dimension:
                raise ValueError(
                    f"A matrix has unexpected shape {a_full.shape} for dimension {dimension}"
                )
            if b_full.ndim != 2 or b_full.shape[0] < dimension or b_full.shape[1] < dimension:
                raise ValueError(
                    f"B matrix has unexpected shape {b_full.shape} for dimension {dimension}"
                )
            
            # Slice to required dimension
            return (
                a_full[:dimension, :dimension],
                b_full[:dimension, :dimension],
                alpha_full[:dimension]
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Missing data files for Schwefel's Problem 2.13 (F{func_number}): {e}"
            )
        except ValueError as e:
            raise
        except Exception as e:
            raise RuntimeError(f"Error processing Schwefel 2.13 data files: {e}")
    
    def load_schwefel_26_data(self, dimension: int, func_number: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the specific data files for Schwefel's Problem 2.6.
        
        Args:
            dimension (int): Required dimension for the vectors and matrices.
            func_number (int, optional): Function number in the CEC suite.
                                        Default is 5 (F05 in CEC2005).
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (A_matrix, shift_vector).
            
        Raises:
            FileNotFoundError: If any required file is not found.
            ValueError: If any loaded data is smaller than the required dimension.
        """
        try:
            # Get A matrix and shift vector using the common loading methods
            a_matrix = self.load_matrix(func_number, dimension, "A")
            shift_vector = self.load_vector(func_number, dimension, "shift")
            
            # Adjust shift vector to bounds as required for Schwefel's Problem 2.6
            # First quarter set to lower bound (-100)
            index = dimension // 4
            if index < 1:  # Handle edge case for very small dimensions
                index = 1
            for i in range(index):
                shift_vector[i] = -100.0
            
            # Last quarter set to upper bound (100)
            index = (3 * dimension) // 4 - 1
            if index < 0:  # Handle edge case for very small dimensions
                index = 0
            for i in range(index, dimension):
                shift_vector[i] = 100.0
                
            return a_matrix, shift_vector
            
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Missing data files for Schwefel's Problem 2.6 (F{func_number}): {e}"
            )
        except ValueError as e:
            raise
        except Exception as e:
            raise RuntimeError(f"Error processing Schwefel 2.6 data files: {e}")
    
    # For backward compatibility
    def load_f05_data(self, dimension: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the specific data files for F05 (Schwefel's Problem 2.6).
        
        This method is maintained for backward compatibility.
        Consider using load_schwefel_26_data() instead.
        
        Args:
            dimension (int): Required dimension for the vectors and matrices.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (A_matrix, shift_vector).
        """
        return self.load_schwefel_26_data(dimension, func_number=5)
    
    def get_bias(self, func_number: int, bias_map: Dict[int, float]) -> float:
        """
        Get the bias value for a function.
        
        Args:
            func_number (int): Function number in the CEC suite.
            bias_map (Dict[int, float]): Map of function numbers to bias values.
        
        Returns:
            float: The bias value for the function.
        """
        func_info = self.get_function_info(func_number)
        
        # First try to get bias from manifest
        if "bias" in func_info:
            return func_info["bias"]
        
        # If not in manifest, use the provided bias map
        return bias_map.get(func_number, 0.0) 