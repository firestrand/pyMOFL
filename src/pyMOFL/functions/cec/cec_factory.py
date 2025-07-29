"""
Factory for creating CEC benchmark functions from JSON configuration.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import get
from pyMOFL.decorators import DECORATOR_REGISTRY


class CECFactory:
    """
    Factory class for creating CEC benchmark functions from JSON configuration.
    
    Parameters
    ----------
    base_path : str or Path
        Base path where the constant data files are located.
    """
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
    
    def _get_default_bounds(self, dimension: int) -> Bounds:
        """Get default bounds for decorators without base functions."""
        return Bounds(
            low=np.full(dimension, -100),
            high=np.full(dimension, 100),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
    
    def create_function(self, config: Dict[str, Any], dimension: int) -> OptimizationFunction:
        """
        Create a function from configuration.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Function configuration from the JSON file.
        dimension : int
            Dimension of the function to create.
            
        Returns
        -------
        OptimizationFunction
            The constructed function.
        """
        # Extract bounds from search space
        bounds_config = config.get("search_space", {}).get("default_bounds", {})
        min_bound = bounds_config.get("min", -100)
        max_bound = bounds_config.get("max", 100)
        
        # Create bounds objects
        init_bounds = Bounds(
            low=np.full(dimension, min_bound),
            high=np.full(dimension, max_bound),
            mode=BoundModeEnum.INITIALIZATION,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        
        oper_bounds = Bounds(
            low=np.full(dimension, min_bound),
            high=np.full(dimension, max_bound),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        
        # Build the function from the nested structure
        function = self._build_function(config["function"], dimension, init_bounds, oper_bounds)
        
        return function
    
    def _build_function(self, 
                       func_config: Dict[str, Any], 
                       dimension: int,
                       init_bounds: Bounds,
                       oper_bounds: Bounds) -> OptimizationFunction:
        """
        Recursively build a function from nested configuration.
        
        Parameters
        ----------
        func_config : Dict[str, Any]
            Function configuration node.
        dimension : int
            Dimension of the function.
        init_bounds : Bounds
            Initialization bounds.
        oper_bounds : Bounds
            Operational bounds.
            
        Returns
        -------
        OptimizationFunction
            The constructed function or decorator.
        """
        func_type = func_config["type"]
        parameters = func_config.get("parameters", {})
        
        # Map lowercase names to decorator registry names
        decorator_mapping = {
            "shift": "Shifted",
            "rotate": "Rotated", 
            "bias": "Biased",
            "scale": "Scale",
            "weight": "Weight",
            "normalize": "Normalize"
        }
        
        # Check if this is a decorator
        decorator_name = decorator_mapping.get(func_type, func_type)
        if decorator_name in DECORATOR_REGISTRY:
            # Check if there's an inner function
            if "function" in func_config:
                # Build the inner function first
                inner_function = self._build_function(
                    func_config["function"], dimension, init_bounds, oper_bounds
                )
                # Apply the decorator
                return self._apply_decorator(decorator_name, inner_function, parameters, dimension)
            else:
                # Some decorators can work without a base function
                # Create the decorator without a base function
                return self._apply_decorator(decorator_name, None, parameters, dimension)
        
        # Otherwise, it's a base function
        # But check if it has a nested "function" property which means decorators are inside
        if "function" in func_config:
            # This means we have decorators nested inside the base function
            # Build the inner decorators first
            inner_decorators = self._build_function(
                func_config["function"], dimension, init_bounds, oper_bounds
            )
            # Create the base function
            base_func = self._create_base_function(func_type, dimension, init_bounds, oper_bounds, parameters)
            # Apply the inner decorators to the base function
            # This requires special handling - we need to "inject" the base function
            return self._apply_inner_decorators(base_func, inner_decorators)
        else:
            # Pure base function with no decorators
            return self._create_base_function(func_type, dimension, init_bounds, oper_bounds, parameters)
    
    def _create_base_function(self, 
                            func_type: str, 
                            dimension: int,
                            init_bounds: Bounds,
                            oper_bounds: Bounds,
                            parameters: Dict[str, Any]) -> OptimizationFunction:
        """
        Create a base optimization function.
        
        Parameters
        ----------
        func_type : str
            Type of the function.
        dimension : int
            Dimension of the function.
        init_bounds : Bounds
            Initialization bounds.
        oper_bounds : Bounds
            Operational bounds.
        parameters : Dict[str, Any]
            Function parameters.
            
        Returns
        -------
        OptimizationFunction
            The created function.
        """
        # Map function names from JSON to registry names
        function_mapping = {
            "sphere": "Sphere",
            "schwefel_1_2": "Schwefel12",
            "rosenbrock": "Rosenbrock",
            "rastrigin": "Rastrigin",
            "griewank": "Griewank",
            "ackley": "Ackley",
            "schwefel_2_13": "Schwefel213",
            "griewank_of_rosenbrock": "GriewankOfRosenbrock",
            "schaffer_f6_expanded": "SchafferF6Expanded"
        }
        
        # Get the function class from registry
        registry_name = function_mapping.get(func_type, func_type)
        try:
            func_class = get(registry_name)
        except KeyError:
            raise ValueError(f"Unknown function type: {func_type} (tried registry name: {registry_name})")
        
        # Create the function with bounds
        return func_class(
            dimension=dimension,
            initialization_bounds=init_bounds,
            operational_bounds=oper_bounds,
            **parameters
        )
    
    def _apply_decorator(self, 
                        decorator_type: str,
                        base_function: OptimizationFunction,
                        parameters: Dict[str, Any],
                        dimension: int) -> OptimizationFunction:
        """
        Apply a decorator to a base function.
        
        Parameters
        ----------
        decorator_type : str
            Type of decorator to apply.
        base_function : OptimizationFunction
            The function to decorate.
        parameters : Dict[str, Any]
            Decorator parameters.
        dimension : int
            Dimension of the function.
            
        Returns
        -------
        OptimizationFunction
            The decorated function.
        """
        decorator_class = DECORATOR_REGISTRY[decorator_type]
        
        # Handle special parameters that need file loading or renaming
        processed_params = {}
        
        # Parameter name mappings from JSON to decorator constructors
        param_mapping = {
            "value": "bias",  # Biased decorator uses 'bias' not 'value'
            "vector": "shift"  # Shifted decorator uses 'shift' not 'vector'
        }
        
        for key, value in parameters.items():
            # Map parameter names
            param_name = param_mapping.get(key, key)
            
            if key == "vector" and isinstance(value, str):
                # Load shift vector from file
                processed_params[param_name] = self._load_vector(value, dimension)
            elif key == "matrix" and isinstance(value, str):
                # Load rotation matrix from file
                processed_params[param_name] = self._load_matrix(value, dimension)
            else:
                processed_params[param_name] = value
        
        # Apply the decorator
        if base_function is None:
            # Some decorators can work without a base function
            # Pass dimension and bounds explicitly
            return decorator_class(
                dimension=dimension,
                initialization_bounds=self._get_default_bounds(dimension),
                operational_bounds=self._get_default_bounds(dimension),
                **processed_params
            )
        else:
            return decorator_class(base_function, **processed_params)
    
    def _load_vector(self, filename: str, dimension: int) -> np.ndarray:
        """
        Load a vector from file.
        
        Parameters
        ----------
        filename : str
            Filename relative to base_path.
        dimension : int
            Expected dimension of the vector.
            
        Returns
        -------
        np.ndarray
            The loaded vector.
        """
        filepath = self.base_path / filename
        data = np.loadtxt(filepath)
        
        # Handle dimension mismatch
        if len(data) > dimension:
            # Truncate to match dimension
            return data[:dimension]
        elif len(data) < dimension:
            raise ValueError(f"Vector in {filename} has {len(data)} elements, "
                           f"but dimension {dimension} was requested")
        
        return data
    
    def _load_matrix(self, filename: str, dimension: int) -> np.ndarray:
        """
        Load a matrix from file.
        
        Parameters
        ----------
        filename : str
            Filename relative to base_path.
        dimension : int
            Expected dimension of the matrix.
            
        Returns
        -------
        np.ndarray
            The loaded matrix.
        """
        filepath = self.base_path / filename
        data = np.loadtxt(filepath)
        
        # Reshape if needed
        if data.ndim == 1:
            # Assume it's a flattened square matrix
            size = int(np.sqrt(len(data)))
            if size * size != len(data):
                raise ValueError(f"Cannot reshape {len(data)} elements into square matrix")
            data = data.reshape((size, size))
        
        # Handle dimension mismatch
        if data.shape[0] > dimension:
            # Truncate to match dimension
            return data[:dimension, :dimension]
        elif data.shape[0] < dimension:
            raise ValueError(f"Matrix in {filename} has shape {data.shape}, "
                           f"but dimension {dimension} was requested")
        
        return data
    
    def _apply_inner_decorators(self, base_function: OptimizationFunction, 
                               decorator_chain: OptimizationFunction) -> OptimizationFunction:
        """
        Apply decorators that were nested inside a base function definition.
        
        This handles the case where the JSON has structure like:
        {
            "type": "sphere",
            "function": {
                "type": "shift", 
                "parameters": {...}
            }
        }
        
        We need to "inject" the base function (sphere) into the decorator chain.
        
        Parameters
        ----------
        base_function : OptimizationFunction
            The base function (e.g., sphere)
        decorator_chain : OptimizationFunction
            The decorator chain built from the nested structure
            
        Returns
        -------
        OptimizationFunction
            The properly composed function
        """
        # Find the innermost decorator in the chain
        current = decorator_chain
        while hasattr(current, 'base_function') and current.base_function is not None:
            current = current.base_function
        
        # Replace the innermost decorator's base_function with our actual base function
        if hasattr(current, 'base_function'):
            current.base_function = base_function
            # Update dimension to match
            current.dimension = base_function.dimension
            return decorator_chain
        else:
            # The decorator chain ends with a leaf node, which shouldn't happen
            # in properly structured JSON
            raise ValueError("Invalid decorator chain structure - no base function slot found")