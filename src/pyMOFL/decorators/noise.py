"""
Noise function decorator implementation.

This module provides a decorator that adds noise to a base optimization function's evaluation.
"""

import numpy as np
from ..base import OptimizationFunction


class NoiseDecorator(OptimizationFunction):
    """
    A decorator that adds noise to a base optimization function's evaluation.
    
    The noise is added to the function value, typically as a multiplicative factor.
    
    Attributes:
        base (OptimizationFunction): The base optimization function to which noise is added.
        noise_type (str): Type of noise to apply ('gaussian', 'uniform').
        noise_level (float): Magnitude of the noise effect (default: 0.1).
        dimension (int): The dimensionality of the function (inherited from the base function).
    """
    
    def __init__(self, base_func: OptimizationFunction, noise_type: str = 'gaussian', 
                 noise_level: float = 0.1, noise_seed: int = None):
        """
        Initialize the noise function decorator.
        
        Args:
            base_func (OptimizationFunction): The base optimization function to which noise is added.
            noise_type (str): Type of noise to apply ('gaussian', 'uniform'). Default is 'gaussian'.
            noise_level (float): Magnitude of the noise effect. Default is 0.1.
            noise_seed (int, optional): Random seed for reproducible noise. If None, random noise is generated.
        """
        self.base = base_func
        self.dimension = base_func.dimension
        self.noise_type = noise_type.lower()
        self.noise_level = noise_level
        
        # Set random seed if provided
        if noise_seed is not None:
            np.random.seed(noise_seed)
        
        # Validate noise type
        if self.noise_type not in ['gaussian', 'uniform']:
            raise ValueError(f"Unsupported noise type: {noise_type}. Use 'gaussian' or 'uniform'.")
        
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
    
    def _generate_noise(self, is_batch: bool = False, size: int = 1) -> float:
        """
        Generate noise value based on the specified type and level.
        
        Args:
            is_batch (bool): Whether to generate noise for a batch evaluation.
            size (int): The size of the batch, if is_batch is True.
            
        Returns:
            float or np.ndarray: The generated noise value(s).
        """
        if self.noise_type == 'gaussian':
            if is_batch:
                return 1.0 + self.noise_level * np.abs(np.random.normal(size=size))
            else:
                return 1.0 + self.noise_level * np.abs(np.random.normal())
        elif self.noise_type == 'uniform':
            if is_batch:
                return 1.0 + self.noise_level * np.random.uniform(size=size)
            else:
                return 1.0 + self.noise_level * np.random.uniform()
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the noisy function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x with added noise.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Evaluate the base function 
        base_value = self.base.evaluate(x)
        
        # Apply noise
        noise = self._generate_noise()
        
        # Return base value with applied noise
        return float(base_value * noise)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the noisy function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point with added noise.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        # Evaluate the base function
        base_values = self.base.evaluate_batch(X)
        
        # Apply noise for each point
        noise = self._generate_noise(is_batch=True, size=X.shape[0])
        
        # Return base values with applied noise
        return base_values * noise 