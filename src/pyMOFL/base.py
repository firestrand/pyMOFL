import numpy as np
from abc import ABC, abstractmethod

class OptimizationFunction(ABC):
    """
    Abstract base class for optimization benchmark functions.
    
    Each optimization function should:
    - Provide an `evaluate(x)` method that returns the function's value at a given point.
    - Expose a `bounds` property representing the lower and upper limits for each dimension.
    - Optionally, implement or override `evaluate_batch(X)` for vectorized evaluations.
    
    For function transformations like shifting, rotation, or adding bias, use the decorator 
    classes provided in the `decorators` module rather than modifying the base functions.
    
    Attributes:
        dimension (int): The dimensionality of the input space.
        _bounds (np.ndarray): A 2D array of shape (dimension, 2) representing [lower, upper] bounds.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None):
        """
        Initialize the optimization function with the given dimension and bounds.
        
        Args:
            dimension (int): The number of dimensions for the input.
            bounds (np.ndarray, optional): A 2D numpy array of shape (dimension, 2) for the bounds.
                                           Defaults to [-100, 100] for each dimension if not provided.
        
        Note:
            To add a bias to the function, use the BiasedFunction decorator from the decorators module.
        """
        self.dimension = dimension
        if bounds is None:
            # Set default bounds to [-100, 100] for each dimension
            self._bounds = np.array([[-100, 100]] * dimension)
        else:
            self._bounds = np.array(bounds)
            if self._bounds.shape != (dimension, 2):
                raise ValueError("Bounds must be of shape (dimension, 2)")
    
    @property
    def bounds(self) -> np.ndarray:
        """
        Get the search space bounds for the function.
        
        Returns:
            np.ndarray: A 2D array of shape (dimension, 2) with lower and upper bounds.
        """
        return self._bounds
    
    def _validate_input(self, x: np.ndarray) -> np.ndarray:
        """
        Validate and preprocess the input for evaluation.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            np.ndarray: The validated and preprocessed input.
            
        Raises:
            ValueError: If the input dimension does not match the function's dimension.
        """
        # Ensure x is a numpy array
        x = np.array(x)
        
        # Check if the input has the correct dimension
        if x.size != self.dimension:
            raise ValueError(f"Expected input dimension {self.dimension}, got {x.size}")
        
        # Apply ravel to ensure consistent shape
        return x.ravel()
    
    def _validate_batch_input(self, X: np.ndarray) -> np.ndarray:
        """
        Validate and preprocess the batch input for evaluation.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The validated and preprocessed batch input.
            
        Raises:
            ValueError: If the input dimension does not match the function's dimension.
        """
        # Ensure X is a numpy array
        X = np.asarray(X)
        
        # Check if the input has the correct shape
        if X.shape[1] != self.dimension:
            raise ValueError(f"Expected input dimension {self.dimension}, got {X.shape[1]}")
        
        return X
    
    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the function at a single point.
        
        Args:
            x (np.ndarray): A 1D numpy array of length `dimension` representing a point in the search space.
        
        Returns:
            float: The function value at point `x`.
        """
        pass

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the function on a batch of points.
        
        Args:
            X (np.ndarray): A 2D numpy array of shape (num_points, dimension) representing multiple points.
        
        Returns:
            np.ndarray: A 1D array containing the function values for each point in `X`.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        # This default implementation uses a list comprehension.
        # Derived classes can override this method for better performance if needed.
        return np.array([self.evaluate(x) for x in X])
