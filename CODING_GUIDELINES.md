# pyMOFL Coding Guidelines

This document outlines the coding standards and best practices for contributors to the pyMOFL project. Following these guidelines ensures code consistency, maintainability, and alignment with the project's architecture.

## Project Structure

- **src/pyMOFL/**: Main package code
  - **base.py**: Core abstractions and base classes
  - **functions/**: Optimization benchmark functions
    - **unimodal/**: Functions with a single optimum
    - **multimodal/**: Functions with multiple optima
    - **cec/**: Functions from CEC benchmarks
    - **hybrid/**: Hybrid functions
  - **decorators/**: Function transformations (shift, rotate, etc.)
  - **composites/**: Function composition mechanisms
  - **constraints/**: Constraint handlers
  - **utils/**: Utility functions and helpers
  - **constants/**: Constant values and configuration
- **tests/**: Test suite mirroring the package structure

## Code Style

1. **Python Version**: Code must be compatible with Python 3.8+
2. **Imports**:
   - Use absolute imports for external packages
   - Use relative imports for internal modules
   - Organize imports by standard library, third-party, and local modules

3. **Documentation**:
   - All modules must include a docstring explaining their purpose
   - All classes and functions must have docstrings following NumPy format
   - Include references to academic papers where applicable

4. **Type Annotations**:
   - Use type hints for function parameters and return values
   - Primarily use NumPy array types for vector/matrix parameters

## Implementation Guidelines

### Base Classes

- Extend `OptimizationFunction` for all optimization functions
- Implement abstract methods properly:
  - `evaluate(x)` for single-point evaluation
  - Consider overriding `evaluate_batch(X)` for vectorized operation

### Function Implementations

- Validate inputs using base class methods (`_validate_input`, `_validate_batch_input`)
- Use vectorized NumPy operations for performance
- Include references to academic sources
- Set appropriate default bounds for each function
- Support both single and batch evaluations

### Decorators

- Implement decorators as subclasses of `OptimizationFunction`
- Maintain the dimension and bounds properties correctly
- Use composition pattern: wrap base function and delegate evaluation

### Composites

- Clearly document the composition formula
- Validate component functions have matching dimensions
- Use vectorized operations when combining function values
- Handle component weights consistently

## Testing Guidelines

1. **Test Structure**:
   - Organize tests to mirror the package structure
   - Use pytest fixtures for common setup
   - Group tests in classes corresponding to the module being tested

2. **Test Coverage**:
   - Test initialization with default and custom parameters
   - Test function evaluation at known points (global minima)
   - Test bounds and dimension validation
   - Test batch evaluation methods
   - Test error cases and input validation

3. **Numeric Testing**:
   - Use appropriate floating-point comparison (e.g., `np.isclose`)
   - Include tests for edge cases (e.g., high dimensions)

## Development Workflow

1. Implement new functions/features in dedicated files/modules
2. Write corresponding tests
3. Ensure all tests pass and maintain 100% test coverage
4. Update documentation to reflect new functionality

## Performance Considerations

- Prefer vectorized NumPy operations over loops
- Optimize `evaluate_batch` for functions where vectorized computation is possible
- Consider caching expensive computations where appropriate

## Package Dependencies

- Core dependencies: `numpy`, `matplotlib`
- Keep external dependencies minimal
- Specify version constraints in pyproject.toml

## Example Implementation Pattern

```python
"""
Module docstring with clear description and references.
"""

import numpy as np
from ...base import OptimizationFunction

class ExampleFunction(OptimizationFunction):
    """
    Class docstring with mathematical formula and references.
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Function-specific bounds.
    """
    
    def __init__(self, dimension: int, bias: float = 0.0, bounds: np.ndarray = None):
        """Constructor with documentation for parameters."""
        super().__init__(dimension, bias, bounds or DEFAULT_BOUNDS)
        # Additional initialization
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the function at a point."""
        x = self._validate_input(x)
        # Implementation using vectorized operations
        return float(result + self.bias)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorized batch evaluation."""
        X = self._validate_batch_input(X)
        # Optimized vectorized implementation
        return result + self.bias
``` 