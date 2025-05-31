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
  - **decorators/**: Function transformations (shift, rotate, bias, etc.)
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

- Validate inputs using base class methods (`_validate_input`, `_validate_batch_input`).
  - All subclasses should use these methods for input validation in `evaluate` and `evaluate_batch`.
  - These methods only check shape and type (they do not ravel or mutate input).
- Use vectorized NumPy operations for performance
- Include references to academic sources
- Set appropriate default bounds for each function
- Support both single and batch evaluations
- Do not include transformations (bias, shift, rotation) in the base function implementation
- For modifications to the function's behavior, use the decorator pattern

### Decorators

- Implement decorators as subclasses of `InputTransformingFunction` or `OutputTransformingFunction` (both inherit from `ComposableFunction`).
- **InputTransformingFunction**: Use for decorators that transform the input before calling the base function (e.g., `Shifted`, `Rotated`, `Scaled`).
- **OutputTransformingFunction**: Use for decorators that transform the output of the base function (e.g., `Biased`, `Noise`, `MaxAbsolute`).
- Do **not** override `evaluate` or `evaluate_batch` in decorator subclasses; only implement `_apply` and `_apply_batch`.
- Maintain the dimension and bounds properties correctly (delegated to base function if present).
- Use composition pattern: wrap base function and delegate evaluation.
- Each decorator should handle a single transformation concern.

#### Example Decorator Usage

```python
# Input-transforming decorator
func = Shifted(base_function=SphereFunction(dimension=2), shift=np.ones(2))

# Output-transforming decorator
biased_func = Biased(base_function=func, bias=5.0)

# Chaining
composed = Biased(base_function=Shifted(base_function=SphereFunction(dimension=2), shift=np.ones(2)), bias=5.0)
```

#### Best Practices
- Choose the correct base class for your decorator's transformation type.
- Never override `evaluate` or `evaluate_batch` in subclasses of `InputTransformingFunction` or `OutputTransformingFunction`.
- Only implement `_apply` and `_apply_batch`.
- This pattern ensures DRY, KISS, and SOLID, and makes the codebase easy to extend and maintain.

### Function Transformations

- Always use decorators for transformations rather than modifying function implementations
- For bias, use the `BiasedFunction` decorator instead of adding bias in the function's `evaluate` method
- Decorator order matters and should be clearly documented when composing multiple transformations
- Testing should include verifying proper behavior with various combinations of decorators

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
   - Test decorator behavior independently and in combination

3. **Numeric Testing**:
   - Use appropriate floating-point comparison (e.g., `np.isclose`)
   - Include tests for edge cases (e.g., high dimensions)

4. **Test Expectations**:
   - Tests should avoid manual calculations in favor of pre-calculated constants
   - Test function values at known points against pre-verified values
   - When testing batch evaluation, derive expected values from individual evaluations
   - Evaluate properties rather than implementation details when possible
   - Test symmetry, boundary behavior, and extreme values
   - Test various dimensionalities where applicable
   - Verify decorator effects additively (e.g., biased function = original value + bias)
   - Use data-driven testing with test cases rather than repetitive assertion blocks

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
        
    Note:
        For transformations like shifting, rotation, or adding bias,
        use the decorator classes from the decorators module.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None):
        """Constructor with documentation for parameters."""
        # Set default bounds if none provided
        if bounds is None:
            bounds = np.array([[-10, 10]] * dimension)
        
        super().__init__(dimension, bounds)
        # Additional initialization
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the function at a point."""
        x = self._validate_input(x)
        # Implementation using vectorized operations
        return float(result)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorized batch evaluation."""
        X = self._validate_batch_input(X)
        # Optimized vectorized implementation
        return result
```

## Example Decorator Usage

```python
# Create a basic function
func = ExampleFunction(dimension=2)

# Add bias - increases function value without changing optimum location
biased_func = BiasedFunction(func, bias=5.0)

# Shift optimum - changes where the optimum is located
shift = np.array([1.0, 2.0])
shifted_func = ShiftedFunction(func, shift)

# Combine transformations
shifted_biased_func = BiasedFunction(ShiftedFunction(func, shift), bias=5.0)
```

## Import and Packaging Conventions

### 1. Use the **"src-layout"** for the package itself

```
yourproject/
├── src/
│   └── yourpkg/
│       ├── __init__.py
│       ├── core.py
│       └── data/
│           └── defaults.json
├── tests/
│   └── test_core.py
└── pyproject.toml
```

* `src/` is **not** on `sys.path` by default, so nothing inside it is import-able until the project is *installed* (`pip install -e .`). This prevents the "it works on my machine" problem where you accidentally import from the working directory instead of the installed wheel. ([Python Packaging][1])

### 2. Inside the package: **import by package name**, not by path

```python
# src/yourpkg/core.py
from yourpkg.utils import helper      # absolute import (preferred)
# or, within the same package,
from .utils import helper             # explicit relative is also fine
```

Avoid `sys.path` hacks or references such as `import src.yourpkg…`—they couple the code to the project layout.

### 3. Tests should import *the installed package*

```bash
# once per virtual-env
pip install -e .
```

```python
# tests/test_core.py
import yourpkg
from yourpkg.core import do_something
```

* **Do not** add `src/` to `PYTHONPATH` in CI; that hides packaging errors.
* If you use **pytest**, keep its default "importlib" mode (pytest ≥ 7) or set `--import-mode=importlib` so tests see the same import resolution an end-user will. ([docs.pytest.org][2])

### 4. Shipping constant files (JSON, CSV, etc.)

1. **Package them**:

   ```toml
   # pyproject.toml (PEP 621 + setuptools)
   [tool.setuptools.package-data]
   "yourpkg.data" = ["*.json"]
   ```

2. **Load them with the std-lib** `importlib.resources` API (3.9+):

   ```python
   from importlib import resources
   import json

   def load_defaults() -> dict:
       text = resources.files("yourpkg.data").joinpath("defaults.json").read_text()
       return json.loads(text)
   ```

   *Works whether the package is on disk, in a wheel, or in a zipapp.* ([Python documentation][3])

3. **Tests that need the same file** call the same helper:

   ```python
   def test_defaults_roundtrip():
       cfg = yourpkg.load_defaults()
       assert "threshold" in cfg
   ```

### 5. Test-only fixtures/data

Put them under `tests/resources/…` and access with `pathlib.Path(__file__).parent / "resources" / "sample.json"`. Those files are not part of the installable package, so they stay out of your wheel.

### 6. Quick checklist

| Scope                      | Where to import from                      | How to reach data                             |
| -------------------------- | ----------------------------------------- | --------------------------------------------- |
| **Library code** (`src/…`) | `import yourpkg…` or `from .sub import …` | `importlib.resources`                         |
| **Tests** (`tests/…`)      | *Same as users*: `import yourpkg…`        | Use library helper *or* `importlib.resources` |
| **Test-only data**         | n/a                                       | `Path(__file__).parent / "resources"`         |

Following this pattern keeps your runtime code, tests, and distributable artifacts aligned and eliminates path-related surprises.

[1]: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/?utm_source=chatgpt.com "src layout vs flat layout - Python Packaging User Guide"
[2]: https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html?utm_source=chatgpt.com "Good Integration Practices - pytest documentation"
[3]: https://docs.python.org/3/library/importlib.resources.html?utm_source=chatgpt.com "importlib.resources – Package resource reading, opening and ..." 