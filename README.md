# pyMOFL: Python Modular Optimization Function Library

A composable optimization function library for benchmarking optimization algorithms.

## Overview

pyMOFL is a Python library that provides a collection of benchmark functions commonly used in optimization research, along with tools for transforming and composing these functions to create complex benchmarks. The library is designed to be modular, extensible, and high-performance.

## Features

- **Modular Design**: Separate core mathematical functions from transformation and composition logic.
- **Composability**: Build complex benchmark functions (e.g., CEC composite functions) by assembling basic functions with transformation decorators.
- **Performance**: Utilize vectorized NumPy operations for high-speed execution.
- **Extensibility**: Provide a clear, consistent interface that users can extend to define new functions, hybrids, or composites.
- **Decorator Pattern**: Apply transformations like shifting, rotation, and bias using a consistent decorator approach.

## Installation

```bash
# Install from source
git clone https://github.com/yourusername/pyMOFL.git
cd pyMOFL
pip install -e .
```

## Usage

### Basic Functions

```python
import numpy as np
from pyMOFL.functions.unimodal import SphereFunction
from pyMOFL.functions.multimodal import RastriginFunction

# Create a 2D Sphere function
sphere = SphereFunction(dimension=2)
print(f"Sphere function at [0, 0]: {sphere.evaluate(np.array([0, 0]))}")

# Create a 2D Rastrigin function
rastrigin = RastriginFunction(dimension=2)
print(f"Rastrigin function at [0, 0]: {rastrigin.evaluate(np.array([0, 0]))}")
```

### Function Transformations with Decorators

pyMOFL uses a decorator pattern for applying transformations to optimization functions. This approach provides a consistent and composable way to transform functions.

```python
import numpy as np
from pyMOFL.functions.unimodal import SphereFunction
from pyMOFL.decorators import ShiftedFunction, RotatedFunction, BiasedFunction

# Create a basic function
sphere = SphereFunction(dimension=2)

# Add bias to shift the function value (not the position)
biased_sphere = BiasedFunction(sphere, bias=10.0)
print(f"Biased Sphere at [0, 0]: {biased_sphere.evaluate(np.array([0, 0]))}")  # 10.0

# Shift the function's position (moves the optimum)
shift = np.array([1.0, 2.0])
shifted_sphere = ShiftedFunction(sphere, shift)
print(f"Shifted Sphere at [1, 2]: {shifted_sphere.evaluate(np.array([1, 2]))}")  # 0.0

# Rotate the function
rotation_matrix = np.array([[0.866, -0.5], [0.5, 0.866]])  # 30-degree rotation
rotated_sphere = RotatedFunction(sphere, rotation_matrix)

# Combine multiple transformations (order matters!)
# First shift, then bias
shifted_then_biased = BiasedFunction(ShiftedFunction(sphere, shift), bias=5.0)
print(f"Shifted then biased at [1, 2]: {shifted_then_biased.evaluate(np.array([1, 2]))}")  # 5.0

# First bias, then shift (equivalent result for sphere function)
biased_then_shifted = ShiftedFunction(BiasedFunction(sphere, bias=5.0), shift)
print(f"Biased then shifted at [1, 2]: {biased_then_shifted.evaluate(np.array([1, 2]))}")  # 5.0
```

### Multiple Transformations Example

Here's an example of combining multiple transformations to create a more complex function:

```python
import numpy as np
from pyMOFL.functions.multimodal import RastriginFunction
from pyMOFL.decorators import ShiftedFunction, RotatedFunction, BiasedFunction

# Create a basic Rastrigin function
rastrigin = RastriginFunction(dimension=2)

# Create a rotation matrix (45-degree rotation)
theta = np.pi/4
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# Apply multiple transformations:
# 1. Shift the function's optimum position
# 2. Rotate the function landscape
# 3. Add a bias to the function value
shift = np.array([2.0, 3.0])
bias = 100.0

# Chain decorators (order matters!)
transformed_function = BiasedFunction(
    RotatedFunction(
        ShiftedFunction(rastrigin, shift),
        rotation_matrix
    ),
    bias
)

# Evaluate the transformed function
original_optimum = np.array([0, 0])  # Original optimum for Rastrigin
shifted_optimum = np.array([2, 3])   # New optimum after shifting

print(f"Original at [0, 0]: {rastrigin.evaluate(original_optimum)}")                  # 0.0
print(f"Transformed at original: {transformed_function.evaluate(original_optimum)}")  # Not minimum
print(f"Transformed at shifted: {transformed_function.evaluate(shifted_optimum)}")    # 100.0 (bias)
```

### Composite Functions

```python
import numpy as np
from pyMOFL.functions.unimodal import SphereFunction, RosenbrockFunction
from pyMOFL.functions.multimodal import RastriginFunction
from pyMOFL.composites import CompositeFunction
from pyMOFL.decorators import BiasedFunction

# Create component functions
sphere = SphereFunction(dimension=2)
rastrigin = RastriginFunction(dimension=2)
rosenbrock = RosenbrockFunction(dimension=2)

# Apply bias to component functions using decorators
biased_rastrigin = BiasedFunction(rastrigin, bias=100.0)
biased_rosenbrock = BiasedFunction(rosenbrock, bias=200.0)

# Create a composite function
components = [sphere, biased_rastrigin, biased_rosenbrock]
sigmas = [1.0, 2.0, 3.0]
lambdas = [1.0, 1.0, 1.0]
composite = CompositeFunction(components, sigmas, lambdas)

print(f"Composite function at [0, 0]: {composite.evaluate(np.array([0, 0]))}")
```

## Decorator Pattern for Function Transformations

pyMOFL uses a decorator pattern for transforming functions, which offers several advantages:

1. **Consistency**: All transformations follow the same pattern
2. **Composability**: Easily combine multiple transformations
3. **Separation of Concerns**: Core function behavior is separate from transformations
4. **Extensibility**: Add new transformations without modifying existing functions

Available decorators:
- `ShiftedFunction`: Shifts the function's optimum position
- `RotatedFunction`: Rotates the function's landscape 
- `BiasedFunction`: Adds a constant value to the function output

The order of decorators matters and produces different results depending on the function:
- `BiasedFunction(ShiftedFunction(f))`: Shifts the function first, then adds bias
- `ShiftedFunction(BiasedFunction(f))`: Adds bias first, then shifts the function

Recommended ordering for most scenarios:
1. Apply `ShiftedFunction` first to move the optimum
2. Apply `RotatedFunction` next to rotate the landscape
3. Apply `BiasedFunction` last to shift the function value

## License

This project is licensed under the MIT License - see the LICENSE file for details.
