# pyMOFL: Python Modular Optimization Function Library

A composable optimization function library for benchmarking optimization algorithms.

## Overview

pyMOFL is a Python library that provides a collection of benchmark functions commonly used in optimization research, along with tools for transforming and composing these functions to create complex benchmarks. The library is designed to be modular, extensible, and high-performance.

## Features

- **Modular Design**: Separate core mathematical functions from transformation and composition logic.
- **Composability**: Build complex benchmark functions (e.g., CEC composite functions) by assembling basic functions with transformation decorators.
- **Performance**: Utilize vectorized NumPy operations for high-speed execution.
- **Extensibility**: Provide a clear, consistent interface that users can extend to define new functions, hybrids, or composites.

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

### Transformations

```python
import numpy as np
from pyMOFL.functions.unimodal import SphereFunction
from pyMOFL.decorators import ShiftedFunction, RotatedFunction

# Create a shifted Sphere function
sphere = SphereFunction(dimension=2)
shift = np.array([1.0, 2.0])
shifted_sphere = ShiftedFunction(sphere, shift)
print(f"Shifted Sphere function at [1, 2]: {shifted_sphere.evaluate(np.array([1, 2]))}")

# Create a rotated Rastrigin function
from pyMOFL.functions.multimodal import RastriginFunction
rotation_matrix = np.array([[0.866, -0.5], [0.5, 0.866]])  # 30-degree rotation
rotated_rastrigin = RotatedFunction(rastrigin, rotation_matrix)
print(f"Rotated Rastrigin function at [0, 0]: {rotated_rastrigin.evaluate(np.array([0, 0]))}")
```

### Composite Functions

```python
import numpy as np
from pyMOFL.functions.unimodal import SphereFunction, RosenbrockFunction
from pyMOFL.functions.multimodal import RastriginFunction
from pyMOFL.composites import CompositeFunction

# Create a composite function
components = [SphereFunction(dimension=2), RastriginFunction(dimension=2), RosenbrockFunction(dimension=2)]
sigmas = [1.0, 2.0, 3.0]
lambdas = [1.0, 1.0, 1.0]
biases = [0.0, 100.0, 200.0]
composite = CompositeFunction(components, sigmas, lambdas, biases)
print(f"Composite function at [0, 0]: {composite.evaluate(np.array([0, 0]))}")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
