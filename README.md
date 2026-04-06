# pyMOFL: Python Modular Optimization Function Library

[![PyPI version](https://img.shields.io/pypi/v/pyMOFL.svg)](https://pypi.org/project/pyMOFL/)
[![Python](https://img.shields.io/pypi/pyversions/pyMOFL.svg)](https://pypi.org/project/pyMOFL/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`pyMOFL` is a modular benchmark-function library for optimization research.
It uses a **function-composition architecture** driven by declarative configs.

**Key features:**

- 260+ registered benchmark functions (unimodal, multimodal, classical, Mishra family, and more)
- 10 bundled benchmark suites: CEC 2005/2013/2014/2015/2017/2020/2021/2022, BBOB, and GNBG
- Extended BBOB variants: noisy, mixed-integer, large-scale, and constrained
- Functional transformation pipeline (shift, rotate, scale, bias, noise, oscillation, asymmetric, penalty, and more)
- Composition and hybrid function creation (weighted, min, hybrid)
- Config-driven construction via `FunctionFactory` for reproducible benchmarks
- High-performance vectorized NumPy operations

## Installation

From PyPI:

```bash
pip install pyMOFL
```

With CLI support:

```bash
pip install "pyMOFL[cli]"
```

From source (for development):

```bash
git clone https://github.com/firestrand/pyMOFL.git
cd pyMOFL
uv sync --extra dev --extra cli
```

## Quick Start

### Direct functions

```python
import numpy as np
from pyMOFL.functions.benchmark import SphereFunction, RastriginFunction

sphere = SphereFunction(dimension=2)
print(sphere.evaluate(np.array([0.0, 0.0])))

rastrigin = RastriginFunction(dimension=2)
print(rastrigin.evaluate(np.array([1.0, 2.0])))
```

### Compositions (shift, rotate, bias)

Transformations are composed explicitly via `ComposedFunction`:

```python
import numpy as np
from pyMOFL.functions.benchmark import SphereFunction
from pyMOFL.functions.transformations import (
    BiasTransform,
    ComposedFunction,
    RotateTransform,
    ShiftTransform,
)

shift = np.array([1.0, 2.0])
rotation = np.array([[0.8660254, -0.5], [0.5, 0.8660254]])

base = SphereFunction(dimension=2)
composed = ComposedFunction(
    base_function=base,
    input_transforms=[
        ShiftTransform(shift),
        RotateTransform(rotation),
    ],
    output_transforms=[BiasTransform(100.0)],
)

print(composed.evaluate(np.array([1.0, 2.0])))
```

### Config-driven construction (recommended for CEC/BBOB-style workloads)

`FunctionFactory` builds functions from nested config dicts:

```python
from pyMOFL.factories import DataLoader, FunctionFactory, FunctionRegistry
from pathlib import Path

loader = DataLoader(base_path=Path("src/pyMOFL/constants/cec/2005"))
factory = FunctionFactory(data_loader=loader, registry=FunctionRegistry())

cfg = {
    "type": "bias",
    "parameters": {"value": -450.0},
    "function": {
        "type": "rotate",
        "parameters": {"matrix": "f03/matrix_rotation_D{dim}.txt"},
        "function": {
            "type": "shift",
            "parameters": {"vector": "f03/vector_shift_D{dim}.txt"},
            "function": {
                "type": "sphere",
                "parameters": {"dimension": 10},
            },
        },
    },
}

func = factory.create_function(cfg)
```

This pattern is how the bundled suite JSONs describe complex variants.

## Benchmark Suites

pyMOFL ships with 10 benchmark suite configurations, each defined as a JSON file under `src/pyMOFL/constants/`:

| Suite | Functions | Description |
|-------|-----------|-------------|
| **CEC 2005** | F1-F25 | Unimodal, multimodal, compositions, and hybrids |
| **CEC 2013** | F1-F28 | BBOB-style transforms with oscillation/asymmetric pipelines |
| **CEC 2014** | F1-F30 | Shifted/rotated with golden-data validation at D10/D30/D50 |
| **CEC 2015** | F1-F15 | Subset of CEC 2014 patterns |
| **CEC 2017** | F1-F30 | Extended CEC 2014 with additional compositions |
| **CEC 2020** | F1-F10 | Remapped functions with non-standard hybrid partitions |
| **CEC 2021** | F1-F10 | Streamlined CEC 2020 subset |
| **CEC 2022** | F1-F12 | Latest CEC single-objective suite |
| **BBOB** | F1-F24 | COCO-compatible noiseless functions (+ noisy, mixed-integer, large-scale, and constrained variants via dedicated factories) |
| **GNBG** | - | Generalized Numerical Benchmark Generator |

Suite configs are loaded by `FunctionFactory` or suite-specific factories (e.g., `BBOBSuiteFactory`, `BBOBNoisySuiteFactory`).

## CLI

The `pymofl` CLI provides suite management utilities (requires the `cli` extra):

```bash
# list functions from a bundled suite
uv run pymofl suite list --suite-id cec2005_suite

# filter by category or search term
uv run pymofl suite list --suite-id cec2014_suite --category multimodal
uv run pymofl suite list --suite-id cec2005_suite --search sphere

# validate that all referenced data files exist
uv run pymofl suite validate --suite-id cec2005_suite

# validate a custom suite file
uv run pymofl suite validate --suite /path/to/suite.json --suite-dir /path/to/resources --strict

# JSON output for scripting
uv run pymofl suite list --suite-id cec2005_suite --json
```

## Project Orientation

- Core benchmark functions are first-class classes (e.g., `SphereFunction`) with `@register` aliases for factory lookup.
- Transformations are pure functions (`VectorTransform`, `ScalarTransform`, `PenaltyTransform`) composed around base functions via `ComposedFunction`.
- Compositions combine multiple functions: `WeightedComposition` (Gaussian weighting), `HybridFunction` (input-vector partitioning), and `MinComposition`.
- Suite descriptors live as JSON config under `src/pyMOFL/constants/.../*_suite.json`.
- CEC/BBOB compatibility is handled via shared config conventions, not separate benchmark APIs.
- Bounds are metadata unless you explicitly wrap with `Quantized` from `pyMOFL.functions.transformations`.

## Citation

If you use pyMOFL in your research, please cite it:

```bibtex
@software{silvers2025pymofl,
    author = {Silvers, Travis},
    title = {pyMOFL: Python Modular Optimization Function Library},
    year = {2025},
    url = {https://github.com/firestrand/pyMOFL},
}
```

## License

MIT License. See `LICENSE`.
