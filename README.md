# pyMOFL: Python Modular Optimization Function Library

`pyMOFL` is a modular benchmark-function library for optimization research.
It now uses a **function-composition architecture** driven by declarative configs.
The stack is managed with `uv`, and lint/type checks are run via `ruff` and `ty`.

## Installation

From source:

```bash
git clone https://github.com/yourusername/pyMOFL.git
cd pyMOFL
uv sync
```

Run the CLI with the project environment:

```bash
uv run pymofl --help
```

## Project orientation

- Core benchmark functions are first-class classes (e.g. `SphereFunction`).
- Compositions are explicit: transformations are composed around a base function.
- Suite descriptors live as JSON config under `src/pyMOFL/constants/.../*_suite.json`.
- CEC/BBOB compatibility is handled via shared config conventions, not separate benchmark APIs.

## Quick start (direct functions)

```python
import numpy as np
from pyMOFL.functions.benchmark import SphereFunction, RastriginFunction

sphere = SphereFunction(dimension=2)
print(sphere.evaluate(np.array([0.0, 0.0])))

rastrigin = RastriginFunction(dimension=2)
print(rastrigin.evaluate(np.array([1.0, 2.0])))
```

## Compositions (shift, rotate, bias)

Transformations are composed explicitly in code via `ComposedFunction`.

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

The same structure is represented in JSON config as nested transform/function nodes.

## Config-driven construction (recommended for CEC/BBOB-style workloads)

`FunctionFactory` builds functions from config objects.

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

This pattern is how the shipped suite JSONs describe complex variants.

## Suite utility CLI (generic)

`suite` is the utility surface, intentionally benchmark-family agnostic:

```bash
# list functions from the bundled CEC 2005 suite config
uv run pymofl suite list --suite-id cec2005_suite

# validate that all referenced suite resources exist
uv run pymofl suite validate --suite-id cec2005_suite

# validate a custom suite file
uv run pymofl suite validate --suite /path/to/bbob_suite.json --suite-dir /path/to/bbob/resources --strict
```

There is no benchmark-specific `cec` CLI group anymore.

## Notes

- Bounds are metadata unless you explicitly wrap/transform with quantization or enforce bounds in your optimizer.
- For quantization, use `Quantized` from `pyMOFL.functions.transformations`.
- `setup.py` is not part of this workflow; package/build is driven from `pyproject.toml` and `uv`.

## License

MIT License. See `LICENSE`.
