# pyMOFL Coding Guidelines

This document outlines the architecture, coding standards, and best practices for contributors to `pyMOFL` (Python Modular Optimization Function Library). Following these guidelines ensures code consistency, maintainability, and alignment with the project's functional composition architecture.

---

## 1. Project Architecture & Directory Layout

`pyMOFL` adopts the standard Python `src-layout`:

```
src/pyMOFL/
  core/                        # Base abstractions: OptimizationFunction, Bounds, Enums, ConstrainedFunction
  functions/
    benchmark/                 # 175 concrete benchmark function classes (Sphere, Rastrigin, Ackley, etc.)
    transformations/           # Pure transformations: Shift, Rotate, Scale, Oscillation, Bias, etc.
  compositions/                # Multi-function compositions: WeightedComposition, HybridFunction, MinComposition
  factories/                   # Config-driven instantiation: FunctionFactory, DataLoader, ConfigParser, Builders
  constants/                   # Benchmark suite JSON configs and data files (CEC 2005-2022, BBOB, GNBG)
  utils/                       # Numerical helpers, rotation generation, BBOB instance generators
  cli/                         # Command-line interface (Typer + Rich)
  registry.py                  # Global component registry with @register decorator and auto-discovery
tests/                         # Test suite mirroring src/
  benchmark_suites/            # Suite-level integration and golden validation tests
  compositions/                # Composition unit tests
  core/                        # Core abstractions tests
  factories/                   # Factory tests
  functions/                   # Benchmark and transformation unit tests
  utils/                       # Shared test helpers: BenchmarkValidator, TransformValidator, GoldenLoader
```

---

## 2. Core Architectural Patterns

### 2.1 Base Class: `OptimizationFunction`

All benchmark functions must inherit from [`OptimizationFunction`](../src/pyMOFL/core/function.py):

* **Evaluation Contract**: Subclasses must implement:
  * `evaluate(x: NDArray) -> float`: Evaluates a single $D$-dimensional point.
  * `evaluate_batch(X: NDArray) -> NDArray`: Vectorized batch evaluation for an $N \times D$ matrix.
  * `get_global_minimum() -> tuple[NDArray, float]`: Returns `(x_opt, f_opt)`.
* **Input Validation**: Subclasses must invoke `self._validate_input(x)` and `self._validate_batch_input(X)` to check input shape, dimension, and data type without modifying input values.
* **Bounds as Metadata**:
  * `initialization_bounds` and `operational_bounds` are data containers defining recommended search domains.
  * Base functions do **not** enforce or clip bounds internally. Boundary penalties can be handled externally via `BoundaryPenaltyTransform`. Quantization handles discrete coordinates without enforcing or clipping domain bounds.

### 2.2 Functional Transformations Pipeline

Transformations compose operators in [`pyMOFL.functions.transformations`](../src/pyMOFL/functions/transformations/). Noise transforms advance RNG state; their ownership, replay and scalar/batch limits are documented in [API compatibility notes](api-compatibility.md).

1. **`VectorTransform` (Input Transformations)**:
   * Subclasses must implement `__call__(x: NDArray) -> NDArray` and `transform_batch(X: NDArray) -> NDArray`.
   * Examples: `ShiftTransform`, `RotateTransform`, `ScaleTransform`, `OscillationTransform`, `AsymmetricTransform`, `DiscretizeTransform`.
2. **`ScalarTransform` (Output Transformations)**:
   * Subclasses must implement `__call__(val: float) -> float` and `transform_batch(vals: NDArray) -> NDArray`.
   * Examples: `BiasTransform`, `PowerTransform`, `NoiseTransform`, `NormalizeTransform`.
3. **`PenaltyTransform` (Additive Penalties)**:
   * Subclasses take raw input vectors and return a scalar penalty: `__call__(x: NDArray) -> float`.
   * Example: `BoundaryPenaltyTransform` for BBOB boundary penalties.

### 2.3 Function Composition: `ComposedFunction`

Complex benchmark landscapes (such as shifted/rotated/biased variants in CEC and BBOB) are constructed via [`ComposedFunction`](../src/pyMOFL/functions/transformations/composed.py):

```python
composed = ComposedFunction(
    base_function=SphereFunction(dimension=10),
    input_transforms=[
        ShiftTransform(shift_vector),
        RotateTransform(rotation_matrix),
    ],
    output_transforms=[BiasTransform(100.0)],
    penalty_transforms=[BoundaryPenaltyTransform(bound=5.0)],
)
```

Evaluation order:
$$\mathbf{z} = T_{\text{input}, k}(\dots(T_{\text{input}, 1}(\mathbf{x})))$$
$$y = f_{\text{base}}(\mathbf{z})$$
$$y' = T_{\text{output}, m}(\dots(T_{\text{output}, 1}(y)))$$
$$f(\mathbf{x}) = y' + \sum_j P_j(\mathbf{x})$$

### 2.4 Multi-Function Composites

Located in [`pyMOFL.compositions`](../src/pyMOFL/compositions/):
* **`WeightedComposition`**: Combines multiple functions using dynamic Gaussian distance-based weights (standard in CEC 2005/2014/2017 composition functions).
* **`HybridFunction`**: Partitions input dimensions across multiple sub-functions with distinct rotation/scaling matrices.
* **`MinComposition`**: Computes the lower envelope (minimum) across multiple attraction basins (used in GNBG).

### 2.5 Config-Driven Instantiation & Registry

* **Unified Component Registry**: Every concrete benchmark function is decorated with `@register("alias_name")` in [`pyMOFL.registry`](../src/pyMOFL/registry.py).
* **`FunctionFactory`**: Constructs complete composed or composite functions from nested JSON configurations, automatically resolving data matrices and shift vectors via `DataLoader`.

---

## 3. Code Style & Standards

1. **Python Version**: Python 3.12+ compatible. Use modern typing syntax:
   * `X | None` instead of `Optional[X]`
   * `list[T]`, `dict[K, V]`, `tuple[T, ...]` instead of `typing.List`, etc.
2. **Formatting & Linting**:
   * Formatted and checked with `ruff`.
   * Line length: 100 characters.
   * Run before committing:
     ```bash
     uv run ruff format src/ tests/
     uv run ruff check src/ tests/
     ```
3. **Vectorized NumPy**:
   * Avoid Python loops in `evaluate_batch`. Use NumPy broadcasting and vectorized operations.
   * Explicitly specify data types (`dtype=np.float64`) for numerical stability.
4. **Documentation**:
   * NumPy docstring format on all public classes, methods, and functions.
   * Provide mathematical formulas, default bounds, global minimum $(x^*, f^*)$, and literature references (e.g. CEC tech reports, BBOB papers).

---

## 4. Testing Guidelines

1. **Test Organization**:
   * Tests in `tests/` strictly mirror `src/pyMOFL/`.
   * Unit tests for each benchmark function live in `tests/functions/benchmark/test_<name>.py`.
2. **Shared Test Helpers**:
   * Use [`BenchmarkValidator`](../tests/utils/benchmark_validation.py) for every benchmark function to verify standard contracts:
     ```python
     BenchmarkValidator.assert_contract(func)
     BenchmarkValidator.assert_contract_multiple_dimensions(FunctionClass, dimensions=[2, 5, 10])
     ```
   * Use [`TransformValidator`](../tests/utils/transform_validation.py) for new transformation classes.
3. **Numerical Accuracy**:
   * Use `pytest.approx` or `np.testing.assert_allclose`.
   * For known reference deviations or upstream library bugs (documented in [`xfail_analysis.md`](xfail_analysis.md)), use `pytest.xfail` with a detailed explanation.
