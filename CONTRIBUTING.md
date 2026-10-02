# Contributing to pyMOFL

Thank you for your interest in contributing to **pyMOFL** (Python Modular Optimization Function Library)! We welcome contributions from researchers, software engineers, and optimization enthusiasts.

---

## 1. Getting Started

### Prerequisites
* Python 3.12 or newer.
* [`uv`](https://docs.astral.sh/uv/) (recommended package and dependency manager).

### Environment Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/firestrand/pyMOFL.git
   cd pyMOFL
   ```

2. Create virtual environment and install dependencies:
   ```bash
   uv sync --extra dev --extra cli
   ```

3. Verify test suite:
   ```bash
   uv run pytest
   ```

---

## 2. Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Implement your changes adhering to the [Coding Guidelines](CODING_GUIDELINES.md).

3. Format and lint your code:
   ```bash
   uv run ruff format src/ tests/
   uv run ruff check src/ tests/
   ```

4. Run the full test suite and verify test coverage:
   ```bash
   uv run pytest
   ```

---

## 3. Adding a New Benchmark Function

To add a new benchmark function:

1. **Implement Function**: Create `src/pyMOFL/functions/benchmark/<function_name>.py`:
   * Inherit from `OptimizationFunction`.
   * Implement `evaluate(x: NDArray) -> float`.
   * Implement `evaluate_batch(X: NDArray) -> NDArray` with vectorized NumPy operations.
   * Implement `get_global_minimum() -> tuple[NDArray, float]`.
   * Register with `@register("<function_name>")` and `@register("<PascalCaseName>")`.
   * Provide docstrings following NumPy format, citing academic sources.

2. **Add Unit Tests**: Create `tests/functions/benchmark/test_<function_name>.py`:
   * Use `BenchmarkValidator.assert_contract(func)` from `tests.utils.benchmark_validation`.
   * Use `BenchmarkValidator.assert_contract_multiple_dimensions(...)` for scalable functions.
   * Test evaluations at the origin, known global optima, and known reference points.

---

## 4. Adding or Modifying Benchmark Suites

* Suite JSON definitions reside under `src/pyMOFL/constants/<suite_name>/`.
* Numerical matrices (shift vectors, rotation matrices) are placed alongside the suite config or referenced via `DataLoader`.
* When writing suite-level tests, validate functions against golden reference outputs if available.
* Document any intentional deviations from reference code (e.g. C-code bugs, float64 precision limits) in [`docs/xfail_analysis.md`](docs/xfail_analysis.md).

---

## 5. Submitting a Pull Request

* Ensure code passes `ruff check` and `ruff format`.
* Ensure all tests pass (`uv run pytest`).
* Write clear, descriptive commit messages.
* Open a Pull Request detailing the purpose, changes made, and test verification results.
