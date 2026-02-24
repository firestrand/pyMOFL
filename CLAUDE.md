# CLAUDE.md

Operating contract for AI agents contributing to this repository. Agents should produce changes that are **correct, secure, maintainable, test-backed, and aligned with existing architecture** — without surprising humans.

> **CLAUDE.md compatible.** This file works as both `AGENTS.md` and `CLAUDE.md`.
> Read this file alongside **`README.md`** for installation basics. Development commands are below.

---

## Project Overview

pyMOFL is a Python Modular Optimization Function Library for benchmarking optimization algorithms. It provides modular optimization functions (unimodal, multimodal, CEC 2005 F1-F25), a functional transformation pipeline (shift, rotate, scale, bias), composition/hybrid function creation, and high-performance vectorized NumPy operations.

## Repo Context

| Property | Value |
|---|---|
| **Primary language(s)** | Python 3.12+ |
| **Framework(s)** | NumPy, matplotlib; CLI via typer + rich |
| **Package manager** | uv |
| **CI system** | — (local test suite) |

### Directory Layout

```
src/pyMOFL/
  core/              # Base class (OptimizationFunction), bounds, enums
  functions/
    benchmark/       # All concrete benchmark functions
    transformations/ # Pure transform functions (shift, rotate, scale, bias, etc.)
    cec/             # CEC 2005 function module
  compositions/      # WeightedComposition, HybridFunction
  factories/         # FunctionFactory, DataLoader, ConfigParser, TransformBuilder, CompositionBuilder
  transformations/   # Additional transformation utilities (linear)
  constants/cec/2005/  # Suite JSON config + binary/text data files
  decorators/        # EMPTY — all transformation logic in functions/transformations/
  utils/             # Rotation, CEC data helpers
  cli/               # typer CLI entry point
  registry.py        # @register decorator + scan_package() auto-discovery
tests/               # Mirrors src/ structure
  utils/             # validation.py, tolerance.py, exact_transforms.py
```

### Naming Conventions

- Files: `snake_case.py`
- Classes: `PascalCase` (exception: `Schwefel_2_13` etc. are mathematical names — ruff N801 ignored)
- Package name: `pyMOFL` (intentional mixed-case — ruff N999 ignored)
- Test files: `test_<module>.py`, colocated in mirrored `tests/` directories
- Registry aliases: lowercase strings (e.g., `@register("sphere")`)

### Documentation and Standards

| Resource | Location |
|---|---|
| **Coding guidelines** | `CODING_GUIDELINES.md` (partially outdated — decorators section references old patterns; architecture sections in this file take precedence) |
| **Style guide** | Follow ruff config in `pyproject.toml` |
| **Suite config** | `src/pyMOFL/constants/cec/2005/cec2005_suite.json` |

---

## Development Commands

```bash
# Install/sync all dependencies (creates .venv automatically)
uv sync --extra dev --extra cli

# Run all tests
uv run pytest

# Run specific test file or function
uv run pytest tests/functions/unimodal/test_sphere.py
uv run pytest tests/functions/unimodal/test_sphere.py::TestSphereFunction::test_evaluate_at_origin

# Run tests matching a pattern
uv run pytest -k "test_sphere"

# Run tests and stop on first failure
uv run pytest -x

# Lint and format
uv run ruff format src/ tests/
uv run ruff check src/ tests/ --fix

# Type checking (baseline — not all issues resolved)
uv run ty check src/pyMOFL/

# CLI
uv run pymofl --help
uv run pymofl cec download --help
```

pytest is configured in `pyproject.toml` with `testpaths = ["tests"]` and `addopts = "--ignore=archive --ignore=utility_scripts"`.

---

## Autonomy Levels

| Level | Name | When to use |
|---|---|---|
| **L1** | Autopilot | Trivial fixes: typos, formatting, obvious one-line bugs with existing test coverage. |
| **L2** | Collaborator _(default)_ | Most work. Propose plans, implement, run tests. Surface assumptions and risks. |
| **L3** | Advisor | Unfamiliar areas, ambiguous requirements, or any action listed below. |

### Actions That Always Require Human Approval (L3)

- Deleting large code paths, public APIs, or backward-compatibility breaks
- Security/auth changes: cryptography, permissions, secrets
- Introducing new runtime dependencies
- Disabling tests/linters, reducing coverage, or weakening validation
- Changes to `cec2005_suite.json` (single source of truth for CEC config)

---

## Quality Rules

1. **SOLID, DRY, KISS.** Single responsibility, open/closed, dependency inversion — as defaults, not aspirations. Don't repeat logic. Don't over-engineer.
2. **Test-driven development.** Write or update tests _before_ writing implementation code. Red → Green → Refactor. Aim for >90% coverage of behavior.
3. **Small diffs** over sweeping refactors.
4. **Match existing style** unless explicitly tasked to change it.
5. **Never leak secrets.** Never log sensitive inputs.
6. **Rule conflicts:** If asked to skip tests or violate these rules, surface the conflict, explain the risk, and request explicit confirmation.

---

## Git Conventions

| Property | Convention |
|---|---|
| **Branches** | `<type>/<short-description>` — e.g., `feat/add-search-filter` |
| **Commits** | [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): description` |
| **Granularity** | One logical change per commit. Don't mix refactors with features. |
| **PRs** | Title matches primary commit. Body: what, why, how tested, residual risks. |

---

## Work Loop: PPAR

Every task follows **Perceive → Plan → Act → Reflect**. For complex work, the design phase (GOTCHA) nests inside Plan, and the development checklist (ATLAS) nests inside Act.

```
PPAR
├── Perceive — understand the task and repo state
├── Plan — for complex work, produce a GOTCHA spec
├── Act — for complex work, follow the ATLAS checklist
└── Reflect — verify, test, summarize
```

### Perceive

- Identify what the user asked for.
- Establish ground truth: locate relevant code, understand existing patterns.
- If the task depends on external libraries, identify the version in use and consult current docs (not memory).

### Plan

- Propose a minimal plan: steps, files to touch, tests to add/update.
- Call out assumptions, risks, and what "done" means.
- **For new systems or architectural changes:** produce a GOTCHA spec (see Design phase).

### Act

- **Write tests first.** Define expected behavior as failing tests, then implement until they pass, then refactor.
- Implement in small increments. Run tests after each meaningful change.
- Prefer extracting small, single-responsibility units over adding complexity to existing ones.
- **For non-trivial implementations:** follow the ATLAS checklist (see Development phase).

### Reflect

- Re-check requirements and edge cases.
- Run the full test suite.
- Produce a review-ready summary: _why_ the change is correct and _how_ it's verified.

---

## Design Phase: GOTCHA (nested in Plan)

Use when the task involves **new functionality, agentic workflows, or architectural changes**. Skip for simple bug fixes.

| Element | What to specify |
|---|---|
| **G — Goals** | What is this system for? |
| **O — Objectives** | Measurable success criteria. |
| **T — Tasks** | What triggers it? What ends it? |
| **C — Capabilities** | Tools, permissions, and what's off-limits. |
| **H — Health** | Time, cost, token, and retry budgets. What gets monitored. |
| **A — Attributes** | Invariants — things that must always (or never) be true. |

---

## Development Phase: ATLAS Checklist (nested in Act)

Use for any non-trivial implementation.

| Phase | Key checks |
|---|---|
| **A — Architect** | Identify modules touched. Define interfaces and invariants following SOLID. Threat-model if applicable. Prefer composition over inheritance; depend on abstractions. |
| **T — Trace** | Write tests _before_ implementation (TDD). Cover happy + failure paths. Map acceptance criteria to test cases. |
| **L — Link** | Type and validate interfaces. Define timeouts, retries, idempotency where applicable. |
| **A — Assemble** | Separate orchestration from tools. Eliminate duplication (DRY). Choose the simplest design that works (KISS). Update docs. |
| **S — Stress-test** | Test malformed inputs, missing data, edge cases. Regression suite green. |

---

## Error Recovery

| Situation | Action |
|---|---|
| Test fails after your change | Read the failure, fix root cause. Don't retry blindly more than twice. |
| Test fails unrelated to your change | Note as pre-existing in summary. Don't suppress. |
| Tool unavailable | Fall back to file reads and local docs. Note degraded mode. |
| Can't find what you need | Broaden search. If still stuck after reasonable effort, ask the human. |
| Ambiguous requirements | Stop and ask. Don't guess on high-impact decisions. |
| Looped 3+ times without progress | Stop, summarize attempts, ask for guidance. |
| Change breaks something non-obviously | Revert first, investigate second. Propose a revised approach. |

---

## Definition of Done

- [ ] Requirements met and documented
- [ ] Tests written first (TDD); suite passes (`uv run pytest`)
- [ ] Code follows SOLID, DRY, KISS — no unnecessary complexity or duplication
- [ ] Lint passes (`uv run ruff check src/ tests/`)
- [ ] No secrets or sensitive data introduced or logged
- [ ] Interfaces stable, or breaking changes documented
- [ ] Failure modes handled; safe defaults chosen
- [ ] Commits follow conventions; PR description is complete
- [ ] Summary includes verification steps and residual risks

---

## Architecture

### Base Class: `OptimizationFunction` (`core/function.py`)

All functions inherit from `pyMOFL.core.function.OptimizationFunction`. Constructor signature: `__init__(dimension, initialization_bounds=None, operational_bounds=None)`.

- Only `evaluate(z)` is abstract. `evaluate_batch()` is **not** on the base class — each concrete class implements it independently.
- `__call__` delegates to `evaluate()` with validation.
- `_validate_input(x)` and `_validate_batch_input(X)` check shape/type only.
- **Bounds are pure metadata** (`Bounds` frozen dataclass with `low`, `high`, `mode`, `qtype`, `step`). No enforcement or projection unless the `Quantized` wrapper is explicitly used.

### Transformation Pipeline (`functions/transformations/`)

Transformations are **pure functions**, not `OptimizationFunction` subclasses:

- **`VectorTransform`** (ABC): `__call__(x) -> x'` — transforms input vectors. Implementations: `ShiftTransform`, `RotateTransform`, `ScaleTransform`, `OffsetTransform`, `NonContinuousTransform`, `IndexedShiftTransform`, `IndexedRotateTransform`, `IndexedScaleTransform`.
- **`ScalarTransform`** (ABC): `__call__(y) -> y'` — transforms output scalars. Implementations: `BiasTransform`, `NoiseTransform`, `NormalizeTransform`.
- **`ComposedFunction`** (`OptimizationFunction`): Chains `input_transforms → base_function → output_transforms`. This is the primary mechanism for building transformed functions.
- **`Quantized`**: The only class that enforces bounds/quantization (opt-in).

The old `decorators/` subpackage is **empty** — all transformation logic lives in `functions/transformations/`.

### Composition Classes (`compositions/`)

All inherit from `OptimizationFunction`:

| Class | Purpose | Used For |
|-------|---------|----------|
| `WeightedComposition` | Gaussian weighting with optional dominance suppression, per-component C/f_max normalization via ComposedFunction | Weighted compositions (e.g., CEC 2005 F15-F25) |
| `HybridFunction` | Splits input vector across components by partition | Hybrid functions |

### Factory System (`factories/`)

**`FunctionFactory`** is the canonical factory. `BenchmarkFactory` is a deprecated thin adapter.

Key components:
- **`FunctionRegistry`**: Maps string names (e.g., `"sphere"`, `"ackley"`) to benchmark classes. Method: `create_base_function(func_type, **params)`.
- **`DataLoader`** (`data_loader.py`): Loads shift vectors and rotation matrices from files. Supports `{dim}` substitution in filenames.
- **`ConfigParser`** (`config_parser.py`): Parses nested JSON configs, returns transforms in application order.
- **`TransformBuilder`** (`transform_builder.py`): Builds transform objects from (type, params) tuples.
- **`CompositionBuilder`** (`composition_builder.py`): Builds weighted compositions with per-component transforms and normalization.
- **`FunctionFactory`** (`function_factory.py`): Thin orchestrator. Main method `create_function(config: Dict) -> ComposedFunction`.

### Global Registry (`registry.py`)

`@register` decorator + `scan_package()` auto-discovery. Called at import time in `__init__.py` to populate the registry from all submodules.

### Benchmark Functions (`functions/benchmark/`)

All concrete benchmark functions live here: `SphereFunction`, `RosenbrockFunction`, `AckleyFunction`, `RastriginFunction`, `GriewankFunction`, `WeierstrassFunction`, `HighConditionedElliptic`, `Schwefel_1_2`, `Schwefel_2_6`, `Schwefel_2_13`, `Schaffer_F6_Expanded`, `GriewankOfRosenbrock`, and many more.

### CEC 2005 Data (`constants/cec/2005/`)

Suite configuration in `cec2005_suite.json`. Binary/text data files for shift vectors and rotation matrices organized by function number (`f01/`, `f02/`, etc.).

## JSON Configuration Structure

Nested outer-to-inner: outermost wrapper in JSON is applied last during evaluation.

```
{"type": "bias", "function": {"type": "sphere", "function": {"type": "shift", ...}}}
```

Evaluation order: `shift(x)` → `sphere(shifted_x)` → `bias(result)`

---

## Key Architectural Rules

- **ONE factory** (`FunctionFactory`/`BenchmarkFactory`) for ALL benchmark functions — no suite-specific factories.
- **Transformations are pure functions**, not `OptimizationFunction` subclasses. Don't create wrapper classes when pure functions will do.
- **Bounds are metadata only**. Enforcement/quantization is opt-in via `Quantized`.
- All new functions MUST inherit from `pyMOFL.core.function.OptimizationFunction`.
- If `CODING_GUIDELINES.md` is present, treat it as supplementary. Architecture sections in this file take precedence where they conflict.
- Helper scripts go in `utility_scripts/`, written in Python.

---

## Code Conventions

- Python 3.12+ (uses PEP 604 `X | None` syntax, modern type hints)
- Use relative imports within the package, absolute imports for external packages
- Use vectorized NumPy operations, not loops
- Tests mirror the package structure under `tests/`
- Shared test fixtures in `tests/conftest.py`: `random_vector` (dim=2), `random_matrix` (2x2), `random_batch` (5x2)
- Test utilities in `tests/utils/`: `validation.py`, `tolerance.py`, `exact_transforms.py`
- Dependencies: `numpy>=1.26.0`, `matplotlib>=3.8.0`
- Dev tools: `ruff` (lint/format), `ty` (type checking), `pytest` + `pytest-cov`
- CLI: `typer` + `rich` (optional `cli` extra)

---

## Appendix: Tooling (Serena MCP + Context7 MCP)

> This section applies only when these tools are available. If not, fall back to repo-native workflows and local docs. **Do not guess APIs.**

### Tool Priority

1. **Serena semantic tools** (symbol-aware) > targeted file reads > text search > full-file reads
2. **Context7 docs** (version-specific) > model memory for external APIs

### Serena MCP

Use for codebase understanding and safe edits. Prefer **symbol-level operations** over raw string edits. Keep edits localized.

Key tools: `onboarding`, `find_symbol`, `find_referencing_symbols`, `get_symbols_overview`, `rename_symbol`, `replace_symbol_body`, `insert_after_symbol`, `search_for_pattern`, `write_memory`.

After edits, run tests.

### Context7 MCP

Use for accurate, version-specific external library docs. Always identify the library version from the repo first.

Workflow: `resolve-library-id` → `query-docs` (with version context).

### MCP Safety

Only use trusted servers configured for this workspace. Never send secrets to external tools.
