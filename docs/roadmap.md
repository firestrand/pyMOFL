# pyMOFL Roadmap

Last updated: 2026-10-04

This document outlines planned future work for pyMOFL. Items are grouped by priority and theme. Completed phases are listed for context; active and planned work is marked accordingly.

## Version 0.4.0 improvements

The [implementation evidence](implementation-progress.md) records verified
changes delivered in v0.4.0: discovery reuse, single-call batch failures, live
suite lookup, explicit RNG injection, supported bounded evaluation, four selected
SPSO definitions, deterministic definition records, and actual reference/report
execution. The [proposal](improvement-proposal.md) preserves full scope and
accepted scope, source limitations and verified local gates. Numerical optimization
defaults were retained after the measured review; broader acceleration remains
research work.

---

## Completed

This section preserves the inherited v0.3.0 milestone record. Current verification
does not establish every historical release/reference claim; the implementation
evidence states which source datasets and checks were actually provisioned.

| Milestone | Description |
|-----------|-------------|
| Core architecture | OptimizationFunction base, transform ABCs, ComposedFunction, factory system |
| CEC 2005 suite | F1-F25 with config-driven factory, golden validation |
| CEC 2013 suite | F1-F28 with BBOB-style transforms, fused asymmetric pipeline |
| CEC 2014 suite | F1-F30, golden validation at D10/D30/D50 |
| CEC 2015 suite | F1-F15, golden validation |
| CEC 2017 suite | F1-F30 (SchafferF7 C-code bug xfailed) |
| CEC 2020 suite | F1-F10 with remapping and non-standard hybrid partitions |
| CEC 2021 suite | F1-F10 |
| CEC 2022 suite | F1-F12 |
| BBOB noiseless | F1-F24 with COCO-compatible instance generation |
| BBOB noisy | 30 functions (10 base x 3 noise models) |
| BBOB mixed-integer | 24 functions with DiscretizeTransform |
| BBOB large-scale | 24 functions with block-diagonal rotations |
| BBOB constrained | 54 functions (9 objectives x 6 constraint configs) |
| Classical benchmarks | Registered benchmark classes and aliases; see the generated catalog for the current inventory |
| GNBG suite factory | GNBGSuiteFactory for 24 problem instances with MinComposition |
| Documentation & Catalog | Function catalog (function_catalog.md), updated coding guidelines, CONTRIBUTING.md, and quickstart notebook |
| CEC 2015 Niching suite | F1-F15 with 8 multimodal niching base functions, compositions, and golden C validation |
| CEC 2019 suite | F1-F10 (100-Digit Challenge) with Chebyshev, Hilbert, Lennard-Jones |
| CEC 2024 suite | 29 active functions evaluated at 30D with U-score protocol |
| CEC 2025 suite | 24 GNBG-II instances with GNBGSuiteFactory and config-driven pipeline |
| CEC 2008 LSGO suite | F1-F7 scalable to D in {100, 500, 1000} with FastFractalDoubleDip and Schwefel 2.21 |
| CEC 2010 LSGO suite | F1-F20 (1000D) with cooperative decomposition pipeline and block rotation matrices |

---

## Completed: CLI Suite Listing Polish & Function Catalog

- [x] Add GNBG to CLI suite listing (`pymofl suite list --suite-id gnbg_suite`)
- [x] Function catalog/index page listing all functions by category with registry aliases
- [x] Final refactoring pass for DRY violations and dead code

---

## Completed: CEC 2024

**CEC 2024 Competition on Single Objective Numerical Optimization** reuses the core CEC primitives already implemented in pyMOFL:

- [x] Obtain CEC 2024 technical report and reference code
- [x] Create `constants/cec/2024/` directory with shift vectors and rotation matrices
- [x] Create `cec2024_suite.json` suite configuration (29 active problems at D=30)
- [x] Validate against competition definition and reference outputs (`tests/benchmark_suites/test_cec2024_suite.py`)

---

## Completed: CEC 2025

**CEC 2025** adopts the **GNBG-II** framework as its standard benchmark suite:

- [x] `GNBGSuiteFactory` provides the 24 GNBG-II instances
- [x] Integrate CEC 2025 competition parameter files (24 GNBG-II instances)
- [x] Create `constants/cec/2025/` directory with GNBG parameter data
- [x] Create `cec2025_suite.json` suite configuration
- [x] Validate against GNBG reference implementation (`tests/benchmark_suites/test_cec2025_suite.py`)

---

## Completed: CEC 2015 Niching

CEC 2015 Multimodal Optimization competition uses 8 specialized "expanded" base functions and 7 composition functions:

| Function | Base Dim | Notes |
|----------|:--------:|-------|
| Expanded Two-Peak Trap | 1D->D | Piecewise with quadratic boundary extension |
| Expanded Five-Uneven-Peak Trap | 1D->D | Exact C piecewise definition |
| Expanded Equal Minima | 1D->D | 5^D equal global minima |
| Expanded Decreasing Minima | 1D->D | Exponentially modulated uneven minima |
| Expanded Uneven Minima | 1D->D | Scaled non-uniform minima |
| Expanded Himmelblau | 2D->D | Non-overlapping pairs |
| Expanded Six-Hump Camel Back | 2D->D | Non-overlapping pairs, normalized minimum |
| Modified Vincent | Scalable | Logarithmic scaling with boundary penalty |

- [x] Implement 8 niching base functions (`src/pyMOFL/functions/benchmark/niching.py`)
- [x] Extract input data and matrices into `constants/cec/2015_niching/`
- [x] Create `cec2015_niching_suite.json`
- [x] Validate against CEC 2015 niching compiled C reference (`tests/benchmark_suites/test_cec2015_niching_suite.py`)

---

## Completed: CEC 2008 & CEC 2010 (Large-Scale Global Optimization Suites)

- [x] Implement `FastFractalDoubleDip` (CEC 2008 F7) verified against C++/Java reference
- [x] Register `Schwefel_2_21` as an official scalable benchmark alias
- [x] Package shift vectors and configure `cec2008_suite.json` (F1-F7, D in {100, 500, 1000})
- [x] Build variable grouping and decomposition pipeline (`GroupingTransform`, `DecomposedTransform`, `DecomposedFunction`)
- [x] Extract 1000D shift vectors, permutation vectors, and 50x50 rotation matrices into `constants/cec/2010/`
- [x] Configure `cec2010_suite.json` across all 20 functions (fully separable, single group m=50, D/(2m) groups, D/m groups, and fully non-separable)
- [x] Validate against official competition code and verify global minimum recovery (`tests/benchmark_suites/test_cec2008_suite.py`, `tests/benchmark_suites/test_cec2010_suite.py`)

---

## Completed: CEC 2013 LSGO (Large-Scale Global Optimization Suite)

- [x] Package official shift vectors, permutation vectors, and 25x25, 50x50, 100x100 block rotation matrices into `constants/cec/2013_lsgo/`
- [x] Extend decomposition infrastructure with `GroupingTransform.from_overlapping_sizes` for non-uniform subcomponent block sizes sharing variables
- [x] Implement per-group conflicting shift support in `ComponentGroup` and `GroupingTransform` (for overlapping F14)
- [x] Configure `cec2013_lsgo_suite.json` across all 15 functions (Fully Separable F1-F3, Partially Additively Separable F4-F11 with 7-20 subcomponents, Overlapping F12-F14 at D=905/1000, and Fully Non-Separable F15)
- [x] Validate against official compiled C++ competition reference code down to machine precision (< 1e-12) (`tests/benchmark_suites/test_cec2013_lsgo_suite.py`)

---

## Backlog

### BBOB Bi-Objective (bbob-biobj)

Requires multi-objective optimization framework (`evaluate` returning vector of objectives, Pareto front computation). Deferred until multi-objective support is architecturally designed.

### SPSO Benchmark Functions

The local implementation provides four selected native-coordinate definitions
from pinned SPSO 2007/2011 sources:

- Tripod (2D, piecewise linear, explicit zero-axis source semantics)
- Network (42D: 38 binary links and four continuous coordinates)
- Gear Train (4D integer)
- Compression Spring (3D mixed-integer, constrained)

Original controls and all 546 source-backed half-step/axis cases pass.
Fixed-factory rejection checks and current coverage are verified; the implementation
is included in v0.4.0. Other source functions and optimizer algorithms remain outside this
selected implementation. [Source review](spso-reference-review.md) records
the exact versions, historical Spring penalty difference and rights limits.

### Additional Benchmark Libraries

Potential future integration with function sets from:
- SOCO (Soft Computing special issue benchmarks)
- CEC-C (constrained optimization, various years)
- LSGO (Large-Scale Global Optimization workshops)

---

## Non-Functional Improvements

### Performance

- [ ] Benchmark evaluation throughput for high-dimensional functions (D>100)
- [ ] Profile and optimize hot paths in transform pipeline
- [ ] Consider optional Numba/JAX backends for batch evaluation

### Testing

- [ ] Increase coverage reporting granularity (per-module)
- [ ] Add property-based tests for transform invertibility where applicable
- [ ] Fuzz testing for input validation edge cases

### Packaging and Distribution

- [ ] Publish to PyPI
- [x] CI configuration (GitHub Actions) with locked checks and installed-artifact gates; remote execution remains unobserved in this work
- [x] Generate API documentation setup (MkDocs configuration in mkdocs.yml)
- [ ] Add type stubs or improve ty/mypy compliance

### Developer Experience

- [ ] Pre-commit hooks for ruff format/check
- [x] Contribution guide (CONTRIBUTING.md)
- [x] Example notebooks (Jupyter) demonstrating common workflows (examples/quickstart_walkthrough.ipynb)
