# pyMOFL Roadmap

Last updated: 2026-04-06

This document outlines planned future work for pyMOFL. Items are grouped by priority and theme. Completed phases are listed for context; active and planned work is marked accordingly.

---

## Completed

These milestones are done and shipped in v0.3.0.

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
| Classical benchmarks | 175 registered classes (336 aliases) |
| GNBG suite factory | GNBGSuiteFactory for 24 problem instances with MinComposition |
| Documentation & Catalog | Function catalog (docs/function_catalog.md), updated coding guidelines, CONTRIBUTING.md, and quickstart notebook |

---

## In Progress

### CLI Suite Listing Polish
- [x] Add GNBG to CLI suite listing (`pymofl suite list --suite-id gnbg_suite`)
- [x] Function catalog/index page listing all functions by category with registry aliases
- [ ] Final refactoring pass for DRY violations and dead code

---

## Planned: CEC 2024

**CEC 2024 Competition on Single Objective Numerical Optimization** reuses the core CEC primitives already implemented in pyMOFL. Requires:

- [ ] Obtain CEC 2024 technical report and reference code
- [ ] Create `constants/cec/2024/` directory with shift vectors and rotation matrices
- [ ] Create `cec2024_suite.json` suite configuration
- [ ] Generate or obtain golden validation datasets
- [ ] Validate against reference implementation

**Expected scope:** ~30 functions (shifted/rotated/composed variants of existing base functions). No new base function code anticipated -- all 26 core CEC primitives are implemented.

---

## Planned: CEC 2025

**CEC 2025** adopts the **GNBG-II** framework as its standard benchmark suite. This is a significant departure from previous CEC years that used fixed-formula functions with shift/rotate transforms.

- [x] `GNBGSuiteFactory` provides the 24 GNBG-II instances
- [ ] Obtain CEC 2025 competition parameter files (24 GNBG-II instances)
- [ ] Create `constants/cec/2025/` directory with GNBG parameter data
- [ ] Create `cec2025_suite.json` suite configuration
- [ ] Validate against GNBG reference implementation

**Expected scope:** 24 functions defined via GNBG parametric generator. The `MinComposition` class and GNBG transform pipeline handle construction; the work is primarily data integration and validation.

---

## Completed: CEC 2019

CEC 2019 "100-Digit Challenge" (10 functions):

- [x] Implement `ChebyshevFunction` (polynomial fitting, D=9 or D=16)
- [x] Implement `HilbertFunction` (matrix norm, D=16)
- [x] Implement `LennardJonesCECFunction` and support `dimension` kwarg in `LennardJonesFunction`
- [x] Create `cec2019_suite.json` with all 10 functions (F1-F10)
- [x] Extract reference data (rotation matrices and shift vectors) into `constants/cec/2019/`
- [x] Validate against official C reference implementation (`tests/benchmark_suites/test_cec2019_suite.py`)


---

## Planned: CEC 2015 Niching

CEC 2015 Multimodal Optimization competition uses 8 specialized "expanded" base functions not yet implemented:

| Function | Base Dim | Notes |
|----------|:--------:|-------|
| Expanded Two-Peak Trap | 1D->D | Sum-of-pairs pattern |
| Expanded Five-Uneven-Peak Trap | 1D->D | |
| Expanded Equal Minima | 1D->D | |
| Expanded Decreasing Minima | 1D->D | |
| Expanded Uneven Minima | 1D->D | |
| Expanded Himmelblau | 2D->D | Base `HimmelblauFunction` exists |
| Expanded Six-Hump Camel Back | 2D->D | Base `SixHumpCamelFunction` exists |
| Modified Vincent | Scalable | |

- [x] Implement 8 niching base functions (`src/pyMOFL/functions/benchmark/niching.py`)
- [ ] Create `cec2015_niching_suite.json`
- [ ] Validate against CEC 2015 niching technical report

---

## Backlog

### CEC 2008 Large-Scale Optimization

7 functions requiring 2 new base implementations (`FastFractalDoubleDip`, `Schwefel_2_21`) and variable grouping infrastructure for large-scale (D=100-1000) problems.

### CEC 2010 Large-Scale Optimization

20 functions requiring variable grouping infrastructure (cooperative decomposition). Dimensions up to D=1000. Significant architectural work for the grouping/decomposition framework.

### BBOB Bi-Objective (bbob-biobj)

Requires multi-objective optimization framework (`evaluate` returning vector of objectives, Pareto front computation). Deferred until multi-objective support is architecturally designed.

### SPSO Benchmark Functions

Standard Particle Swarm Optimisation suites (SPSO 2007 and SPSO 2011). Most functions are already implemented as CEC 2005 shifted variants or standalone classical functions. Remaining unique functions:

- Tripod (2D, piecewise quadratic)
- Network (42D mixed-integer)
- Gear Train (4D integer)
- Compression Spring (3D mixed-integer, constrained)

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
- [ ] Add CI/CD pipeline (GitHub Actions)
- [x] Generate API documentation setup (MkDocs configuration in mkdocs.yml)
- [ ] Add type stubs or improve ty/mypy compliance

### Developer Experience

- [ ] Pre-commit hooks for ruff format/check
- [x] Contribution guide (CONTRIBUTING.md)
- [x] Example notebooks (Jupyter) demonstrating common workflows (examples/quickstart_walkthrough.ipynb)
