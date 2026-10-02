# Development Plan: pyMOFL Function Coverage Expansion

**Strategy:** Domain-First. All work adds pure benchmark functions, transforms, and suite configs — no UI/CLI changes until Phase 5 polish.

**Source:** Extracted from [Cross-Library Gap Analysis](cross_library_gap_analysis.md).

---

## Phase 0: Foundation & Standards (Coverage ≥95%)

No new modules or interfaces are needed — pyMOFL already has a stable `OptimizationFunction` base class, `VectorTransform`/`ScalarTransform` ABCs, `ComposedFunction`, `@register` auto-discovery, and the factory system. This phase verifies the starting state and establishes test infrastructure for the new work.

### Task 0.1: Verify baseline test suite health

**Type:** Test
**Description:** Run the full test suite and confirm the existing pass/fail state. Document any pre-existing failures so they are not confused with regressions.
**Acceptance Criteria:**
- [x] `uv run pytest` runs to completion.
- [x] Pre-existing failures (CEC F23–F25 non-optimum validation) are documented.
- [x] Coverage baseline is recorded.

### Task 0.2: Create shared test utilities for new benchmark functions

**Type:** Implement
**Description:** Add a parametrized test helper in `tests/utils/` that validates any `OptimizationFunction` subclass against a standard contract: `evaluate()` returns a float, `evaluate_batch()` returns correct shape, `get_global_minimum()` returns correct value at the optimum, bounds are set, and input validation rejects wrong dimensions.
**Acceptance Criteria:**
- [x] Helper function/class exists in `tests/utils/benchmark_validation.py`.
- [x] Exercised by at least one existing benchmark function test as a smoke test.
- [x] Coverage ≥95% maintained.
**Dependencies:** Task 0.1

### Task 0.3: Create shared test utilities for new transform classes

**Type:** Implement
**Description:** Add a parametrized test helper in `tests/utils/` that validates any `VectorTransform` or `ScalarTransform` subclass against its ABC contract: `__call__` returns correct type/shape, `transform_batch` is consistent with element-wise calls, identity-like behavior for neutral parameters.
**Acceptance Criteria:**
- [x] Helper function/class exists in `tests/utils/transform_validation.py`.
- [x] Exercised by at least one existing transform test as a smoke test.
- [x] Coverage ≥95% maintained.
**Dependencies:** Task 0.1

> **Stage for human review**

---

## Phase 1: BBOB Noiseless Suite — Base Functions (Coverage ≥95%)

Add the 5 missing BBOB base functions that have no pyMOFL equivalent. Each follows the established pattern: inherit `OptimizationFunction`, implement `evaluate()` + `evaluate_batch()` + `get_global_minimum()`, register with `@register`.

References: BBOB function definitions from Hansen et al. (2009, updated 2019), INRIA RR-6829.

### Task 1.1: Test LinearSlopeFunction

**Type:** Test
**Description:** Write tests for BBOB f5 (Linear Slope). The function is `f(x) = sum(5|s_i| - s_i * z_i)` where `s_i = sign(x_opt_i) * 10^((i-1)/(D-1))`. Optimum is at the domain boundary. Tests must cover: evaluate at known points, batch evaluation, global minimum, scalability across dimensions (2, 10, 30), boundary optimum behavior.
**Acceptance Criteria:**
- [x] Tests in `tests/functions/benchmark/test_linear_slope.py`.
- [x] Tests cover evaluate, evaluate_batch, get_global_minimum, dimension scaling.
- [x] Tests fail (red) — class does not exist yet.
**Dependencies:** Task 0.2

### Task 1.2: Implement LinearSlopeFunction

**Type:** Implement
**Description:** Create `src/pyMOFL/functions/benchmark/linear_slope.py` implementing BBOB f5. Register as `"LinearSlope"` and `"linear_slope"`. Default bounds: `[-5, 5]^D`. Optimum at boundary `x_opt = 5 * sign_vector`. Uses vectorized NumPy.
**Acceptance Criteria:**
- [x] All Task 1.1 tests pass (green).
- [x] Follows existing class pattern (see `SphereFunction`).
- [x] Coverage ≥95% maintained.
**Dependencies:** Task 1.1

---

### Task 1.3: Test AttractiveSectorFunction

**Type:** Test
**Description:** Write tests for BBOB f6 (Attractive Sector). The function applies asymmetric scaling (`s_i = 100` if `z_i * x_opt_i > 0`, else `1`) after rotation/conditioning, then `T_osz(sum)^0.9`. Tests must cover: evaluate at known points, asymmetric sector behavior, batch evaluation, dimension scaling.
**Acceptance Criteria:**
- [x] Tests in `tests/functions/benchmark/test_attractive_sector.py`.
- [x] Tests cover the asymmetric weighting behavior.
- [x] Tests fail (red).
**Dependencies:** Task 0.2

### Task 1.4: Implement AttractiveSectorFunction

**Type:** Implement
**Description:** Create `src/pyMOFL/functions/benchmark/attractive_sector.py` implementing BBOB f6. Register as `"AttractiveSector"` and `"attractive_sector"`. The asymmetric sector weighting is internal to the function (not a separate transform). Default bounds: `[-5, 5]^D`.
**Acceptance Criteria:**
- [x] All Task 1.3 tests pass (green).
- [x] Coverage ≥95% maintained.
**Dependencies:** Task 1.3

---

### Task 1.5: Test SharpRidgeFunction

**Type:** Test
**Description:** Write tests for BBOB f13 (Sharp Ridge). The function is `f(x) = z_1^2 + 100 * sqrt(sum(z_i^2, i=2..D))`. The ridge along `z_2..z_D = 0` is non-differentiable. Tests must cover: evaluate at origin, along the ridge, off the ridge, batch evaluation, dimension scaling.
**Acceptance Criteria:**
- [x] Tests in `tests/functions/benchmark/test_sharp_ridge.py`.
- [x] Tests verify the non-differentiable ridge behavior (gradient discontinuity at ridge).
  - `test_non_differentiable_at_ridge`: V-shaped slope at ridge breakpoint (slope = 100, not vanishing).
  - `test_ridge_linear_vs_quadratic_growth`: O(h) growth off-ridge, not O(h²).
- [x] Tests fail (red).
**Dependencies:** Task 0.2

### Task 1.6: Implement SharpRidgeFunction

**Type:** Implement
**Description:** Create `src/pyMOFL/functions/benchmark/sharp_ridge.py` implementing BBOB f13. Register as `"SharpRidge"` and `"sharp_ridge"`. Default bounds: `[-5, 5]^D`.
**Acceptance Criteria:**
- [x] All Task 1.5 tests pass (green).
- [x] Coverage ≥95% maintained.
**Dependencies:** Task 1.5

---

### Task 1.7: Test SchwefelSinFunction

**Type:** Test
**Description:** Write tests for BBOB f20 base (Schwefel x*sin(x)). The **base function** implements the classical Schwefel: `f(x) = 418.9829*D - sum(x_i * sin(sqrt(|x_i|)))`. The full BBOB f20 formula `-(1/(100D)) * sum(z_i * sin(sqrt(|z_i|))) + 4.189828... + penalty` is equivalent to `classical(z)/(100D)` with added penalty; the BBOB normalization (`1/(100D)` scaling), bias adjustment, and boundary penalty are applied externally via `ComposedFunction` transforms in Phase 3 (consistent with all other BBOB/CEC functions). Tests must cover: evaluate at known points, global minimum value (~0), deceptive structure (second-best optimum far from global), batch evaluation.
**Acceptance Criteria:**
- [x] Tests in `tests/functions/benchmark/test_schwefel_sin.py`.
- [x] Tests fail (red).
**Dependencies:** Task 0.2

### Task 1.8: Implement SchwefelSinFunction

**Type:** Implement
**Description:** Create `src/pyMOFL/functions/benchmark/schwefel_sin.py` implementing the classical Schwefel base for BBOB f20. Register as `"SchwefelSin"` and `"schwefel_sin"`. This is distinct from the existing `Schwefel_1_2`, `Schwefel_2_6`, and `Schwefel_2_13` variants. Default bounds: `[-500, 500]^D` (traditional Schwefel domain). **Note:** BBOB-specific normalization (`1/(100D)` scaling) and penalty are Phase 3 transform-layer work, not part of the base function.
**Acceptance Criteria:**
- [x] All Task 1.7 tests pass (green).
- [x] Coverage ≥95% maintained.
**Dependencies:** Task 1.7

---

### Task 1.9: Test GallagherPeaksFunction

**Type:** Test
**Description:** Write tests for BBOB f21/f22 (Gallagher's Gaussian Peaks). The function creates N Gaussian peaks with random positions, weights, and per-peak conditioning matrices. Parametrized by `n_peaks` (101 for f21, 21 for f22). Tests must cover: construction with 101 and 21 peaks, evaluate at the global optimum peak, evaluate away from peaks, seeded reproducibility, batch evaluation, dimension scaling.
**Acceptance Criteria:**
- [x] Tests in `tests/functions/benchmark/test_gallagher_peaks.py`.
- [x] Tests cover both n_peaks=101 and n_peaks=21 configurations.
- [x] Tests verify seeded reproducibility.
- [x] Tests fail (red).
**Dependencies:** Task 0.2

### Task 1.10: Implement GallagherPeaksFunction

**Type:** Implement
**Description:** Create `src/pyMOFL/functions/benchmark/gallagher_peaks.py` implementing BBOB f21/f22. Constructor takes `n_peaks` (default 101), `seed` for reproducible peak generation, and standard dimension/bounds. Register as `"GallagherPeaks"` and `"gallagher_peaks"`. Peak positions drawn from `[-4, 4]^D`, conditioning matrices generated per BBOB spec. Default bounds: `[-5, 5]^D`.
**Acceptance Criteria:**
- [x] All Task 1.9 tests pass (green).
- [x] `GallagherPeaksFunction(dimension=D, n_peaks=101)` produces BBOB f21.
- [x] `GallagherPeaksFunction(dimension=D, n_peaks=21)` produces BBOB f22.
- [x] Coverage ≥95% maintained.
**Dependencies:** Task 1.9

> **Stage for human review**

---

## Phase 2: BBOB Noiseless Suite — Transforms (Coverage ≥95%)

Add the 3 BBOB-specific transformations missing from pyMOFL. Each follows the established `VectorTransform` or `ScalarTransform` pattern.

### Task 2.1: Test OscillationTransform (T_osz)

**Type:** Test
**Description:** Write tests for the BBOB oscillation transformation T_osz. Element-wise: `T_osz(x_i) = sign(x_i) * exp(x_hat + 0.049*(sin(c1*x_hat) + sin(c2*x_hat)))` where `x_hat = log(|x_i|)` and c1/c2 depend on sign. Tests must cover: identity-like behavior near zero, positive values, negative values, batch transform, known BBOB reference values.
**Acceptance Criteria:**
- [x] Tests in `tests/functions/transformations/test_oscillation.py`.
- [x] Tests verify element-wise correctness against hand-computed values.
- [x] Tests verify `transform_batch` consistency.
- [x] Tests fail (red).
**Dependencies:** Task 0.3

### Task 2.2: Implement OscillationTransform

**Type:** Implement
**Description:** Create `src/pyMOFL/functions/transformations/oscillation.py` implementing `OscillationTransform(VectorTransform)`. Vectorized NumPy implementation. Export from `__init__.py`.
**Acceptance Criteria:**
- [x] All Task 2.1 tests pass (green).
- [x] Vectorized `__call__` and `transform_batch` (no Python loops).
- [x] Coverage ≥95% maintained.
**Dependencies:** Task 2.1

---

### Task 2.3: Test AsymmetricTransform (T_asy)

**Type:** Test
**Description:** Write tests for the BBOB asymmetric transformation T_asy^beta. Element-wise: `T_asy(x_i) = x_i^(1 + beta*(i-1)/(D-1)*sqrt(x_i))` for `x_i > 0`, identity otherwise. Parametrized by `beta`. Tests must cover: identity for x<=0, power scaling for x>0, beta=0 yields identity, beta=0.5 (common BBOB value), dimension-dependent exponents, batch transform.
**Acceptance Criteria:**
- [x] Tests in `tests/functions/transformations/test_asymmetric.py`.
- [x] Tests verify dimension-dependent exponent scaling.
- [x] Tests fail (red).
**Dependencies:** Task 0.3

### Task 2.4: Implement AsymmetricTransform

**Type:** Implement
**Description:** Create `src/pyMOFL/functions/transformations/asymmetric.py` implementing `AsymmetricTransform(VectorTransform)`. Constructor takes `beta: float` and `dimension: int`. Vectorized NumPy implementation. Export from `__init__.py`.
**Acceptance Criteria:**
- [x] All Task 2.3 tests pass (green).
- [x] Vectorized implementation.
- [x] Coverage ≥95% maintained.
**Dependencies:** Task 2.3

---

### Task 2.5: Test BoundaryPenaltyTransform (f_pen)

**Type:** Test
**Description:** Write tests for the BBOB boundary penalty. Implemented as a `PenaltyTransform` (new ABC: vector→scalar additive penalty). The penalty computes `sum(max(0, |x_i| - bound)^2)` on the raw input vector and is added to the function output by `ComposedFunction`. Design decision resolved: `PenaltyTransform` ABC in `base.py`, `BoundaryPenaltyTransform` inherits from it, `ComposedFunction` has `penalty_transforms` parameter. Tests cover: zero penalty inside `[-5, 5]`, quadratic penalty outside, symmetric behavior, batch, ABC conformance, ComposedFunction integration.
**Acceptance Criteria:**
- [x] Tests in `tests/functions/transformations/test_boundary_penalty.py`.
- [x] Tests verify zero penalty for in-bounds inputs.
- [x] Tests verify correct quadratic growth for out-of-bounds inputs.
- [x] Tests fail (red).
**Dependencies:** Task 0.3

### Task 2.6: Implement BoundaryPenaltyTransform

**Type:** Implement
**Description:** Created `PenaltyTransform` ABC in `base.py`, `BoundaryPenaltyTransform(PenaltyTransform)` in `boundary_penalty.py`. Extended `ComposedFunction` with `penalty_transforms` parameter (penalties computed on raw input, added to output). Added routing in `TransformBuilder` for `boundary_penalty`/`f_pen`. Updated `build_many()` to 3-tuple. Export from `__init__.py`.
**Acceptance Criteria:**
- [x] All Task 2.5 tests pass (green).
- [x] Vectorized implementation.
- [x] Coverage ≥95% maintained.
**Dependencies:** Task 2.5

> **Stage for human review**

---

## Phase 3: BBOB Noiseless Suite — Suite Config & Validation (Coverage ≥90%)

Wire the 24 BBOB noiseless functions into a suite config and validate against COCO reference values.

### Task 3.1: Test BBOB suite JSON config parsing

**Type:** Test
**Description:** Write tests verifying that a `bbob_suite.json` config can be parsed by `ConfigParser` and assembled by `FunctionFactory` into 24 `ComposedFunction` instances (one per BBOB function ID). Tests cover: all 24 functions instantiate without error, correct base function types, correct transform chains per BBOB spec, dimension parametrization (2, 3, 5, 10, 20, 40).
**Acceptance Criteria:**
- [x] Tests in `tests/functions/bbob/test_bbob_suite.py`. *(Phase 3 Corrections — enriched bbob_suite.json + test file created)*
- [x] Tests verify all 24 functions instantiate for dimensions 2, 10, 40.
- [x] Tests fail (red) → pass (green) after suite JSON enrichment.
**Dependencies:** Phase 2

### Task 3.2: Create BBOB suite JSON config

**Type:** Implement
**Description:** Create `src/pyMOFL/constants/bbob/bbob_suite.json` defining all 24 BBOB noiseless functions. Each function entry specifies: base function type, input transforms (T_osz, T_asy, Lambda, rotation, shift), output transforms (bias, boundary penalty), and per-function parameters. Follow the nested config convention established by `cec2005_suite.json`.
**Acceptance Criteria:**
- [x] All Task 3.1 tests pass (green). *(Phase 3 Corrections — enriched with nested construction templates, string IDs, dimensions)*
- [x] Config follows existing JSON convention (nesting = composition order). *(Templates use null for instance-specific params)*
- [x] All 24 functions are defined.
**Dependencies:** Task 3.1

### Task 3.3: Implement BBOB instance generation

**Type:** Implement
**Description:** BBOB functions are parametrized by "instance" (seeded random shifts, rotations, biases). Add an instance parameter to the BBOB config/factory that generates `x_opt`, `f_opt`, and rotation matrices from a seed. This may extend `DataLoader` or add a `BBOBInstanceGenerator` utility.
**Acceptance Criteria:**
- [x] Instance 1 through 15 produce reproducible, distinct function instances. *(Phase 3 Corrections — updated to COCO-compatible seeding)*
- [x] `x_opt` drawn from `[-4, 4]^D`, `f_opt` from Cauchy distribution per BBOB spec.
- [x] Rotation matrices are orthogonal (verified by test).
- [x] Coverage ≥90% maintained.
**Dependencies:** Task 3.2

### Task 3.4: Validate BBOB suite against COCO reference values

**Type:** Test
**Description:** Obtain reference function values from the COCO framework (via `cocoex` Python package or pre-computed tables) for a selection of BBOB functions, dimensions, and instances. Compare pyMOFL's output against these reference values within tolerance.
**Acceptance Criteria:**
- [x] At least f1, f8, f15, f20, f24 validated against COCO reference values. *(Phase 3 Corrections — strict numeric parity tests with cocoex)*
- [x] Tolerance: `|pyMOFL - COCO| < 1e-8` for noiseless functions.
- [x] Validated for dimensions 2, 10, 40.
- [x] Coverage ≥90% maintained.
**Dependencies:** Task 3.3

> **Stage for human review**

---

## Phase 4: High-Priority Classical Functions (Coverage ≥90%)

Add ~17 commonly used classical benchmark functions missing from pyMOFL. These are straightforward `OptimizationFunction` subclasses with well-known formulas.

Grouped into sub-phases by dimensionality to allow incremental delivery.

### Phase 4.1: Scalable Classical Functions

Functions that work in arbitrary dimension D.

#### Task 4.1.1: Test scalable classical functions

**Type:** Test
**Description:** Write parametrized tests for the following scalable functions: `StyblinskiTangFunction`, `SalomonFunction`, `MichalewiczFunction`, `LangermannFunction`, `BrownFunction`, `ChungReynoldsFunction`, `QingFunction`, `QuarticFunction` (De Jong's 4th). Use the shared benchmark validation helper from Task 0.2 plus function-specific tests for known optima and characteristic behavior.
**Acceptance Criteria:**
- [x] Tests in `tests/functions/benchmark/test_classical_scalable.py`.
- [x] Each function tested for: evaluate correctness, batch evaluation, global minimum, dimensions 2/10/30.
- [x] Tests fail (red).
**Dependencies:** Task 0.2

#### Task 4.1.2: Implement scalable classical functions

**Type:** Implement
**Description:** Create implementation files for the 8 scalable functions. Each inherits `OptimizationFunction`, registered with `@register`. Use vectorized NumPy. One file per function or group logically related functions.

| Function | File | Registry Alias |
|----------|------|---------------|
| Styblinski-Tang | `styblinski_tang.py` | `"styblinski_tang"` |
| Salomon | `salomon.py` | `"salomon"` |
| Michalewicz | `michalewicz.py` | `"michalewicz"` |
| Langermann | `langermann.py` | `"langermann"` |
| Brown | `brown.py` | `"brown"` |
| Chung-Reynolds | `chung_reynolds.py` | `"chung_reynolds"` |
| Qing | `qing.py` | `"qing"` |
| Quartic | `quartic.py` | `"quartic"` |

**Acceptance Criteria:**
- [x] All Task 4.1.1 tests pass (green).
- [x] All functions registered and discoverable via registry.
- [x] Coverage ≥90% maintained.
**Dependencies:** Task 4.1.1

> **Stage for human review**

### Phase 4.2: Fixed-Dimension Classical Functions (2D–6D)

Functions with fixed dimensionality, commonly used in benchmarking literature.

#### Task 4.2.1: Test fixed-dimension classical functions

**Type:** Test
**Description:** Write tests for the following fixed-dimension functions: `BealeFunction` (2D), `BoothFunction` (2D), `BohachevskyFunction` (2D, 3 variants), `BukinFunction6` (2D), `SixHumpCamelFunction` (2D), `ThreeHumpCamelFunction` (2D), `CrossInTrayFunction` (2D), `DropWaveFunction` (2D), `EggholderFunction` (2D), `HolderTableFunction` (2D), `HartmannFunction` (3D and 6D variants), `ColvilleFunction` (4D).
**Acceptance Criteria:**
- [x] Tests in `tests/functions/benchmark/test_classical_fixed.py`.
- [x] Each function tested for: evaluate at known optima, evaluate at characteristic points, batch evaluation, correct dimension enforcement.
- [x] Tests fail (red).
**Dependencies:** Task 0.2

#### Task 4.2.2: Implement fixed-dimension classical functions

**Type:** Implement
**Description:** Create implementation files for ~13 fixed-dimension functions. Each enforces its fixed dimension in `__init__`.

| Function | File | Dim | Registry Alias |
|----------|------|:-:|---------------|
| Beale | `beale.py` | 2 | `"beale"` |
| Booth | `booth.py` | 2 | `"booth"` |
| Bohachevsky (3 variants) | `bohachevsky.py` | 2 | `"bohachevsky1"`, `"bohachevsky2"`, `"bohachevsky3"` |
| Bukin N.6 | `bukin.py` | 2 | `"bukin6"` |
| Six-Hump Camel | `camel.py` | 2 | `"six_hump_camel"` |
| Three-Hump Camel | `camel.py` | 2 | `"three_hump_camel"` |
| Cross-in-Tray | `cross_in_tray.py` | 2 | `"cross_in_tray"` |
| Drop-Wave | `drop_wave.py` | 2 | `"drop_wave"` |
| Eggholder | `eggholder.py` | 2 | `"eggholder"` |
| Holder Table | `holder_table.py` | 2 | `"holder_table"` |
| Hartmann 3 | `hartmann.py` | 3 | `"hartmann3"` |
| Hartmann 6 | `hartmann.py` | 6 | `"hartmann6"` |
| Colville | `colville.py` | 4 | `"colville"` |

**Acceptance Criteria:**
- [x] All Task 4.2.1 tests pass (green).
- [x] All functions registered and discoverable.
- [x] Dimension mismatch raises `ValueError`.
- [x] Coverage ≥90% maintained.
**Dependencies:** Task 4.2.1

> **Stage for human review**

---

## Phase 5: BBOB Extended Suites (Coverage ≥90%)

Each extended suite builds on the base BBOB suite from Phase 3. Human selected all 4 non-biobj extensions; biobj deferred.

### Task 5.1: **Human Decision** — Scope BBOB extensions

**Type:** Document
**Description:** Before implementing, decide which BBOB extensions to support. Selected: bbob-noisy, bbob-mixint, bbob-largescale, bbob-constrained. Deferred: bbob-biobj (requires multi-objective framework).
**Acceptance Criteria:**
- [x] Human has selected which extensions to implement.
- [x] Scope documented for subsequent tasks.
**Dependencies:** Phase 3

### Task 5.2: Implement bbob-noisy (30 functions, f101–f130)

**Type:** Implement
**Description:** 10 base BBOB functions × 3 COCO noise models. Created 3 new `ScalarTransform` subclasses (`GaussianNoiseTransform`, `UniformNoiseTransform`, `CauchyNoiseTransform`) using `np.random.default_rng`. `BBOBNoisySuiteFactory` wraps `BBOBSuiteFactory` and appends noise output transforms.
**Acceptance Criteria:**
- [x] 3 noise transform classes implemented with TDD (33 tests).
- [x] `BBOBNoisySuiteFactory` creates all 30 functions (15 tests).
- [x] Transforms registered in `TransformBuilder` (`gaussian_noise`, `uniform_noise`, `cauchy_noise`).
- [x] Coverage ≥90% maintained.
**Dependencies:** Task 5.1

### Task 5.3: Implement bbob-mixint (24 functions)

**Type:** Implement
**Description:** 24 BBOB functions with 80% discretized variables. Created `DiscretizeTransform(VectorTransform)` with per-variable arity mapping [2, 4, 8, 16, continuous]. `BBOBMixintSuiteFactory` prepends discretize transform to input chain. Dimension must be divisible by 5.
**Acceptance Criteria:**
- [x] `DiscretizeTransform` implemented with TDD (12 tests).
- [x] `BBOBMixintSuiteFactory` creates all 24 functions at D=10,20 (9 tests).
- [x] Transform registered in `TransformBuilder` (`discretize`).
- [x] Coverage ≥90% maintained.
**Dependencies:** Task 5.1

### Task 5.4: Implement bbob-largescale (24 functions, D=20–640)

**Type:** Implement
**Description:** 24 BBOB functions with block-diagonal rotations + truncated swap permutations replacing full D×D rotations for D>40. Created `PermutationTransform(VectorTransform)` and `BlockDiagonalRotateTransform(VectorTransform)`. Added `generate_permutation()` and `generate_block_diagonal_rotation()` to `BBOBInstanceGenerator`. `BBOBLargeScaleSuiteFactory` replaces `RotateTransform` with P1·B·P2 chain.
**Acceptance Criteria:**
- [x] `PermutationTransform` and `BlockDiagonalRotateTransform` implemented with TDD (16 tests).
- [x] Instance generation methods added with TDD (9 tests).
- [x] `BBOBLargeScaleSuiteFactory` creates all 24 functions; D=40 matches standard BBOB (8 tests).
- [x] Coverage ≥90% maintained.
**Dependencies:** Task 5.1

### Task 5.5: Implement bbob-constrained (54 = 9 objectives × 6 constraint configs)

**Type:** Implement
**Description:** 9 BBOB objectives × 6 linear constraint configurations. Created `LinearConstraint` (core class), `ConstrainedFunction(OptimizationFunction)` with `evaluate_constraints()`, `violations()`, `is_feasible()`. `BBOBConstraintGenerator` generates constraints with active binding at optimum and inactive feasible at optimum. `BBOBConstrainedSuiteFactory` produces all 54 functions.
**Acceptance Criteria:**
- [x] `LinearConstraint` and `ConstrainedFunction` implemented with TDD (19 tests).
- [x] `BBOBConstraintGenerator` implemented with TDD (13 tests).
- [x] `BBOBConstrainedSuiteFactory` creates all 54 functions (10 tests).
- [x] Coverage ≥90% maintained.
**Dependencies:** Task 5.1

> **Phase 5 complete.** 132 new functions, 144 new tests. 1697 passed, 1 skipped, 27 xfailed, lint clean.

---

## Phase 6: Remaining Classical Functions (Coverage ≥90%, Optional)

~60 additional classical functions from the opfunu catalog. Add on-demand based on user requests.

### Task 6.1: **Human Decision** — Prioritize remaining classical functions

**Type:** Document
**Description:** Review the medium and low priority lists in the [gap analysis](cross_library_gap_analysis.md) and select which functions to add next. Selected next batch: `adjiman`, `box_betts`, `deb01`, `deb03`, `exponential`, `keane`, `kowalik`, `miele_cantrell`, `parsopoulos`, `rana`.
**Acceptance Criteria:**
- [x] Prioritized list of next-batch functions agreed upon.
**Dependencies:** Phase 4

### Task 6.2: Implement selected classical functions

**Type:** Implement
**Description:** Follow the same TDD pattern as Phase 4: test task → implement task → register → validate. Batch into scalable vs fixed-dimension groups. To reduce manual registry drift, `FunctionRegistry` auto-discovers benchmark classes exported by `pyMOFL.functions.benchmark` and derives stable snake_case aliases.
**Acceptance Criteria:**
- [x] Selected batch functions are available through `FunctionRegistry` aliases.
- [x] `create_base_function()` instantiates selected batch types via factory tests.
- [x] Phase 6 registry integration tested and passing.
**Dependencies:** Task 6.1

> **Stage for human review**

---

## Phase 7: CEC 2013–2022 Suites (Partially Unblocked)

> **Status:** CEC 2014 suite config + data are integrated. Full multi-year rollout remains blocked on external golden datasets and year-specific data integration.

All 26 core CEC base primitives are already implemented. Each year requires only:
- Suite JSON config file (like `cec2005_suite.json`)
- Shift vector and rotation matrix data files
- Golden dataset for validation
- Validation tests against golden dataset

No new Python code needed for CEC 2013, 2014, 2015, 2017, 2020, 2021, 2022.

Exceptions requiring new base functions:
- CEC 2008: `FastFractal DoubleDip`, `Schwefel 2.21`
- CEC 2010: Variable grouping infrastructure
- CEC 2015: 8 niching primitives (see [CEC gap analysis](cec_gap_analysis.md))
- CEC 2019: `Chebyshev`, `Hilbert`

| Year | Functions | New Code Needed |
|------|:-:|---|
| CEC 2013 | 28 | None — config + data only |
| CEC 2014 | 30 | None — config + data only |
| CEC 2017 | 29 | None — config + data only |
| CEC 2015 | 15 | None — config + data only |
| CEC 2022 | 12 | None — config + data only |
| CEC 2020 | 10 | None — config + data only |
| CEC 2021 | 10 | None — config + data only |
| CEC 2019 | 10 | 2 special functions |
| CEC 2008 | 7 | 2 base functions + large-scale |
| CEC 2010 | 20 | Variable grouping infrastructure |

### Task 7.1: Unblock — Integrate golden datasets

**Type:** Integrate
**Description:** When the external golden dataset project delivers validated data for a CEC year, integrate the data files and create the suite JSON config. Follow the `cec2005_suite.json` pattern.
**Acceptance Criteria:**
- [x] CEC 2014 suite JSON config parses correctly.
- [x] All CEC 2014 functions instantiate via `FunctionFactory`.
- [ ] Validation tests pass against golden dataset (environment-dependent; requires `CEC_BENCHMARKS_PATH` datasets).
- [ ] Coverage ≥90% maintained.
**Dependencies:** External golden dataset project.

> **Stage for human review**

---

## Phase 8: Polish & Documentation (Coverage ≥90%)

### Task 8.1: Update README and function catalog

**Type:** Document
**Description:** Update `README.md` to reflect the expanded function catalog. Add a function index or catalog page listing all available functions by category with registry aliases.
**Acceptance Criteria:**
- [ ] README accurately lists supported suites and function counts.
- [ ] Function catalog is navigable and complete.

### Task 8.2: Update CLI suite listing

**Type:** Implement
**Description:** Ensure the `pymofl` CLI's suite listing commands reflect the new BBOB suite and any new classical functions.
**Acceptance Criteria:**
- [ ] `pymofl` CLI lists BBOB suite functions.
- [ ] No regressions in existing CLI functionality.

### Task 8.3: Final refactoring pass

**Type:** Refactor
**Description:** Review all new code for DRY violations, naming consistency, and alignment with existing patterns. Remove any dead code introduced during development.
**Acceptance Criteria:**
- [ ] `uv run ruff check src/ tests/` passes.
- [ ] `uv run ruff format src/ tests/` reports no changes.
- [ ] No duplicate logic across new functions.
- [ ] Coverage ≥90% maintained.

> **Stage for human review**

---

## Summary

| Phase | Description | New Functions | Blocked? |
|:-----:|-------------|:------------:|:--------:|
| 0 | Foundation & test utilities | 0 | No |
| 1 | BBOB base functions | 5 classes | No |
| 2 | BBOB transforms | 3 transforms | No |
| 3 | BBOB suite config & validation | 24 (composed) | No |
| 4 | High-priority classical functions | ~17 classes | No |
| 5 | BBOB extended suites (noisy, mixint, largescale, constrained) | 132 (4 factories) | No |
| 6 | Remaining classical functions | TBD | Human decision |
| 7 | CEC 2013–2022 suites | ~171 (composed) | **Partially (CEC 2014 integrated)** |
| 8 | Polish & documentation | 0 | No |
