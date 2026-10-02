# Cross-Library Gap Analysis: pyMOFL vs opfunu vs BBOB/COCO

Compares function coverage across three benchmark optimization libraries to identify gaps and inform pyMOFL's roadmap.

**Last updated**: 2026-02-24

---

## Executive Summary

| Library | Classical Functions | CEC Suites | BBOB Suites | Total Functions |
|---------|:-:|:-:|:-:|:-:|
| **pyMOFL** | 48 | 2005 (25) | — | ~73 |
| **opfunu** | 125 | 2005–2022 (196) | — | 321 |
| **BBOB/COCO** | — | — | noiseless (24) + extensions | 24–300+ |

### Key Gaps

1. **CEC 2006–2025**: pyMOFL has the base functions and infrastructure (compositions, hybrids, transforms) — needs suite JSON configs and data files for each year.
2. **BBOB noiseless suite (f1–f24)**: ~60% of base functions already exist in pyMOFL. Missing 8 base functions + BBOB-specific transformations (T_osz, T_asy, Gallagher peaks).
3. **Classical functions**: pyMOFL covers the most-used ~48; opfunu has 125 name-based functions. ~80 classical functions are absent from pyMOFL.

---

## 1. CEC Competition Suites

### 1.1 CEC Suite Coverage

pyMOFL currently implements CEC 2005 via JSON config + data files. The factory system (`FunctionFactory`, `CompositionBuilder`, `HybridFunction`, `WeightedComposition`) is generic and supports all CEC years — only suite-specific configuration and data files are needed.

| CEC Year | Functions | opfunu | pyMOFL | Gap |
|----------|:-:|:-:|:-:|---|
| 2005 | 25 | F1–F25 | F1–F25 | **None** — full parity |
| 2008 | 7 | F1–F7 | — | Config + data files needed. Large-scale (D=100–1000). |
| 2010 | 20 | F1–F20 | — | Config + data files needed. Large-scale with grouping (separable/partially-separable). |
| 2013 | 28 | F1–F28 | — | Config + data files needed. All 26 base primitives already implemented. |
| 2014 | 30 | F1–F30 | — | Config + data files needed. All base primitives already implemented. |
| 2015 | 15 | F1–F15 | — | Config + data files needed. All base primitives already implemented. |
| 2017 | 29 | F1–F29 | — | Config + data files needed. All base primitives already implemented. |
| 2019 | 10 | F1–F10 | — | Config + data files needed. 2 special functions missing (Chebyshev, Hilbert). |
| 2020 | 10 | F1–F10 | — | Config + data files needed. Subset of 2014/2017 bases. |
| 2021 | 10 | F1–F10 | — | Config + data files needed. Same structure as 2020. |
| 2022 | 12 | F1–F12 | — | Config + data files needed. All base primitives already implemented. |
| 2024–2025 | ~30 | — | — | No public library covers these yet. |
| **Total** | **196** | **196** | **25** | **171 functions = config + data files** |

### 1.2 What's Needed Per CEC Year

Each additional CEC year requires:

1. **Suite JSON config** (e.g., `cec2013_suite.json`) — defines function composition, transforms, and parameters
2. **Data files** — shift vectors and rotation matrices (binary or text, per dimension)
3. **No new Python code** for years using only existing base primitives (2013, 2014, 2015, 2017, 2020, 2021, 2022)

Exceptions requiring new base functions:

| Year | Missing Base Function | Notes |
|------|----------------------|-------|
| 2008 | `FastFractal DoubleDip` (F7) | Unique to 2008 |
| 2008 | `Schwefel 2.21` (F2) | max(abs(x_i)) — trivial to add |
| 2010 | (grouping infrastructure) | Variable partitioning into separable/non-separable groups |
| 2015 | 8 niching primitives | See [CEC gap analysis](cec_gap_analysis.md) |
| 2019 | `Chebyshev` (F1), `Hilbert` (F2) | Special non-standard problems |

### 1.3 CEC Base Primitive Status

All 26 core CEC primitives used across CEC 2013–2025 are **already implemented** in pyMOFL:

Sphere, Elliptic, Rosenbrock, Ackley, Rastrigin, Weierstrass, Griewank, Schwefel, Griewank-Rosenbrock, Schaffer F6 Expanded, Bent Cigar, Discus, Different Powers, Schaffers F7, Katsuura, Lunacek Bi-Rastrigin, HappyCat, HGBat, Zakharov, Levy, Dixon-Price, Sum Different Powers, Schwefel 1.2, Schwefel 2.6, Schwefel 2.13, Lennard-Jones.

See [CEC Base Function Gap Analysis](cec_gap_analysis.md) for detailed per-function status.

---

## 2. BBOB/COCO Functions

### 2.1 BBOB Noiseless Suite (f1–f24)

The BBOB suite uses a specific transformation framework distinct from CEC. Base functions overlap significantly with pyMOFL but BBOB applies its own T_osz, T_asy, and Lambda transformations.

| BBOB ID | Name | pyMOFL Base | Status | Notes |
|:-:|------|-------------|:------:|-------|
| f1 | Sphere | `SphereFunction` | **Have base** | |
| f2 | Ellipsoidal (separable) | `HighConditionedElliptic` | **Have base** | BBOB uses T_osz transformation |
| f3 | Rastrigin (separable) | `RastriginFunction` | **Have base** | BBOB uses T_osz + T_asy + Lambda |
| f4 | Bueche-Rastrigin | `RastriginFunction` | **Have base** | Needs asymmetric scaling variant |
| f5 | Linear Slope | — | **Missing** | Purely linear, optimum at boundary |
| f6 | Attractive Sector | — | **Missing** | Asymmetric sector function |
| f7 | Step Ellipsoidal | `HighConditionedElliptic` | **Partial** | Needs floor/rounding variant |
| f8 | Rosenbrock (original) | `RosenbrockFunction` | **Have base** | |
| f9 | Rosenbrock (rotated) | `RosenbrockFunction` | **Have base** | Rotation via existing transforms |
| f10 | Ellipsoidal (rotated) | `HighConditionedElliptic` | **Have base** | Non-separable via rotation |
| f11 | Discus | `DiscusFunction` | **Have base** | |
| f12 | Bent Cigar | `BentCigarFunction` | **Have base** | |
| f13 | Sharp Ridge | — | **Missing** | sqrt(sum) ridge structure |
| f14 | Sum of Different Powers | `SumDifferentPowersFunction` | **Have base** | BBOB exponent schedule differs slightly |
| f15 | Rastrigin (rotated) | `RastriginFunction` | **Have base** | Non-separable via rotation |
| f16 | Weierstrass | `WeierstrassFunction` | **Have base** | |
| f17 | Schaffer F7 (cond. 10) | `SchaffersF7Function` | **Have base** | |
| f18 | Schaffer F7 (cond. 1000) | `SchaffersF7Function` | **Have base** | Different conditioning parameter |
| f19 | Griewank-Rosenbrock F8F2 | `GriewankOfRosenbrock` | **Have base** | |
| f20 | Schwefel x*sin(x) | — | **Missing** | Deceptive; x*sin(sqrt(abs(x))) |
| f21 | Gallagher's 101 Peaks | — | **Missing** | Gaussian peaks composition |
| f22 | Gallagher's 21 Peaks | — | **Missing** | Fewer peaks, higher conditioning |
| f23 | Katsuura | `KatsuuraFunction` | **Have base** | |
| f24 | Lunacek Bi-Rastrigin | `LunacekBiRastriginFunction` | **Have base** | |

**Summary**: 16/24 base functions exist, 8 missing (f5, f6, f7-variant, f13, f14-variant, f20, f21, f22).

### 2.2 BBOB-Specific Transformations Missing from pyMOFL

| Transformation | Description | pyMOFL Equivalent | Gap |
|----------------|-------------|-------------------|-----|
| **T_osz** | Oscillation: smooth local irregularities around identity | — | **Missing** — element-wise nonlinear transform |
| **T_asy^beta** | Asymmetric: breaks symmetry for x>0 elements | — | **Missing** — element-wise power transform |
| **Lambda^alpha** | Diagonal conditioning matrix | `ScaleTransform` | **Covered** — equivalent to per-dimension scaling |
| **f_pen** | Boundary penalty: sum(max(0, abs(x_i)-5)^2) | — | **Missing** — output penalty transform |
| **Rotation Q, R** | Random orthogonal matrices | `RotateTransform` | **Covered** |
| **Shift x_opt** | Optimum shift | `ShiftTransform` | **Covered** |
| **Bias f_opt** | Output offset | `BiasTransform` | **Covered** |

New transforms needed: `OscillationTransform` (T_osz), `AsymmetricTransform` (T_asy), `BoundaryPenaltyTransform` (f_pen).

### 2.3 BBOB Extended Suites

| Suite | Functions | Description | Dependency |
|-------|:-:|-------------|------------|
| **bbob-noisy** | 30 | 3 noise types applied to base functions | Needs base bbob + `NoiseTransform` (already exists) |
| **bbob-biobj** | 55 | Bi-objective pairs of 10 base functions | Needs base bbob + multi-objective support (new) |
| **bbob-biobj-ext** | 92 | Extended bi-objective with all 24 bases | Needs base bbob + multi-objective support (new) |
| **bbob-largescale** | 24 | D=20–640, permuted block-diagonal rotations | Needs base bbob + block-diagonal rotation (new) |
| **bbob-mixint** | 24 | 80% variables discretized | Needs base bbob + `Quantized` (already exists) |
| **bbob-constrained** | 54 | 9 objectives x 6 constraint levels | Needs base bbob + constraint framework (new) |

---

## 3. Classical/Traditional Functions

### 3.1 Functions in Both pyMOFL and opfunu

These 25+ functions exist in both libraries (exact name mappings vary):

| pyMOFL Class | opfunu Class | Category |
|-------------|-------------|----------|
| `AckleyFunction` | `Ackley01` | Multimodal |
| `AlpineFunction1` | `Alpine01` | Multimodal |
| `AlpineFunction2` | `Alpine02` | Multimodal |
| `BraninFunction` | `Branin01` | Multimodal (2D) |
| `BraninFunction2` | `Branin02` | Multimodal (2D) |
| `DixonPriceFunction` | `DixonPrice` | Unimodal |
| `EasomFunction` | `Easom` | Multimodal (2D) |
| `GoldsteinPriceFunction` | `GoldsteinPrice` | Multimodal (2D) |
| `GriewankFunction` | `Griewank` | Multimodal |
| `HimmelblauFunction` | `Himmelblau` | Multimodal (2D) |
| `KatsuuraFunction` | `Katsuura` | Multimodal |
| `LennardJonesFunction` | `LennardJones` | Multimodal |
| `LevyFunction` | `Levy03`/`Levy05`/`Levy13` | Multimodal |
| `MatyasFunction` | `Matyas` | Unimodal (2D) |
| `McCormickFunction` | `McCormick` | Multimodal (2D) |
| `RosenbrockFunction` | (CEC only) | Multimodal |
| `SphereFunction` | (CEC only) | Unimodal |
| `ZakharovFunction` | `Zacharov` | Unimodal |

Note: opfunu does not have standalone name-based classes for Rosenbrock, Rastrigin, Sphere, Schwefel, Schaffer, Bent Cigar, Discus, etc. — these only appear as CEC components. pyMOFL provides these as first-class standalone functions.

### 3.2 Classical Functions in opfunu but NOT in pyMOFL (~80 functions)

Organized by priority (frequency of use in benchmarking literature):

#### High Priority — Commonly Used Benchmarks

| opfunu Class | Description | Dimension |
|-------------|-------------|:-:|
| `Beale` | Unimodal, narrow valley | 2D |
| `Bohachevsky1`/`2`/`3` | Multimodal with cosine terms | 2D |
| `Booth` | Unimodal, simple quadratic | 2D |
| `Bukin06` | Narrow ridge, non-convex | 2D |
| `CamelSixHump` | 6 local minima, 2 global | 2D |
| `CamelThreeHump` | 3 humps | 2D |
| `Colville` | Unimodal, 4D with valleys | 4D |
| `CrossInTray` | Multimodal, 4 global minima | 2D |
| `DropWave` | Multimodal, wave-like | 2D |
| `EggHolder` | Multimodal, deceptive | 2D |
| `Hartmann3` | Multimodal, Gaussian peaks | 3D |
| `Hartmann6` | Multimodal, Gaussian peaks | 6D |
| `HolderTable` | Multimodal, 4 global minima | 2D |
| `Langermann` | Multimodal, exponential-cosine | scalable |
| `Michalewicz` | Steep ridges and valleys | scalable |
| `Salomon` | Multimodal, concentric rings | scalable |
| `Styblinski-Tang` | Multimodal, 4th-degree polynomial | scalable |

#### Medium Priority — Used in Specific Benchmarking Contexts

| opfunu Class | Description | Dimension |
|-------------|-------------|:-:|
| `Adjiman` | 2D multimodal | 2D |
| `BartelsConn` | Absolute value landscape | 2D |
| `BiggsExp02`–`05` | Exponential fitting problems | 2–5D |
| `Bird` | Multimodal | 2D |
| `BoxBetts` | Quadratic sum | 3D |
| `Brent` | Smooth valley | 2D |
| `Brown` | Smooth unimodal | scalable |
| `ChenBird` / `ChenV` | Multimodal variants | 2D |
| `Chichinadze` | Multimodal with sine terms | 2D |
| `ChungReynolds` | Sum of squares, squared | scalable |
| `CosineMixture` | Smooth with cosine perturbations | scalable |
| `Csendes` | Product function | scalable |
| `Damavandi` | Deceptive near-flat regions | 2D |
| `Deb01` / `Deb03` | Multi-modal with many optima | scalable |
| `Deceptive` | Deceptive landscape | scalable |
| `DeflectedCorrugatedSpring` | Spring-like oscillation | scalable |
| `EggCrate` | Egg-crate surface | 2D |
| `Exponential` | Smooth exponential | scalable |
| `FreudensteinRoth` | 2D nonlinear system | 2D |
| `Giunta` | Multimodal with sine | 2D |
| `Gulf` | Exponential fitting | 3D |
| `Hansen` | Multimodal product of sines | 2D |
| `Hosaki` | Multimodal | 2D |
| `JennrichSampson` | Exponential curve fitting | 2D |
| `Keane` | Constrained multimodal | scalable |
| `Kowalik` | 4D parameter fitting | 4D |
| `Leon` | Smooth valley | 2D |
| `MieleCantrell` | 4D exponential-polynomial | 4D |
| `Mishra01`–`11` | Various multimodal variants | varies |
| `OddSquare` | Multimodal | scalable |
| `Parsopoulos` | Multimodal | 2D |
| `Qing` | Sum of (x_i^2 - i)^2 | scalable |
| `Quartic` | De Jong's 4th with noise | scalable |
| `Quintic` | 5th-degree polynomial | scalable |
| `Rana` | Complex multimodal | scalable |

#### Low Priority — Rare or Specialized

| opfunu Class | Description |
|-------------|-------------|
| `Cigar` | Variant of Bent Cigar |
| `Cola` | 17D marketing problem |
| `Corana` | Discontinuous |
| `CrossLegTable` / `CrownedCross` | Table-like surfaces |
| `Decanomial` | 10th-degree polynomial |
| `DeckkersAarts` | Multimodal |
| `DeVilliersGlasser01`/`02` | Exponential fitting |
| `Dolan` | 5D exponential |
| `Eckerle4` | Data fitting |
| `ElAttarVidyasagarDutta` | 2D nonlinear |
| `Exp2` | Exponential fitting |
| `Gear` | Integer optimization |
| `HelicalValley` | 3D helical |
| `Infinity` | Sinusoidal product |
| `Judge` | Data fitting |
| `Meyer` | Exponential model fitting |
| `MultiModal` | Generic multi-modal |
| `NeedleEye` | Discontinuous |
| `NewFunction01`/`02` | Named generics |
| `TestTubeHolder` | 2D multimodal |
| `Ursem01` | 2D multimodal |
| `VenterSobiezcczanskiSobieski` | 2D engineering |
| `Watson` | Polynomial fitting |
| `XinSheYang01` | Random-weighted multimodal |
| `YaoLiu04` | Max-absolute variant |
| `ZeroSum` | Constraint-like |
| `Zettl` / `Zirilli` / `Zimmerman` | 2D multimodal variants |

---

## 4. Transformation & Infrastructure Comparison

### 4.1 Transformation Coverage

| Capability | pyMOFL | opfunu | BBOB/COCO |
|-----------|:-:|:-:|:-:|
| Shift (translate optimum) | `ShiftTransform` | Built-in per class | x - x_opt |
| Rotation | `RotateTransform` | Built-in per class | Q, R matrices |
| Scale / conditioning | `ScaleTransform` | Built-in per class | Lambda^alpha |
| Bias (output offset) | `BiasTransform` | Built-in per class | f_opt |
| Noise | `NoiseTransform` | Built-in per class | Gaussian/Uniform/Cauchy |
| Non-continuous | `NonContinuousTransform` | Built-in per class | — |
| Quantization | `Quantized` | — | bbob-mixint discretization |
| Oscillation (T_osz) | — | — | Element-wise nonlinear |
| Asymmetric (T_asy) | — | — | Power-based symmetry breaking |
| Boundary penalty | — | — | f_pen |
| Composable pipeline | `ComposedFunction` | — | Fixed per function |
| Weighted composition | `WeightedComposition` | Hard-coded | — |
| Hybrid partitioning | `HybridFunction` | Hard-coded | — |
| Variable grouping | — | Hard-coded | Permuted block-diagonal |
| Multi-objective | — | — | bbob-biobj pairing |
| Constraints | — | — | Linear constraint generator |

### 4.2 Architecture Comparison

| Feature | pyMOFL | opfunu | BBOB/COCO |
|---------|--------|--------|-----------|
| **Design** | Modular composition | Monolithic per-function classes | C framework + Python wrapper |
| **Config** | JSON-driven factory | Hard-coded in each class | C source code |
| **Extensibility** | Add JSON config for new suites | Subclass per function | Fork/modify C code |
| **Data loading** | Dynamic (`DataLoader`) | Embedded in package | Compiled into binary |
| **Batch eval** | Per-class `evaluate_batch()` | `evaluate()` with array support | C-level vectorization |
| **Registry** | `@register` + auto-discovery | Import by class name | Suite enumeration |

---

## 5. Prioritized Roadmap

### Phase 1: BBOB Noiseless Suite (High Impact, Medium Effort)

The BBOB suite is widely used in academic benchmarking and 16/24 base functions already exist in pyMOFL.

| Step | Work | Effort |
|------|------|--------|
| 1a | Add missing base functions (Linear Slope, Attractive Sector, Sharp Ridge, Schwefel x*sin, Gallagher Peaks) | 5 new classes |
| 1b | Add BBOB-specific transforms (T_osz, T_asy, f_pen) | 3 new transform classes |
| 1c | Create BBOB suite JSON config + instance generation | Config + seeded RNG |
| 1d | Validate against COCO reference implementation | Test suite |

### Phase 2: High-Priority Classical Functions (~17 functions)

Add the most commonly used classical benchmarks not yet in pyMOFL: Beale, Booth, Bohachevsky, Bukin, Six-Hump Camel, Colville, Cross-in-Tray, Drop-Wave, Eggholder, Hartmann (3D & 6D), Holder Table, Langermann, Michalewicz, Salomon, Styblinski-Tang.

### Phase 3: BBOB Extended Suites (Low Priority)

These require architectural additions beyond single-objective optimization:
- **bbob-biobj**: Multi-objective framework
- **bbob-constrained**: Constraint handling framework
- **bbob-largescale**: Block-diagonal rotation generation
- **bbob-mixint**: Mostly covered by existing `Quantized` wrapper

### Phase 4: Remaining Classical Functions (~60 functions, Optional)

Lower priority. Add on-demand based on user requests.

### Phase 5: CEC 2013–2022 Suites (Blocked — Awaiting Golden Dataset)

> **Blocked on external project** generating validated golden datasets for each CEC year.

**All base primitives are implemented.** Each year needs:
- Suite JSON config file
- Shift vectors and rotation matrices (data files)
- Golden dataset for validation
- Validation tests against golden dataset

Estimated effort per year: **config + data files only** (no new Python code for most years).

| Year | Functions | New Code Needed | Notes |
|------|:-:|---|---|
| CEC 2013 | 28 | None — config + data only | |
| CEC 2014 | 30 | None — config + data only | |
| CEC 2017 | 29 | None — config + data only | |
| CEC 2015 | 15 | None — config + data only | |
| CEC 2022 | 12 | None — config + data only | |
| CEC 2020 | 10 | None — config + data only | |
| CEC 2021 | 10 | None — config + data only | |
| CEC 2019 | 10 | 2 special functions (Chebyshev, Hilbert) | |
| CEC 2008 | 7 | 2 base functions + large-scale support | |
| CEC 2010 | 20 | Variable grouping infrastructure | |

---

## Appendix A: Function Count Summary

| Category | pyMOFL | opfunu | BBOB/COCO | Notes |
|----------|:-:|:-:|:-:|---|
| Standalone classical | 48 | 125 | — | pyMOFL has CEC primitives as standalone; opfunu does not |
| CEC 2005 | 25 | 25 | — | Full parity |
| CEC 2008 | — | 7 | — | Large-scale |
| CEC 2010 | — | 20 | — | Large-scale with grouping |
| CEC 2013 | — | 28 | — | Config + data only |
| CEC 2014 | — | 30 | — | Config + data only |
| CEC 2015 | — | 15 | — | 8 niching functions need new code |
| CEC 2017 | — | 29 | — | Config + data only |
| CEC 2019 | — | 10 | — | 2 special functions need new code |
| CEC 2020 | — | 10 | — | Config + data only |
| CEC 2021 | — | 10 | — | Config + data only |
| CEC 2022 | — | 12 | — | Config + data only |
| BBOB noiseless | — | — | 24 | 16/24 bases exist in pyMOFL |
| BBOB noisy | — | — | 30 | Legacy suite |
| BBOB bi-objective | — | — | 55–92 | Needs multi-objective support |
| BBOB large-scale | — | — | 24 | Needs block-diagonal rotations |
| BBOB mixed-integer | — | — | 24 | `Quantized` mostly covers this |
| BBOB constrained | — | — | 54 | Needs constraint framework |
| Composition classes | 2 | (hard-coded) | — | `WeightedComposition`, `HybridFunction` |
| Transform types | 11 | (built-in) | 7+ | pyMOFL missing 3 BBOB-specific transforms |

## Appendix B: pyMOFL Architectural Advantage

pyMOFL's modular design means CEC suite expansion is primarily a **data problem, not a code problem**:

- **Compositions**: `WeightedComposition` handles all CEC composition functions (2005 F15–F25, 2013 CF1–CF8, 2014 CF1–CF8, 2017 CF1–CF10, etc.)
- **Hybrids**: `HybridFunction` handles all CEC hybrid functions (2014 HF1–HF6, 2017 HF1–HF10, etc.)
- **Transforms**: The composable pipeline (`ShiftTransform` → `RotateTransform` → `ScaleTransform` → base → `BiasTransform`) covers the standard CEC transform chain
- **Factory**: `FunctionFactory` reads JSON config and assembles any function — no year-specific code paths

In contrast, opfunu hard-codes each function as a separate class with embedded data, leading to significant code duplication across years.
