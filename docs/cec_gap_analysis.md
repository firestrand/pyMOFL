# CEC Base Function Gap Analysis

Compares CEC benchmark primitives (extracted from C/C++ evaluator source across CEC 2013–2025) against pyMOFL's current inventory. Identifies missing functions and prioritizes by reuse frequency.

**Last updated**: 2026-02-22

---

## Summary

| Category | Implemented | Missing | Total |
|----------|:-----------:|:-------:|:-----:|
| Core CEC primitives | 26 | 0 | 26 |
| CEC 2015 niching | 0 | 8 | 8 |
| CEC 2019 special | 1 | 2 | 3 |
| Meta-functions (composition/hybrid) | 2 | 0 | 2 |
| Transformation pipeline | 10 | 0 | 10 |
| **Total** | **39** | **10** | **49** |

---

## Core CEC Primitives

Status key: **Done** = fully implemented, **Missing** = not implemented, **Partial** = base exists but expanded/variant form needed.

### Implemented

| # | CEC Name | pyMOFL Class | Registry Alias | CEC Years Used |
|---|----------|-------------|----------------|----------------|
| 1 | Sphere | `SphereFunction` | `sphere` | 2005, 2013–2020, 2024–2025 |
| 2 | High Conditioned Elliptic | `HighConditionedElliptic` | `high_conditioned_elliptic` | 2005, 2013–2022, 2024–2025 |
| 3 | Rosenbrock | `RosenbrockFunction` | `rosenbrock` | 2005, 2013–2022, 2024–2025 |
| 4 | Ackley | `AckleyFunction` | `ackley` | 2005, 2013–2022, 2024–2025 |
| 5 | Rastrigin | `RastriginFunction` | `rastrigin` | 2005, 2013–2022, 2024–2025 |
| 6 | Weierstrass | `WeierstrassFunction` | `weierstrass` | 2005, 2013–2020, 2024–2025 |
| 7 | Griewank | `GriewankFunction` | `griewank` | 2005, 2013–2022, 2024–2025 |
| 8 | Schwefel | `SchwefelFunction` | (none) | 2005, 2013–2022, 2024–2025 |
| 9 | Griewank-Rosenbrock (F8F2) | `GriewankOfRosenbrock` | `griewank_of_rosenbrock` | 2005, 2013–2022, 2024–2025 |
| 10 | Expanded Scaffer F6 | `Schaffer_F6_Expanded` | `schaffer_f6_expanded` | 2005, 2013–2022, 2024–2025 |
| 11 | Schwefel 2.6 | `Schwefel_2_6` | `Schwefel_2_6` | 2005 |
| 12 | Schwefel 2.13 | `Schwefel_2_13` | `Schwefel_2_13` | 2005 |
| 13 | Noncontinuous Rastrigin | `NonContinuousTransform` + `RastriginFunction` | (composed at runtime) | 2013–2022, 2024–2025 |
| 14 | Lennard-Jones | `LennardJonesFunction` | (none) | 2019 |
| 15 | Bent Cigar | `BentCigarFunction` | `bent_cigar` | 2013–2022, 2024–2025 |
| 16 | Discus (Tablet) | `DiscusFunction` | `discus` | 2013–2022, 2024–2025 |
| 17 | Different Powers | `DifferentPowersFunction` | `different_powers` | 2013–2020, 2024–2025 |
| 18 | Schaffers F7 | `SchaffersF7Function` | `schaffers_f7` | 2013–2022, 2024–2025 |
| 19 | Katsuura | `KatsuuraFunction` | `katsuura` | 2013–2022, 2024–2025 |
| 20 | Lunacek Bi-Rastrigin | `LunacekBiRastriginFunction` | `lunacek_bi_rastrigin` | 2013–2022, 2024–2025 |
| 21 | HappyCat | `HappyCatFunction` | `happycat` | 2014–2022, 2024–2025 |
| 22 | HGBat | `HGBatFunction` | `hgbat` | 2014–2022, 2024–2025 |
| 23 | Zakharov | `ZakharovFunction` | `zakharov` | 2017, 2020, 2022, 2024–2025 |
| 24 | Levy | `LevyFunction` | `levy` | 2017, 2020, 2022, 2024–2025 |
| 25 | Dixon-Price | `DixonPriceFunction` | `dixon_price` | 2017, 2020, 2024–2025 |
| 26 | Sum of Different Powers | `SumDifferentPowersFunction` | `sum_different_powers` | 2017, 2020, 2024–2025 |

---

## CEC 2015 Niching Primitives

These are 1D/2D base functions "expanded" to D dimensions via sum-of-pairs or product patterns. Only used in CEC 2015 niching competition.

| # | CEC Name | Base Dimension | Status |
|---|----------|:--------------:|--------|
| 27 | Expanded Two-Peak Trap | 1D→D | **Done** (`ExpandedTwoPeakTrapFunction`) |
| 28 | Expanded Five-Uneven-Peak Trap | 1D→D | **Done** (`ExpandedFiveUnevenPeakTrapFunction`) |
| 29 | Expanded Equal Minima | 1D→D | **Done** (`ExpandedEqualMinimaFunction`) |
| 30 | Expanded Decreasing Minima | 1D→D | **Done** (`ExpandedDecreasingMinimaFunction`) |
| 31 | Expanded Uneven Minima | 1D→D | **Done** (`ExpandedUnevenMinimaFunction`) |
| 32 | Expanded Himmelblau | 2D→D | **Done** (`ExpandedHimmelblauFunction`) |
| 33 | Expanded Six-Hump Camel Back | 2D→D | **Done** (`ExpandedSixHumpCamelFunction`) |
| 34 | Modified Vincent | Scalable | **Done** (`ModifiedVincentFunction`) |

---

## CEC 2019 Special Functions

| # | CEC Name | Status | Notes |
|---|----------|--------|-------|
| 35 | Chebyshev | **Done** | `ChebyshevFunction` (CEC 2019 F1) |
| 36 | Hilbert | **Done** | `HilbertFunction` (CEC 2019 F2) |
| 37 | Lennard-Jones | **Done** | `LennardJonesFunction`, `LennardJonesCECFunction` (CEC 2019 F3) |


---

## Meta-function Infrastructure

All composition and hybrid meta-function support is **complete**. No new classes needed.

| Component | Class | Status | Used For |
|-----------|-------|--------|----------|
| Weighted composition | `WeightedComposition` | Done | cf01–cf10 across all CEC years |
| Hybrid partitioning | `HybridFunction` | Done | hf01–hf10 across all CEC years |
| Transform chaining | `ComposedFunction` | Done | Per-component transform pipelines |
| Shift | `ShiftTransform` | Done | All CEC suites |
| Rotate | `RotateTransform` | Done | All CEC suites |
| Scale | `ScaleTransform` | Done | Composition lambda scaling |
| Bias | `BiasTransform` | Done | Per-function bias offsets |
| Noise | `NoiseTransform` | Done | CEC 2005 F4/F17/F24/F25 |
| Non-continuous | `NonContinuousTransform` | Done | Noncontinuous Rastrigin, F22–F25 |
| Normalize | `NormalizeTransform` | Done | C/f_max normalization in compositions |

---

## Remaining Gaps

### Phase 4 — CEC 2015 Niching (8 functions, optional)

Lower priority unless niching competition support is explicitly needed. Requires sourcing 1D trap function definitions from CEC 2015 technical report.

### Phase 5 — CEC 2019 Special (2 functions, optional)

Chebyshev and Hilbert are unusual non-standard problems. Low priority.

---

## Notes

### Naming conventions in CEC source vs pyMOFL

| CEC Source Name | pyMOFL Convention | Notes |
|----------------|-------------------|-------|
| `sphere_func` | `SphereFunction` | PascalCase + "Function" suffix |
| `bent_cigar_func` | `BentCigarFunction` | |
| `hf01` | Built via `HybridFunction` | Not a standalone class |
| `cf01` | Built via `WeightedComposition` | Not a standalone class |
| `schwefel_func` | `SchwefelFunction` | Offset form (CEC convention) |
| `escaffer6_func` | `Schaffer_F6_Expanded` | "Expanded Scaffer F6" |

### Noncontinuous Rastrigin

Not a standalone class — built at runtime as `ComposedFunction(NonContinuousTransform → RastriginFunction)`. This matches the CEC evaluator pattern where `noncontinuous_rastrigin` applies the non-continuous mapping before Rastrigin evaluation.

### Formula sources and corrections

- CEC 2013–2017: Formulas available in CEC technical reports and C source
- CEC 2005: Already fully implemented via factory + `cec2005_suite.json`
- Schaffers F7: Often confused with Schaffer F6 (different function entirely). Uses `1/(D-1)` normalization (not `1/D`).
- Different Powers vs Sum of Different Powers: Different exponent schedules — `2+4(i-1)/(D-1)` vs `i+1`
- Different Powers: CEC wraps entire sum in `sqrt()` — commonly omitted in reference docs
- HappyCat/HGBat: Denominator is **D**, not 2D (corrected from some reference descriptions)
- Lunacek Bi-Rastrigin: Rastrigin cosine is centered at μ₀ — `cos(2π(xᵢ-μ₀))`, not `cos(2πxᵢ)`. Parameter `s` is dimension-dependent: `s = 1 - 1/(2√(D+20) - 8.2)`.
