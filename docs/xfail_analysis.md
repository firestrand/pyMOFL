# Known Deviations from Reference Implementations

This document catalogues all known cases where pyMOFL's output intentionally
differs from reference C/COCO implementations, grouped by root cause.

---

## Summary

| # | Root Cause | Count | Category |
|---|------------|-------|----------|
| 1 | Float64 precision amplification | 2 | Fundamental |
| 2 | SchafferF7 stale buffer (C bug) | 8 | Reference bug |
| 3 | CEC 2013 composition buffer aliasing | 24 | Reference bug + architecture limit |
| 4 | BBOB monolithic functions | 27 | Architecture limit |
| | **Total** | **61** | |

---

## 1. Float64 Precision Amplification in Asymmetric Transform

| Suite | Functions | Dims | Count |
|-------|-----------|------|-------|
| CEC 2013 | F8 (Ackley) | D10, D30 | 2 |

**Root cause:** The CEC 2013 asymmetric transform (`asyfunc`) raises rotated
input values to exponents up to ~8. When rotated values are large (e.g., 219
for `bounds_max` at D30), the result reaches ~1.22×10¹⁹. At this magnitude,
float64 ULP (unit in the last place) is ~10⁴, so even bit-level agreement in
the rotation output produces absolute differences of ~10⁴ in the asymmetric
result. These differences propagate through the subsequent conditioning and
rotation steps, ultimately causing a ~7.5×10⁻³ difference in the final Ackley
value against a tolerance of 6.78×10⁻⁴.

**Why it cannot match:** Both C and Python compute identical exponents and use
the same float64 arithmetic. The difference arises from non-associativity of
floating-point matrix multiplication in the rotation step — the two
implementations multiply the same values in a different order (BLAS vs C loop),
producing results that agree to 15 significant digits but differ in the last
ULP. With an 8th-power amplification, that ULP difference becomes macroscopic.

**Verification:** The intermediate values (shifted, rotated, asy-transformed)
were compared element-by-element between a compiled C reference and Python.
All values agree to machine epsilon; the final difference is purely from
float64 amplification.

---

## 2. SchafferF7 Stale Global Buffer in Hybrid Functions

**pyMOFL implements SchafferF7 correctly; the reference C code contains a bug.**

| Suite | Functions | Dims | Count |
|-------|-----------|------|-------|
| CEC 2017 | F14 (HF04), F20 (HF10) | D10, D30, D50 | 6 |
| CEC 2022 | F7 (HF10) | D10, D20 | 2 |

**Root cause:** The CEC C reference implementation uses global buffers `y[]`
and `z[]` that are shared across all function calls within a single evaluation.
In `schaffer_F7_func`, line 536 reads from `y[]`:

```c
z[i] = pow(y[i]*y[i] + y[i+1]*y[i+1], 0.5);
```

However, `y[]` was last written by `sr_func` inside the **previous** hybrid
component (e.g., `ackley_func` or `schwefel_func`), not by `schaffer_F7_func`'s
own `sr_func` call. The correct code should read from `z[]` (the output of
`schaffer_F7_func`'s own `sr_func`). This is a bug in the reference C code.

**Why we don't replicate it:** Reproducing this bug would require the
`HybridFunction` composition to leak intermediate buffer state between
components — each component's `sr_func` side-effect would need to be captured
and injected into the next component. This fundamentally contradicts the
modular, stateless transform architecture.

**Affected hybrid chains:**
- HF04: `ellips → ackley → **schaffer_F7** → rastrigin`
  (SchafferF7 reads stale `y[]` from ackley's `sr_func`)
- HF10: `hgbat → katsuura → ackley → rastrigin → schwefel → **schaffer_F7**`
  (SchafferF7 reads stale `y[]` from schwefel's `sr_func`)

---

## 3. CEC 2013 Composition Functions — Buffer Aliasing + Hardcoded Internals

| Suite | Functions | Dims | Count |
|-------|-----------|------|-------|
| CEC 2013 | F21–F28 | D10, D30, D50 | 24 |

**Root cause:** CEC 2013 composition functions (CF01–CF08) combine the buffer
aliasing bug from `asyfunc` (§1) with hardcoded internal behaviors in
`schwefel_func`:

1. **`asyfunc` buffer aliasing:** When `asyfunc(z, y)` encounters `z[i] ≤ 0`,
   `y[i]` retains the value from a previous computation step. In compositions,
   each component function operates on different shifted/scaled input, making
   the stale buffer values context-dependent across components.

2. **Hardcoded `schwefel_func` internals:** The C code's `schwefel_func` has
   hardcoded `y *= 1000/100` scaling, internal conditioning
   (`10^(0.5*i/(D-1))`), offset (+420.97), and boundary handling — all inside
   the function rather than as external transforms.

3. **Component interaction:** Each composition component runs through
   `shift → scale → rotate → asy/osz → conditioning → rotate → base_func`,
   with the global `y[]`/`z[]` buffers carrying state between steps. The exact
   buffer contents when `asyfunc` encounters negative values depend on the
   entire prior computation chain, including the composition builder's shift
   and scale operations.

**Why it cannot match:** Full reproduction would require a monolithic
reimplementation of the C code's global-state pipeline for each composition
function, defeating the purpose of modular composition. The individual
component functions (sphere, rastrigin, ackley, schwefel, etc.) are implemented
correctly; the discrepancy arises from C-code-specific buffer aliasing in the
composition pipeline.

**Magnitude:** Typical differences are 0.5–5% of the function value, varying
by dimension and test point.

---

## 4. BBOB Monolithic Functions (COCO)

| Suite | Functions | Dims | Tests | Count |
|-------|-----------|------|-------|-------|
| BBOB | f5 (Linear Slope), f20 (Schwefel), f24 (Lunacek) | D2, D10, D40 | ×3 test classes | 27 |

**Root cause:** Three BBOB/COCO functions have monolithic internal
implementations that cannot be decomposed into the external transform pipeline:

- **f5 (Linear Slope):** COCO applies `max(xopt_i * x_i, 5 * xopt_i)` with
  sign-dependent boundary handling that is coupled to the function's slope
  computation.
- **f20 (Schwefel):** COCO's implementation has internal hat-function
  conditioning and boundary corrections interleaved with the main computation.
- **f24 (Lunacek Bi-Rastrigin):** COCO-specific `xopt`/`fopt` generation
  couples optimum location to the function's internal structure, plus internal
  affine transforms and penalty scaling.

**Why it cannot match:** These functions' internal logic is tightly coupled —
the transforms and base function are not separable. Reproducing exact COCO
values would require monolithic reimplementations that bypass the
`ComposedFunction` architecture. The mathematically equivalent functions are
provided; only the COCO-specific implementation details differ.
