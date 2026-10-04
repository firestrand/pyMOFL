---
title: pyMOFL selected SPSO source review
version: 1.1.0
last_updated: 2026-10-04
status: supported-scope-reviewed
owner: product-owner
tags: [spso, reference-validation, provenance]
---

# Selected SPSO source review

This acquisition covers Tripod (ID 4, D2), Network (11, D42), Gear Train
(18, D4) and Compression Spring (21, D3). It does not validate the full SPSO
test catalog or embed an optimizer.

The official [program listing](https://www.particleswarm.info/Programs.html)
links the [2007 archive](https://www.particleswarm.info/standard_pso_2007.zip)
and [2011 C archive](https://www.particleswarm.info/standard_pso_2011_c.zip).
Their SHA256 identities are respectively
f9524f7f9568009b4ab5c76cd32d91c255fef978b4ff64891b460cb520f34bd1 and
11692f658158b18aafd97d667eeebdc7527cf21147d530d73d4c7eb795af0557.
The 2007 source was updated through 2011-01-08; the 2011 distribution includes
later updates, including the 2011-05-16 Spring correction. The labels identify
these pinned distributions, not untouched releases from their namesake years.
Overall source/data redistribution rights were not established. Archives,
source, executables and captures remain outside the repository.

## Capture contract

External source bytes were checked against each pinned archive. For 2007,
the reference harness renames optimizer main and never invokes it. For 2011,
the harness links the original perf/problemDef/tools and exactly extracted
quantis implementation. Unused GSL header declarations are removed. Abort-only
PSO and alea_normal stubs cover unselected optimizer/noise branches; an
unexpected call terminates rather than supplying a made-up value. Independent
oracle review approved both unreachable boundaries before execution.

Captures use native coordinates: 2011's final constructor quantum-normalization
branch is disabled in the external adapter, preserving the native step constants,
and the returned SS.normalise flag is set to zero before quantis/perf. Disabling
only the returned flag would leave normalized step sizes and be incorrect.
Objective and constraint formulas remain unchanged.

Inputs are the actual constructor lower/upper bounds and source-documented
Tripod (0,-50) and Gear (16,19,43,49) solutions. Each version produced four
metadata records and ten evaluation records. The owner-authorized boundary
extension below preserves these initial controls unchanged.

GCC 13.3.0 on Linux aarch64 compiled C99 with O2, -fno-fast-math,
-ffp-contract=off, -Wall, -Wextra and -Wformat=2. Each executable repeated its
capture twice with identical bytes. Separate O1 UBSan builds also repeated
identically, emitted no runtime diagnostics, and matched all release bytes.
UBSan command flags include the shared O2 flags followed by -O1 (which takes
precedence), -fsanitize=undefined and -fno-sanitize-recover=all; provenance
records each build's complete flag list separately.
The external provenance file records source/patch/driver/executable hashes and
actual compiler/platform. Capture outputs have SHA256:

| Distribution | Release and UBSan JSONL SHA256 |
| :--- | :--- |
| 2007 | 422812db86a36dcdee25526dfea724f4698bc2a107feb8772a5c6aeef64c73cf |
| 2011 | 3d338e9d7f4b9e54cbbbd07957b46fbf6ca494ed1aa1a668742bac4de17183ef |

Compiler warnings were retained. The 2007 warnings include uninitialized
optimizer variables in the unused optimizer main/PSO paths. The 2011 constructor
contains an unrelated ID 24/D80 path exceeding its DMax42 arrays; the harness
calls only IDs 4/11/18/21. Unused variables/stub parameters and an integer fabs
in the unused Gamma helper also warn. Zero observed UBSan diagnostics establish
only the captured selected paths, not general source safety.

## Differences requiring explicit variants

| Definition | Pinned source contract | Existing library behavior / disposition |
| :--- | :--- | :--- |
| Tripod | Source sign(0)=0 produces half coefficients on axes | Existing Tripod treats zero as nonnegative. Add an explicit source-backed variant; preserve existing default |
| Network | First 38 coordinates are quantized binary assignments; remaining four are continuous BSC coordinates in [0,20] | Existing node coordinates agree; apply source half-up quantization explicitly through a vector transform |
| Gear | Squared ratio error, then absolute distance from target 2.7e-12 | Reuse GearTrain, Power(2), Bias(-target), Absolute; preserve generic GearTrain's absolute ratio error |
| Spring | Native [N,D,d], bounds [1,.6,.207] to [70,3,.5], steps [1,0,.001]; multiplicative cubic penalties and distance from target 2.6254214578 | Existing Spring uses [d,D,N], other constraints, scaling and additive squared penalties. Add an explicit variant |
| Quantization | q*floor(.5+x/q) for q>1e-40, zero-step passthrough | Existing Quantized uses np.rint. Add a separate primitive; do not change its default |

The pinned 2007 Spring source has a historical bug: when the free-length
constraint g2 is positive, it uses (1+g1)^3 instead of (1+g2)^3. The 2011 source
corrects this. At the actual upper constructor bound, reference objective
distances are about 4.2687e17 and 2.8909e7 respectively. A source-faithful
2007 definition must name and disclose this difference. Neither source target
establishes a certified optimum point for Spring; do not invent one.

The oracle approved the selected source captures and supported-data implementation.
Both suites now match all 20 captured scalar/batch values and quantized inputs
within the declared tolerance, with matching bounds and fixed dimensions.
Authorized boundary, quantization and fixed-dimension rejection checks now pass.
New mathematical/quantization primitives have 100% measured statement/branch
coverage; final integrated review is recorded in implementation-progress.md.
Source-equivalence claims apply within declared search bounds. Network's
existing thresholding is equivalent after source quantization for in-bounds
0/1 assignments; it differs from source link sums outside those bounds. Bounds
remain metadata without clipping or any external-domain equivalence promise.

Explicit reproducible setup (local pinned archives only, GCC required):

```bash
uv run --locked python scripts/capture_spso.py --archives /path/to/local-archives --output /path/to/empty-external-output
PYMOFL_SPSO_REFERENCE_PATH=/path/to/empty-external-output uv run --locked pytest tests/benchmark_suites/test_spso_validation.py -q
```

The retained procedure reproduced both approved JSONL hashes in a fresh external
directory. Default tests do not fetch or compile the source; four source checks
skip when the explicit reference directory is not provisioned.

## Authorized source boundary extension

`scripts/capture_spso_boundaries.py` uses the same pinned archives and initial
controls. Inputs are the nine Tripod axis/bound combinations and, for each
active quantum, lower/upper half-step points plus adjacent floating-point
neighbors. Per version the coverage is Tripod9, Network228, Gear24 and Spring12:
273 cases, 546 across both distributions. Official `quantis` and `perf` supply
all expected values. No library-generated expected output is used.

The provenance records driver/input/executable/output hashes and flags.
Each release/UBSan payload is repeated twice; all outputs agree with zero
runtime diagnostics. Tests independently derive all prescribed inputs from
the pinned metadata, verify original control hashes, require every selected
ID/native dimension/count/order, compare source-quantized coordinates and
scalar/batch values, and check input ownership. The oracle independently
reran all 546 source cases on both builds.

```bash
uv run --locked python scripts/capture_spso_boundaries.py --archives /path/to/local-archives --output /path/to/empty-external-boundary-output
PYMOFL_SPSO_REFERENCE_PATH=/path/to/empty-external-boundary-output PYMOFL_SPSO_BOUNDARY_PATH=/path/to/empty-external-boundary-output uv run --locked pytest tests/benchmark_suites/test_spso_validation.py tests/factories/test_fixed_dimension.py -q
```

Default tests require explicit external source provisioning and otherwise skip
source-only comparisons. Operational producer failures are separately measured
using labeled fault wrappers; they never replace authoritative reference data.
