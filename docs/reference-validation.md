---
title: pyMOFL reference validation provenance
version: 1.1.0
last_updated: 2026-10-04
status: locally-verified
owner: product-owner
tags: [reference-validation, provenance, reporting]
---

# Reference validation provenance

This document records acquired reference evidence and the implemented required
manifest contract. The selected CEC2014 scope passes local required execution
and rejection/failure checks. Remote CI and other suites remain unverified.

## Official source snapshots

Sources were obtained from the benchmark author's GitHub account, linked by the
[author's benchmark page](https://sites.google.com/view/suganthan-p-n/cec-benchmark/cec13-special-session).
The local snapshots are outside this repository. No license file was observed in
the selected archives; rights are recorded as unspecified. This work does not
redistribute reference sources/data or assert an open-source license for them.

| Repository | Pinned revision | Selected archive | Verification status |
| :--- | :--- | :--- | :--- |
| [CEC2013](https://github.com/P-N-Suganthan/CEC2013) | d087389fded052af4f127b23efaeb13120b7ea15 | cec13-c-code.zip | Inventory only; known buffer behavior needs independent controls |
| [CEC2014](https://github.com/P-N-Suganthan/CEC2014) | 98488087d590c29aaded9978ccfe2a356d10dd63 | cec14-c-code.zip | Local captures and existing tests verified below |
| [CEC2017](https://github.com/P-N-Suganthan/CEC2017-BoundContrained) | 2c54cad22f015e803edb09ea86d4c961f5bab644 | CEC17_fast_pow-C++.zip | Inventory only |
| [CEC2020](https://github.com/P-N-Suganthan/2020-Bound-Constrained-Opt-Benchmark) | d8b4c52f161562cd462e9b3352885e8df6fd2e41 | Software.zip | Inventory only |
| [CEC2022](https://github.com/P-N-Suganthan/2022-SO-BO) | de20505283b76ec6bf17a2e8fc6052e655830691 | CEC2022.zip | Inventory only |

Archive SHA256 values are retained in the task's official-reference-inventory.json.
Only the CEC2014 snapshot is currently approved for the proposed required job;
this scope does not claim reference coverage for every bundled suite or COCO.

## CEC2014 capture procedure

The selected archive SHA256 is
`1a210560398ca7a50be6adf1e5e90602222519ef23b6e31aba8847e109761876`.
The capture uses its cec14-c-code/cec14_test_func.cpp and input_data files.
Source inspection found no shell/network execution or writes; reads use relative
input_data paths. Extraction rejects absolute and parent-traversal archive paths.

Two compatibility patches are retained separately from the original source:
remove the unused WINDOWS.H include and replace five fscanf `%Lf` formats with
`%lf`. The destination arrays are double*, while `%Lf` expects long double* on
Linux. No formulas, ordering, constants, tolerances or reference bugs are changed.
The official void-main demonstration is replaced by a small input/output driver.

Compiler: g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0, Linux aarch64.
Flags: -std=c++17 -O2 -fno-fast-math -ffp-contract=off -Wall -Wextra -Wformat=2.
Each evaluator subprocess has a 30-second timeout; compilation has 120 seconds.
Every call has an explicit valid function number, dimension and five finite rows.
Required matrix/shuffle files are checked before execution. The capture engine
never enters pyMOFL's import or evaluation path.

Coverage is F1–F30 at D10, D30 and D50: 90 files, 450 records. Inputs are:

- `shift`: the first D coordinates of the official function shift file, as used
  in the official demonstration driver.
- `zeros`: the official demonstration driver's zero vector.
- `random`, `bounds_min`, `bounds_max`: unchanged corresponding rows from
  tests/validation_data/cec/2005/f01.json, with that file's SHA256 recorded.

Outputs come from the compiled official evaluator, not pyMOFL. Each file is
captured twice with byte-identical process output. JSONL records preserve the
existing loader fields func_id/dim/case/x/value; serialization rejects nonfinite
outputs. The provenance manifest includes source, patch, driver, executable,
input payload, archive, data-file and capture SHA256 values and compiler flags.
It is currently local at /tmp/pymofl-implementation/official-captures/provenance.json,
SHA256 `21952b03baed5389126448a55a8d6ed2184fb0ab0857c933545d5c7ae7ca77d6`.
Task-local acquisition script: /tmp/pymofl-implementation/capture-cec2014.py.
These initial temporary paths are historical observations. Retained setup,
required manifest and explicit CI provisioning are described below.

The reviewed procedure is now retained in scripts/capture_cec2014.py. It accepts
a local archive and an empty external directory, checks the approved archive and
DATA-01 hashes before compiling, bounds compiler-version inspection to 30 seconds,
and preserves the original source/patch/driver inside the external output. It
reproduced all 90 approved file/input hashes. A different actual official archive
(CEC2013) was rejected before compilation. No source or captures are bundled.

```bash
uv run --locked python scripts/capture_cec2014.py --archive /tmp/pymofl-implementation/official-cec2014/cec14-c-code.zip --output /tmp/pymofl-reference-captures
```

The output directory must be empty and outside this checkout. The explicit
download/source acquisition is separate; ordinary evaluation never fetches data.
The owner authorized the remaining synthetic test data on 2026-10-04.
V5.2/V5.4 missing/duplicate/invalid/provenance and operational failure checks
now pass; original authoritative captures remain untouched.

## Required manifest producer and observed report

The optional `pyMOFL.reference_validation` helper reads the actual retained
setup and emits a version1 manifest. Required coverage is independently fixed
to F1–F30, D10/30/50 and all five named cases. It checks the pinned setup
identity and rehashes available patched source, patch, driver, executable,
data and capture files, plus reconstructed input payloads. Original source
and archive hashes are recorded setup identities when those bytes are absent;
they are not described as newly rehashed originals. Hashes do not establish
authenticity or latest-release status.

Create the manifest from approved local captures:

```python
import json
from pathlib import Path
from pyMOFL.reference_validation import prepare_reference_manifest

manifest = prepare_reference_manifest("/tmp/pymofl-reference-captures")
with Path("/tmp/pymofl-reference-manifest.json").open("x") as output:
    output.write(json.dumps(manifest, indent=2, allow_nan=False))
```

The explicit stdlib runner uses that manifest and a fresh report path:

```bash
uv run --locked python scripts/verify_reference_manifest.py --capture-root /tmp/pymofl-reference-captures --manifest /tmp/pymofl-reference-manifest.json --report /tmp/pymofl-reference-report.json
```

The report records all450 canonical IDs, actual scalar/batch execution flags
and counts, observed passed/failed/unavailable/deviation statuses, source/
software identities, and absolute-difference metrics without raw coordinates.
Each path retains the existing strict tolerance
`abs(actual - expected) < 1e-6 * max(1, abs(expected))`. The approved CEC2014
deviation set is empty; supplied reasons cannot override that policy.
Zero exit requires all required cases observed passed. Malformed/unavailable
provenance and operational errors do not become deviations or successful skips.
Preflight malformed/unavailable inputs report 450 unavailable cases without
claiming execution. Real construction/scalar/batch errors report failed cases
with honest per-path counters. Both required helper and runner have 100%
measured statements and branches, including authorized rejection checks.

With the existing `cli` extra, the optional adapter produces the same report:

```bash
uv run --locked pymofl validate --capture-root /tmp/pymofl-reference-captures --manifest /tmp/pymofl-reference-manifest.json --report /tmp/pymofl-cli-reference-report.json
```

Use a fresh report path. Global `--json` also emits the report on stdout;
`--quiet` suppresses the summary while retaining the report file. The command
uses the library-neutral validator; it adds no independent numerical logic.
Default/JSON/quiet reporting, generated failure/status/exit sensitivity and
fresh-report write errors are verified. The optional adapter has 100% measured
statements and branches.

## Observed controls and results

All 204 compared shift, rotation and required shuffle arrays were exactly equal
to the corresponding bundled constants, including their shapes. Official F1 at
its shift vector returned exactly 100.0 at all three dimensions. All 450 inputs
were additionally repeated five times each in fresh processes: all 2,250 outputs
exactly matched their ordered captures, with no observed row-history dependence.

Provisioned existing validation ran with:

```bash
CEC_BENCHMARKS_PATH=/tmp/pymofl-implementation/official-captures uv run --locked pytest tests/utils/test_golden_loader.py tests/benchmark_suites/test_cec2014_validation.py -q
```

Result: 123 passed, zero skipped. Existing assertions and tolerances were
unchanged. Raw log: /tmp/pymofl-implementation/cec2014-capture-check.log.
This is empirical agreement on these cases, not a proof for all possible inputs.

The compiler warns about unchecked fscanf return values; capture controls verify
the actual data and lengths before invoking the reference. It also warns about
Weierstrass sum2 and oszfunc xx. In the supported positive-dimension Weierstrass
path, sum2 is initialized within the coordinate loop before its final use.
oszfunc has no call site in this selected evaluator. No warning was suppressed
or reference function patched to obtain passing comparisons.

A second build with -O1 -fsanitize=undefined -fno-sanitize-recover=all,
-fno-fast-math and -ffp-contract=off ran all 90 processes / 450 records.
There were no sanitizer diagnostics and its outputs were bit-for-bit equal to
the -O2 capture on this environment. This does not claim exhaustive absence of
undefined behavior. The oracle independently verified all capture hashes,
reran the 123 tests, inspected the source/patch/driver and approved this local
capture scope at the existing tolerances on 2026-10-03.

Other suite snapshots, COCO, historical CEC2005 generation claims and documented
reference deviations remain separate evidence obligations. Their absence must
stay visible in reports; they cannot be counted as passed by this capture.

## Explicit required integration workflow

The manual `.github/workflows/reference-validation.yml` uses Python 3.12 and
3.13, pinned actions/uv and locked dependencies. It retrieves the exact pinned
CEC2014 archive over HTTPS, verifies its SHA256, runs the retained source setup,
creates the actual manifest and executes the required runner and selected
numerical/helper/CLI tests. A final JUnit/report guard rejects any required skip,
error, failure, empty test set or incomplete 450 scalar/batch execution.
No reference engine enters ordinary library evaluation or default CI.

Local equivalents passed on both minors: 450 scalar and batch cases and 177
selected tests with zero skips/errors/failures. External artifacts are in
`/tmp/pymofl-implementation/ci-equivalent312` and `ci-equivalent313`.
The workflow is configured and locally exercised; remote execution remains
unobserved while these changes are uncommitted/unpublished. No archive,
executable or rights-unspecified reference output is uploaded by the workflow.
