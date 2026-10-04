---
title: pyMOFL implementation evidence
version: 1.3.1
last_updated: 2026-10-04
status: oracle-approved-local-implementation
owner: product-owner
tags: [verification, packaging, architecture]
---

# pyMOFL implementation evidence

Implementation follows the [approved proposal](./improvement-proposal.md), version
1.4.2, against baseline `e2c56aa90598898cd148b850f18329c532591496`.
The current local implementation covers all selected V0–V8 scope, including
owner-authorized generated failure/boundary checks. The oracle approved the complete selected local implementation on 2026-10-04. The owner separately authorized main-branch commit/push, a version bump and tagging on 2026-10-04.

The sections through “Remaining gates after supported-scope audit” are dated
2026-10-03 checkpoints. Their unanswered/pending descriptions preserve what was
known then; the 2026-10-04 completion record below supersedes those gates.
Baseline findings describe the original revision, not this modified worktree.

## Installed-package and discovery changes

The following bindings record observed local implementation/evidence results.
They do not promote the approved proposal's requirement lifecycles to Active or
approve later facts. V1's local verification is complete and its changes are
staged for human review; remote CI and final owner acceptance are not recorded.

| Fact | Current implementation result | Evidence | Current result and binding |
| :--- | :--- | :--- | :--- |
| F01: installed core optionality | Implemented; locally verified | E01 | Passed: installed-artifact checker against real wheel, separate clean core/CLI environments, bundled F1/D10 capture, locked/minimum/compatible dependency checks below |
| F02: extension discovery compatibility | Implemented; locally verified | E02 | Passed: all 344 captured baseline identities, explicit scanning, genuine late registration through loading and factory, registry/API tests on Python 3.12/3.13/3.14 |

This V1 section does not establish later facts; V2–V4 results follow below. The oracle
independently approved V1's local verification and 90% checker branch coverage on
2026-10-03; these results do not establish later-phase completion.

Automatic discovery imports the benchmark package once, rather than scanning the
whole library at import and on every load. Explicit `scan_package()` and live
registration remain available, including the existing scanner attributes exposed
by the package and loader. Objective mathematics and numerical defaults are
unchanged. Optional CLI modules are excluded from automatic discovery.

The actual baseline inventory is retained in
[baseline-aliases.json](../tests/fixtures/registry/baseline-aliases.json): all 344
alias-to-qualified-class mappings are protected by
[registry tests](../tests/test_registry.py). A late-registration check replaces
the real `sphere` alias with the existing `RastriginFunction`, verifies loading
and factory registration, and restores the original class. A separate local
sensitivity check removed and then remapped `decomposed_function`; the durable
baseline check rejected both mutations and passed after restoration.

[The installed-artifact checker](../scripts/verify_installed_package.py) installs
the built wheel into separate clean core and CLI environments outside checkout.
It checks import origin, package/version metadata, bundled CEC2005 F1/D10 optimum
evaluation, Sphere alias loading, and the installed CLI entry point. Core checks
require optional CLI/documentation/test packages to be absent. Provisioning is
explicit, bounded, and separate from default pytest execution.

CI now uses locked explicit extras, reviewed action commit pins, uv 0.12.22,
separate [build constraints](../scripts/build-constraints.txt), and the same
artifact checker. Python 3.12 also exercises the
[declared minimum runtime versions](../scripts/minimum-runtime-constraints.txt).
Dependabot proposes action updates monthly. These workflow edits have been
reviewed locally; a GitHub Actions run has not been observed.

## Verification results

Local Linux aarch64 verification used uv 0.12.22. After the final registry tests,
each full suite reported the following results. The warning is the inherited
Lunacek one-dimensional runtime warning.

| Interpreter | Passed | Skipped | Xfailed | Warnings |
| :--- | ---: | ---: | ---: | ---: |
| Python 3.12.14 | 2,853 | 531 | 1 | 1 |
| Python 3.13.15 | 2,853 | 531 | 1 | 1 |
| Python 3.14.8 | 2,853 | 531 | 1 | 1 |

Ruff formatting, Ruff lint, ty, and `git diff --check` passed. The focused registry
and API suite passed all 22 tests on all three interpreters. On Python 3.12,
statement coverage was 92.6261% and branch coverage 76.6497%, preserving the
proposal's rounded legacy baseline. Changed discovery statements were covered;
registry branch coverage was 100%. The existing uncovered loader branches were
not changed by V1. These percentages measure package-source coverage, not the
entire standards library.

Separate checker instrumentation combined successful file/directory invocation,
with and without constraints, successful imported `verify()` invocation, the
actual baseline wheel's missing-Typer failure, rejection of the existing `tests/`
directory containing no wheels, and rejection of a genuine non-pyMOFL wheel.
The checker measured 97.2222% statement coverage and 90% branch coverage
(18 of 20 branches), meeting the proposal's changed-executable threshold without
exclusions or a lowered gate. Missing uv and incorrect CLI version remain
uncovered; those paths have not been certified by this measurement.

The rejected foreign artifact was the already-existing `pip-26.0.1` wheel
embedded in virtualenv under the local uv cache, SHA256
`bdb1b08f4274833d62c1aa29e20907365a2ceb950410df15fc9521bad440122b`.
Only its package metadata was read; it was neither modified nor installed.
No fixture was generated and no environment or return value was patched. The
additional V1.4 failure-fixture authorization question is therefore no longer
needed to meet this coverage gate; it grants no permission for future fixtures.

Separate clean core and CLI artifact checks passed with:

| Dependency setup | Python | NumPy | Matplotlib |
| :--- | :--- | :--- | :--- |
| Exported lock constraints | 3.12.14, 3.13.15, 3.14.8 | 2.4.2 | 3.10.8 |
| Declared minimum runtime constraints | 3.12.14 | 1.26.0 | 3.8.0 |
| Compatible ranges without lock constraints | 3.12.14 | 2.5.3 | 3.11.2 |

Wheel and source distribution builds passed using the separate build constraints.
The final locally built wheel SHA256 was
`949d1dae8a19196d82e9bda1558bc264137132840729d8a0b44f078e7984e931`;
the source distribution SHA256 was
`3024b646f8cc997bfc092518d57ddadd21b75b7c62f45e39b893c736b6364906`.
These identify the observed local artifacts; reproducible byte-identical builds
across environments have not been established.

To reproduce the package gate, run the proposal's common verification commands,
then:

```bash
uv build --no-sources --build-constraints scripts/build-constraints.txt --out-dir /tmp/pymofl-artifacts
uv export --locked --no-dev --extra cli --no-emit-project --output-file /tmp/pymofl-artifact-constraints.txt
uv run --locked python scripts/verify_installed_package.py --wheel /tmp/pymofl-artifacts --constraints /tmp/pymofl-artifact-constraints.txt
uv run --locked python scripts/verify_installed_package.py --wheel /tmp/pymofl-artifacts --constraints scripts/minimum-runtime-constraints.txt
```

Use Python 3.12 for the minimum-dependency check. Select a wheel file explicitly
when an output directory contains more than one wheel. Omit `--constraints` to
exercise freshly resolved compatible ranges, recording observed versions.

The independent oracle approved the V1 production/test/checker changes on
2026-10-03 after requiring stronger durable alias and late-registration evidence.
That approval does not cover later phases or unresolved data/compatibility
decisions. Local raw logs and GOTCHA/ATLAS working artifacts are under
`/tmp/pymofl-implementation`; they are temporary, not retained CI artifacts.

## Discovery measurement and limits

A same-environment source-construction probe used nine trials of 30
`load('sphere', dimension=30)` calls per trial, without generated numerical
workloads. Median time per construction changed from 939.3509 microseconds at
the baseline to 3.6507 microseconds with cached automatic discovery. Trial ranges
were 935.9775–946.2144 and 3.6086–5.3899 microseconds respectively. This measures
warm repeated construction on this machine, not cold import, function evaluation,
large-batch performance, or a portable latency guarantee. No timing assertion
was added to pytest.

## V2–V4 corrections and stream ownership

The owner accepted all five decision recommendations on 2026-10-03; proposal
1.2.0 records the exact scopes and expanded tasks. The oracle approved the V3/V4
plan and independently reviewed V2/V3 code (69 passed, one inherited skip) and
V4 code (87 focused checks passed). These approvals cover the integrated changes
and API documentation, not later reference adoption or numerical optimization.

| Facts | Evidence | Observed local result |
| :--- | :--- | :--- |
| F03/F04: single-call failures and valid batch preservation | E03/E04: composed/vectorization checks | Passed: original code produced four double-call failures; eight failure cases cover base/output, buffer-capable/bufferless, with/without output. Real F1/D10 captured values, input ownership, default fallback, opaque NumPy callable, live replacements and buffer identity pass |
| F05: canonical mutable suite lookup | E05: API tests | Passed: 17 new cases failed against the stale constructor index; all 34 API checks now pass, including every list mutator, metadata edits, duplicates, numeric-string migration and unrelated family IDs |
| F06: optional stream ownership | E06: noise ownership and existing noise/factory tests | Passed: 13 new cases were RED and four protected defaults already passed; supplied streams, seed conflicts, actual captured values and existing draw behavior now pass |

All three full suites (Python 3.12.14, 3.13.15 and 3.14.8) now report 2,906 passed,
531 skipped, one xfailed, and the inherited warning. Formatting, Ruff lint and ty
pass. Python 3.14 statement coverage is 92.8253%, branch coverage 77.1886%.
Changed dispatch helpers, composed batch evaluation, suite lookup, stream
resolver and legacy noise draw branches have 100% measured coverage; unchanged
constructor normalization and Uniform zero-batch branches remain legacy gaps.

Single-call dispatch decides keyword support before executing the current
method. Weak keys retain class-method capability metadata only while its function
exists, avoiding component/data retention; instance-bound closures are inspected
live. On nine trials of 1,000 actual four-row F1/D10 batch calls, the original
source measured a 2.7449-microsecond median and the final fix 3.6073 microseconds.
This is a correctness fix with measured overhead on a very small batch, not a
speedup claim. Larger workloads remain V6 work.

The seeded ownership evidence uses NumPy 2.4.2, PCG64 and legacy MT19937 with
seed 20261003, authorized by V0.2.2. All global state changes in tests are restored.
No noise formulas or scalar/batch draw order were changed. The shared resolver
validates the optional stream; it is not a new provider framework.

Migration and replay limitations are in [API compatibility notes](./api-compatibility.md),
linked from README. V2/V3 numeric and exception semantics were explicitly accepted
by the owner; other compatibility breaks are not implied.

The rebuilt wheel and sdist passed. Observed wheel SHA256:
`dfd6d9df8fe8baaed508b1909b9e5da6fa21b451a0c6349965bfa5ccce3c47fc`;
sdist SHA256:
`81c908d1e7f96180cad86c3dbc9ec333d57ff22fc07103fba02ad7c4003bf4a2`.
Clean core and CLI checks passed on Python 3.12 with runtime dependency floors
and Python 3.13 with exported lock constraints. Remote CI is still unobserved.

## Reference provenance and remaining gates

The 25 existing CEC2005 capture files were hashed before reuse. Their adjacent
[README](../tests/validation_data/cec/2005/README.md) attributes them to a
modernized CEC2005-C generator but does not pin a generating revision/environment.
That historical generation claim has not been independently reproduced. The
artifact checker reuses only the existing F1/D10 optimum record as packaging
evidence, not proof of full CEC conformance.

A read-only snapshot of the owner's
[cec-benchmarks repository](https://github.com/firestrand/cec-benchmarks) at
`4a27eebcd2a369c8e484d2fe600e20cfeb7d175e` contains CEC2005/2006 source and
captures, with a different JSON schema. It lacks the later-year
`datasets/<year>/func_<id>_D<dimension>/golden.jsonl` layout required by the current
golden loader. Running the existing focused validation tests against that root
reported 30 passed, 58 skipped, 26 xfailed, and one failure: the CEC2014 loader
test expected five records and received none. Directory existence therefore
does not establish reference readiness.

A structural comparison of existing CEC2005 captures with that snapshot found
400 corresponding records, 148 with identical input vectors, and 133 of those
with matching outputs at `rtol=1e-12, atol=1e-12`. The other 15 are F4 (8) and
F5 (7). Different RNG runs may explain noisy F4; deterministic F5 needs source
investigation. Neither explanation has been verified. Existing expected values
and tolerances have not been changed, and this snapshot has not been adopted as
a new reference authority.

The owner accepted all five recommended decisions on 2026-10-03. DEC-01–DEC-04
are recorded in proposal version 1.2.0; official reference acquisition/preparation
is authorized. CEC2014 reference readiness is verified below; required execution and measured
performance targets remain future work. V5–V8, including all four selected enhancements, remain
unfinished. The full goal has not been achieved.

Official local reference snapshots have been acquired. Only the pinned CEC2014
capture scope below has passed acquisition review. Sources are external to the
package. The selected archives have no observed license files; rights are
unspecified and Linux compatibility patches are retained separately.
No reference sources have been vendored or redistributed.

| Author repository | Pinned revision | Local archive |
| :--- | :--- | :--- |
| [CEC2013](https://github.com/P-N-Suganthan/CEC2013) | `d087389fded052af4f127b23efaeb13120b7ea15` | `cec13-c-code.zip` |
| [CEC2014](https://github.com/P-N-Suganthan/CEC2014) | `98488087d590c29aaded9978ccfe2a356d10dd63` | `cec14-c-code.zip` |
| [CEC2017](https://github.com/P-N-Suganthan/CEC2017-BoundContrained) | `2c54cad22f015e803edb09ea86d4c961f5bab644` | `CEC17_fast_pow-C++.zip` |
| [CEC2020](https://github.com/P-N-Suganthan/2020-Bound-Constrained-Opt-Benchmark) | `d8b4c52f161562cd462e9b3352885e8df6fd2e41` | `Software.zip` |
| [CEC2022](https://github.com/P-N-Suganthan/2022-SO-BO) | `de20505283b76ec6bf17a2e8fc6052e655830691` | `CEC2022.zip` |

Archive hashes, member inventories and read-only acquisition logs are recorded
under `/tmp/pymofl-implementation`.

V4.5 acquisition produced 90 CEC2014 files / 450 records: F1–F30 at D10,
D30 and D50. Official driver shift/zero vectors and unchanged DATA-01 F1
random/lower/upper inputs were evaluated by the pinned official C++ evaluator.
All 204 bundled shift/rotation/required-shuffle comparisons were exactly equal;
F1 shift controls returned 100.0. Duplicate captures were byte-identical and
2,250 repeated-input evaluations matched the captures exactly. A UBSan -O1
build ran all records without diagnostics and exactly matched the -O2 outputs.
Existing golden loader/CEC2014 validation tests: 123 passed, zero skipped.
The independent oracle reproduced the tests and verified the hashes/source/
compatibility patch before approving these local captures.

[Reference provenance](./reference-validation.md) records scope, compiler flags,
rights limits, source warnings, hashes and commands. F07/E07 remains Unknown:
acquisition readiness does not establish V5 required-manifest execution.
Other snapshots and COCO remain unverified/unavailable; no reference sources or
new reference vectors were redistributed.


## V5 setup and V6 optimization review

The retained capture setup in scripts/capture_cec2014.py reproduced all 90
approved capture/input hashes in a fresh external output directory. It checks
both source-archive and DATA-01 identity, validates data geometry, retains the
reviewed Linux patches, and bounds every subprocess including compiler-version
inspection. The oracle reviewed these boundaries. Required execution F07/E07
remains Unknown: V5.2/V5.4 deliberately invalid copies require a separate DP-05
scope-bound authorization, currently requested but unanswered. No such fixtures
have been created; V5.1 and independent probes proceeded.

[Performance review](./performance-review.md) and its retained raw measurement
fields establish F08/E08 for the declared diagnostic scope: 48 workloads, nine
alternating trials, supported input hashes and 195 working-source hashes.
The oracle independently recomputed timing summaries and verified both source
and report identities. Candidate-only patch/dispatch overhead, tracked versus
native memory, combined-worker RSS and missing exceptional-value/COCO evidence
are explicit. Universal Min/weights default rewrites were rejected; large-batch
weights remain a candidate for a separately protected future investigation.
Gallagher was profiled without a candidate/backend change. No numerical rewrite
was adopted by the probe.

Current source-tool checks: 365 files formatted, Ruff lint and ty passed;
default Python3.14 suite still 2906 passed, 531 skipped, one xfailed and one
existing warning. V4's supported-minor/artifact evidence above remains the last
observed production-package verification; standalone scripts are not installed
core dependencies. Remote CI remains unobserved.

Official SPSO 2007/2011 source archives were acquired externally for the selected
suite feature. Source inspection found Tripod zero-axis behavior and the spring
objective/constraints differ from the existing library definitions. No source
was vendored and no existing function changed. Source-backed explicit variants,
quantization/error semantics and reference capture must be specified/reviewed
before claiming SPSO equivalence. All four selected enhancement slices remain
unfinished; the full goal is still active.

## V7.1 supported-data helper and SPSO acquisition preparation

The oracle approved the explicit helper plan after clarifying NumPy integer/
floating input and child-result conversion to float64. V7.1.1 recorded 18
assertion-level failures for unavailable public evaluate_chunks, with unchanged
DATA01 CEC2005 F1/D10 inputs and outputs. The implementation makes all 18 pass;
helper plus API tests passed 52 checks on Python 3.12, 3.13 and 3.14. Each child
receives an owned float64 input chunk, results retain row order, and output
reuse/empty/readonly/strided-input behavior is covered. Existing evaluators keep
their signatures. Oracle review corrected the required declaration signature
and identified self-overlapping output elements; a static guard now rejects
those before evaluation. Its generated failure check remains pending.

The supported-data-only coverage run measured helper statement coverage
77.78% and branch coverage 66.67% before the extra output-element guard. It is
not final feature coverage. A first focused-module coverage collection failed
with a NumPy duplicate-module import; collecting the whole pyMOFL package then
passed all 18 tests and wrote the coverage report. V5.0/V7.0 generated negative
scopes remain unanswered; no such fixtures have been generated. FEAT-01 is
partial, not complete, until required failure/coverage/artifact checks pass.

The exact CI source type-check gate, whole-repository Ruff lint and format
passed (367 formatted files). An exploratory wider ty run found diagnostics in
old examples/tests and standalone scripts outside the CI type-check scope,
including missing competitor extras and the pstats Stats.stats stub boundary
in the new profiling script. Those wider checks did not pass; no diagnostics
were suppressed or CI scope weakened. The helper's full suites then passed on
all three minors: 2924 passed, 531 skipped, one xfailed and one existing warning.
Installed core and CLI artifacts passed on Python 3.12 with minimum NumPy1.26/
Matplotlib3.8 and on Python3.13 with locked NumPy2.4.2/Matplotlib3.10.8; the smoke
also evaluated all four unchanged DATA01 rows with chunk3 and an output buffer.
The built wheel SHA256 is faa47b6f29f3ce6daefe94619572a333992324c748db7aaa404c9342e5fcc9c8.
A subsequent extra positive test for exactly integral real bounds and reversed
output storage passed; focused helper checks now total 19. Its production code
is unchanged from artifact/full-suite verification. No final negative-path
coverage or whole-feature closure is claimed.

The oracle approved V7.3.0 official-source acquisition planning. External
drivers were prepared and reviewed before source execution. Inspection
found 2011 constructor quantum normalization, requiring an explicit native-
coordinate adapter before source quantis/perf, and a 2007 Spring g2 multiplier
bug corrected in 2011. Captures preserve and label the pinned original
formulas; these findings do not authorize changing existing library functions.
Both versions produced 20 cases and eight metadata records collectively,
each repeated twice under release/UBSan with exact byte agreement and zero
runtime diagnostics. The oracle independently reproduced and approved this
selected source scope. A first 2011 link attempt found an unselected CEC2005
noise dependency; an abort-only RNG stub was reviewed as unreachable for the
selected IDs before resuming. Warnings outside selected paths remain explicit
in [SPSO source review](./spso-reference-review.md). The full V7.3 tasks were
approved for supported-data implementation with native-domain limits, explicit
historical/corrected Spring variants, two primitive transforms and a fixed
heterogeneous dimension policy. Library implementation and generated boundary
checks remain unfinished.

The selected source implementation then passed six provisioned checks covering
20 real scalar/batch outputs, quantized inputs, empty source slices, reusable
transform buffers, actual bounds and unchanged caller configs. The oracle
independently ran 64 source/API/registry/helper checks and approved this supported
scope. A narrow opt-in FunctionFactory fixed_dimension keyword avoids changing
legacy constructor/parser behavior; original conflicting dimensions and
composition delegation are rejected before constructing a fixed base.

The supported full suites passed 2931 checks on both Python3.12 and3.13 with
531 existing skips, one xfail and one warning. Default3.14 passed2927 with535
skips; four additional skips are the explicitly unprovisioned SPSO source tests.
Separate initialization/operational defaults for the new direct Spring alias
were corrected after this run; focused source checks and both installed artifact
environments passed after that metadata change. New SPSO JSON configs and public
loading were exercised outside checkout in core and CLI environments. The
artifact wheel SHA256 is452633ad439ee52993073f3b5f243659eddbbedd9d779a0c420d6e32cde3e4e5.
The artifact smoke initially assumed a composed optimum method; existing
ComposedFunction does not implement one. It now uses the actual Tripod base's
documented optimum without changing that legacy wrapper contract.

The retained scripts/capture_spso.py takes explicit local archive/output paths,
preserves originals/patches/flags, avoids optimized-away assert guards, and
reproduced both approved JSONL hashes. Final invalid-control/boundary coverage
remains pending V7.0, not disguised as source-case coverage. V7.4's optional
definition/export/replay contract was independently approved before dispatch.

## V7.4 supported definition export and replay

Seven canonical CEC2005, BBOB, GNBG, CEC2014 and selected SPSO requests now
produce strict JSON records through the real exporter and replay numerically
through the existing loader. Availability assertions supplied the initial RED.
The first supported test incorrectly assumed BBOB had bundled data artifacts;
its actual factory-generated configuration correctly has an empty artifact list.
The type checker also required explicit narrowing of untrusted JSON fields.
Both were corrected without fabricated records or weakened gates.

The oracle identified two static implementation issues: Python dictionary
equality conflated JSON booleans/numbers, and GNBG provenance used a generic
resolver rather than the factory's selected configuration. Canonical digest
comparison and observing the factory's existing selected path corrected them.
The oracle independently reran all seven checks and approved this supported
scope. No corrupt fixture was created to claim negative-path verification.

Final supported full suites, with both acquired CEC2014 and SPSO references
provisioned, passed on Python3.12.14,3.13.15 and3.14.8: 3163 passed,272 skipped,
35 xfailed,one existing warning per environment. Setting the shared CEC root
also enables existing other-year checks: their pre-existing xfail declarations
run before missing-file checks. Those counts do not establish additional
golden-data coverage for other CEC years. The previous default3.14 run was
2934 passed,535 skipped,one xfail andone warning; provisioned-SPSO-only3.12/3.13
runs passed2938 with531 skips,one xfail andone warning.

Whole-repository Ruff lint/format (375 files), exact CI source ty gate and diff
whitespace checks passed. Installed core and CLI wheel checks passed on3.12
with minimum NumPy1.26.0/Matplotlib3.8.0 and3.13 with locked
NumPy2.4.2/Matplotlib3.10.8. They prove the helper is absent from core imports,
then explicitly import it and replay actual CEC2005/BBOB definitions outside
the checkout. Wheel SHA256:
88afdb211d9e7a9153dc170c86f4fb5cf89976e8623b910223a628f070250942.
Sdist SHA256:
f7b0354dee09939cc66febbba93a7d649fdbc3c99244535909c4c9623719341f.

V7.4.4 corrupt/incompatible-record verification and final coverage are pending
the unanswered named V7.0 authorization. At this checkpoint, V5 required-manifest
execution, V7.2 validation/report CLI and integrated V8 delivery work remained
unfinished; later supported results are recorded below.
The full implementation goal is active; all changes are local and uncommitted.

## V5 supported required-data producer and execution

The proposal now separates supported actual production work from the unchanged
named V5.2/V5.4 invalid-copy gates. The oracle approved this revision before
dispatch. Three initial availability assertions failed as intended; the actual
manifest producer, validator and stdlib runner now pass all three provisioned
checks. An observer delegates to real functions and sees450 scalar calls,
90 batch calls and450 batch rows, matching every report entry and count.

The validator independently requires90 capture files/450 case IDs and144
selected data paths:30 shifts,90 matrices and24 shuffles. The latter guard
was added after oracle static review found that validating only a supplied
inventory could omit required source data. No invalid inventory was generated.
All360 actual recorded data files are rehashed, along with available patched
source/patch/driver/executable and90 captures:454 files in total. All90
reconstructed input payload hashes match. Original source/archive hashes are
pinned setup identities, not falsely reported as rehashed absent originals.
Canonical manifest bytes preserve JSON types and keep approved deviations
independently empty. No compiler/reference executable runs during validation.

The actual external required report passed all450 scalar and batch-row
evaluations, zero failed/unavailable/deviation cases, zero required skips and
zero exit. Raw producer/report files are retained at
/tmp/pymofl-implementation/v5-supported-manifest.json and
/tmp/pymofl-implementation/v5-supported-report.json. Supported source-provisioned
full suites passed3166 tests on Python3.12.14,3.13.15 and3.14.8, with272 skips,
35 pre-existing source-root-enabled xfails andone existing warning each.
Whole Ruff lint/format (378 files), exact CI source ty gate and whitespace
checks passed. Installed core/CLI artifacts at3.12 minimum dependency floors
and3.13 locked versions passed real450-case validation outside the checkout;
the optional helper is absent from initial core import and explicitly imported
for the integration. Wheel SHA256:
5017566ec671a1826725069bec39aee24a356c5baabbaaca910259d25b123989.
The oracle independently reran the three supported checks and approved the
source/helper/runner scope. V5.2–V5.4 rejection sensitivity, final coverage and
the explicitly provisioned CI workflow remain unfinished; F07 is not closed.

## V7.2 supported optional validation CLI

Four initial command-availability assertions failed as intended. The optional
`pymofl validate` adapter now delegates to the V5 helper, consumes actual producer
manifests and writes strict JSON reports through exclusive file creation. Four
provisioned checks pass: availability, default summary, global JSON output and
quiet mode. The oracle independently reran and approved this supported scope.
No reference engine, compiler or network runs during validation.

The integrated source-provisioned suites pass 3170 tests on Python 3.12.14,
3.13.15 and 3.14.8, with 272 skips, 35 pre-existing source-root-enabled xfails and
one existing warning each. Installed core/CLI wheel checks pass outside the
checkout on 3.12 minimum NumPy 1.26.0/Matplotlib 3.8.0 and 3.13 locked
NumPy 2.4.2/Matplotlib 3.10.8. The actual installed CLI consumes a real producer
manifest and successfully validates all 450 cases. Latest tested wheel SHA256:
3d44143e9e402867d824aaf98957086aa97a7b24236980fb1f357ab00c277701.
Sdist SHA256:
9e39ff34a9aac151ae880b2ced7604c16f59b12a188b44c51986e71a3738d375.

The seven joint supported validation/CLI checks yield 81.99% statement and 65.62%
branch coverage for the reference helper, 84.21% statement and 100% branch
coverage for the CLI adapter. Exception edges are not counted in that branch
figure; neither result closes failure sensitivity or final coverage gates.
Reports/logs and the supported ATLAS handoff remain in
`/tmp/pymofl-implementation`. V7.2.4 failure/status/exit verification still
depends on the unanswered named generated-input authorization. F10 and the
full implementation goal remain incomplete; all changes are local/uncommitted.

## V8.1 supported documentation maintenance

The oracle approved the updated documentation and plan scope. README and docs
entry pages describe the selected working-tree additions and actual CLI
workflow. The roadmap distinguishes historical claims from current local
evidence. The SPSO handbook now matches the selected source definitions and
links their capture/provenance limitations. Descriptive guideline corrections
explain advancing noise state and quantization without bounds enforcement;
subclass, API and CI requirements are preserved.

The catalog's existing producer inspects the actual registry and constructor
signatures: 179 classes and 346 component aliases. Dimension labels identify
constructor defaults or absent metadata, not unsupported claims of fixed or
scalable dimensionality. The generated catalog includes the new selected SPSO
classes. No new vectors, invented reference values or numerical changes were
needed for this documentation slice.

Strict MkDocs build passed; existing literature link diagnostics remained at
their configured informational level. Whole-repository Ruff lint/format
(380 files), the exact CI source ty gate and both staged/unstaged whitespace
checks passed. The latest docs build/log are retained at
`/tmp/pymofl-implementation/site-v8-final` and `v8-mkdocs-final.log`.
V8.2 full verification remains dependent on the named failure/boundary,
coverage and required-reference CI gates. No commits or publication occurred.

## V7.4.3.1 deterministic replay breadth

The oracle approved the task and independently reran the implementation:
116 focused checks passed, comprising the seven initial cases and 109
source-derived deterministic requests. Each request creates a real exporter
record, strict-JSON round-trips it, reconstructs it and compares a second fresh
producer's full observations. The inventory is 23 CEC2005 entries at D10,
30 CEC2014 at D10, 24 GNBG at D30, 24 BBOB at D2/instance1 and eight selected
SPSO entries at native dimensions. Explicit exclusions are the actual noisy
CEC2005 F04 and F17 configurations; no unexpected deterministic failure was
skipped. Originals and source/reference outputs are unchanged.

With retained sources provisioned, 61 requests also replay unchanged own
CEC2005, CEC2014 or selected-source SPSO coordinates and preserve input arrays.
The remaining 48 BBOB/GNBG requests verify metadata only. Actual JUnit execution
properties record that distinction. This establishes replay at the named
dimensions/instance, not independent accuracy or every possible dimension.
Focused checks passed on Python 3.12.14, 3.13.15 and 3.14.8.

An initial full-suite command supplied the datasets directory rather than its
parent to the legacy `CEC_BENCHMARKS_PATH` variable. The actual golden-loader
check failed and extra references skipped. Correcting only the environment
path yielded 3279 passed, 272 skipped, 35 existing xfails and one existing
warning on each of the three interpreters. Failed and corrected logs remain
separate in `/tmp/pymofl-implementation`; only the corrected runs establish
the integration result. The other-year unavailable references and inherited
warning retain their earlier disclosed limits.

Whole-repository Ruff lint/format (380 files), the exact CI source ty gate and
staged/unstaged whitespace checks pass. No production source changed in this
extension, so the V7.2 installed-artifact evidence still describes the current
numerical/helper implementation. The supported ATLAS handoff is retained at
`/tmp/pymofl-implementation/atlas-v7431-supported.md`.
V7.4.4 incompatible/corrupt-record checks and full coverage remain gated.

## Remaining gates after supported-scope audit

The oracle approved the supported-scope audit and found no independently
dispatchable work remaining in the expanded plan. This does not approve full
implementation closure. All task-owned changes are staged for review; no
commits, pushes, publication, runtime dependencies or CEC2005 suite changes
were made.

| Remaining scope | Prerequisite |
| :--- | :--- |
| V5.2–V5.4 invalid-reference sensitivity and final required job | Unanswered V5.0 named invalid-copy authorization and its dependent verification |
| V7.1.4 bounded-helper failure/rejection checks | Unanswered V7.0 bounded-helper scope |
| V7.2.4 report status/exit failure checks | Unanswered V7.0 report/CLI scope |
| V7.3.4 source half-step/axis boundaries | Unanswered V7.0 source-boundary scope |
| V7.3.5 fixed-dimension guard checks | Separately requested, unanswered additional V7.0 fixed-dimension scope |
| V7.4.4 corrupt/incompatible definition records | Unanswered V7.0 definition-rejection scope |
| Full coverage, required-reference CI and V8.2 final delivery | Completed named prerequisites and independent integrated review |

The original accepted DEC01–DEC04 remain resolved. The later named fixture
requests do not reopen those decisions and remain pending; silence and oracle
approval do not grant generated-data authority under the plan's DP-05 policy.

## 2026-10-04 authorized failure and boundary verification

The owner explicitly approved synthetic test data. Deterministic stdlib,
NumPy and pytest fixtures cover the named V5.2–V5.4, V7.1.4, V7.2.4,
V7.3.4/V7.3.5 and V7.4.4 rejection/failure scopes. Initial versions were
Python 3.14.8, NumPy 2.4.2 and pytest 9.0.2. Genuine source/capture copies and
controlled real-evaluation wrappers isolate faults; originals, expected values
and tolerances remain unchanged. No adversarial observation is positive
reference evidence.

| Fact / evidence | Current integrated local result |
| :--- | :--- |
| F01–F02 / E01–E02 | Pass: genuine wheel outside checkout, clean core/CLI optionality, minimum and locked dependencies, all baseline aliases and explicit extensions |
| F03–F04 / E03–E04 | Pass: single invocation on controlled failure, exact exception propagation, protecting deterministic values/order/ownership/output buffers |
| F05 / E05 | Pass: canonical selectors, ambiguity and every supported list mutation resolve current contents |
| F06 / E06 | Pass: explicit/owned/legacy RNG modes, conflict rejection, state isolation and documented draw-order limits |
| F07 / E07 | Pass locally: fixed 450-case manifest; every scalar and batch case executes; malformed/missing provenance is unavailable, actual evaluation errors are failed; independent deviation policy remains empty |
| F08 / E08 | Pass for diagnostic scope: 48 original nine-trial workloads; discovery retained, universal min/weight rewrites rejected, Gallagher remains profiling-only |
| F09 / E09 | Pass: bounded deterministic helper, array/type/declaration/output guards, per-chunk ownership and exactly-once failures with documented partial writes |
| F10 / E10 | Pass: default/JSON/quiet optional CLI, honest counts/statuses, fresh-report failures and nonzero invalid required exits |
| F11 / E11 | Pass for selected native suites: original 20 controls and 546 source-generated boundary cases; fixed-dimension and quantization guards; retained 2007 Spring difference |
| F12 / E12 | Pass: 109 deterministic requests, 61 retained-coordinate/48 metadata-only replay cases, strict type-sensitive integrity and corrupt/incompatible-record rejection |

Requirement lifecycle remains the approved proposal's Proposed lifecycle;
these results bind implementation evidence without silently promoting public
requirements or asserting independently verified all-suite accuracy.

### Integrated verification and coverage

Full suites passed on Python 3.12.14, 3.13.15 and 3.14.8: **3,427 passed,
272 skipped, 35 xfailed, one existing warning** each. Full-suite NumPy is
2.4.2 and pytest is 9.0.2. Skips concern unprovisioned other datasets; inherited
xfails/warning are unchanged. Logs are `closure-matrix312.log`,
`closure-matrix313.log` and `closure-matrix314.log` under
`/tmp/pymofl-implementation`.

Whole-package Python 3.12 coverage is 93.64% statements, 79.78% branches and
90.82% combined, exceeding the original 92.62%/76.65%/89.47% baseline.
New runtime modules have 100% measured statements/branches. The oracle mapped
67 changed loader executable lines and all 32 changed branch arcs as covered.
[Coverage review](./evidence/coverage-review.md) records raw retained-tool
metrics and the exact nine defenses unreachable after immutable pin checks.
No new coverage exclusions or weakened numerical assertions were introduced.
The exact installed embedded smoke is separately measured at 52 statements
and ten branches, all covered; parent tracing is not substituted for children.

### Required reference workflow

The retained workflow `.github/workflows/reference-validation.yml` explicitly
fetches the immutable CEC2014 archive, verifies its SHA256, executes source
controls, produces a real manifest, runs all required scalar/batch cases and
rejects any selected integration skip/error/failure. Ordinary CI stays
hermetic. Local workflow equivalents use a fresh setup downloaded through the
exact pinned URL: 450 scalar and 450 batch executions, zero failures/deviations,
and 177 selected checks with zero skips on Python 3.12 and 3.13. Reports, JUnit
and logs reside in `ci-equivalent312` and `ci-equivalent313` under the external
evidence directory. Remote GitHub execution is unobserved; no commit/push was
authorized or performed.

### Tested runtime artifacts and limits

The tested runtime checkpoint was built with retained backend constraints.
Its wheel is identical to the final rebuilt wheel. Wheel SHA256 is
`533889b25745110ca6e01bf7a3ea118a707850b539c0aabc07fe64dd5ede6e06`;
the earlier checkpoint sdist SHA256 is
`34ba3a6b6a0ed800c3e15223ac0c2d5932ae04bcc99d3ee369a265804827e46d`.
Clean installed checks passed on Python 3.12 with NumPy 1.26.0/Matplotlib
3.8.0 and Python 3.13 with locked NumPy 2.4.2/Matplotlib 3.10.8, including
450-case helper and CLI execution. These dependency-floor runs are separate
from the full-suite matrix. Unconstrained secondary checker execution also
passed and records its actual resolved versions without updating the lock.

New fixture/tool evidence is local and reproducible through retained commands.
Source/data redistribution rights remain unspecified; archive/executable
outputs are not published. Other CEC years/COCO, arbitrary SPSO IDs, external
Network coordinates and broad numerical acceleration remain outside these
claims. No runtime dependency, CEC2005 suite configuration, commit or
publication was added. Final integrated review and current lint/type/docs
checks are recorded after their execution.

### Final common and documentation checks

Whole-repository Ruff lint/format passed for 382 files. The exact CI gate
`ty check src/pyMOFL/`, staged/unstaged whitespace checks and strict MkDocs
passed. Existing literature-link diagnostics remain informational. The default
hermetic Python 3.12 suite passed: 3,149 passed, 584 explicit optional-reference
skips, one inherited xfail and one inherited warning. This default result is
separate from the fully provisioned selected-source matrix.

A documentation-inclusive frozen build passed in `artifacts-closure`; its wheel
is byte-identical to the installed artifact above. Thus the existing clean
minimum/locked installed checks apply to that exact wheel. The earlier sdist
hash records the runtime checkpoint before final documentation reconciliation;
current sdist inventory/hashes are external frozen-build evidence, avoiding a
self-referential hash inside its own documentation. Current sdist includes the
new retained boundary setup and all task-owned reviewed files. Build/docs logs
are `closure-build.log` and `closure-mkdocs.log`; the ATLAS report is
`atlas-complete.md` in the external evidence directory.

### Final oracle disposition

On 2026-10-04 the oracle approved the complete selected local implementation,
including the reconciled proposal/documentation, runtime/tests/tools, manual
workflow, exact scoped coverage record and checked packaged bytes. No blocking
design, code, test or documentation findings remain. This approval covers the
local V0–V8 plan scope and does not certify remote CI, release, all-suite
scientific accuracy or deferred research features. All task-owned changes are
staged for review; no commit or push was performed.

## Version 0.4.0 delivery

The owner authorized committing and pushing the reviewed changes to `main`,
a version bump and tagging. Version 0.4.0 reflects the new optional capabilities
and changed numeric-string suite selector semantics. Project metadata, runtime
version and lockfile agree; the CLI version check follows the runtime version.
The earlier 0.3.0 artifact hashes and uncommitted/remote-unobserved statements
above describe their dated verification checkpoints. Versioned artifact and
remote delivery results are recorded separately after execution. No package-index
publication or broader feature/dependency work is included.

Release verification passed with version 0.4.0: the fully provisioned Python
3.14 suite reports 3,427 passed, 272 existing skips, 35 inherited xfails and one
existing warning. New clean core/CLI wheel checks pass on Python 3.12 with
minimum NumPy 1.26.0/Matplotlib 3.8.0 and Python 3.13 with locked NumPy
2.4.2/Matplotlib 3.10.8, including all 450 scalar/batch references and installed
CLI execution. Locked metadata checks, lint/format, the exact CI type gate,
strict documentation and wheel/sdist build pass. All 60 dependency versions
are unchanged. Versioned logs/artifacts are retained externally under
`/tmp/pymofl-implementation/release-040-*`.

## Version 0.4.1 reference workflow repair

RELEASE.1 (Test): GitHub rejected the initially pushed reference workflow and
manual dispatch with HTTP 422, identifying `runner.temp` in job-level `env`
as an unavailable context. The local equivalent tests verified execution logic
but did not validate GitHub's expression-context rules. This is retained failure
evidence, not a passed remote run.

RELEASE.2 (Implement): Move both reference path bindings into an initial runner
shell step that writes `RUNNER_TEMP` paths through `GITHUB_ENV`. Preserve all
source pins, actual capture/manifest/report execution, required counts and
zero-skip gates. Bump metadata/runtime/lock to 0.4.1 so the final tag includes
the fix without rewriting published v0.4.0. Remote syntax/dispatch and both
matrices must pass before final delivery; dependency versions remain unchanged.
