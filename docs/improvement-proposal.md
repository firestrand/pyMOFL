---
title: pyMOFL improvement and enhancement proposal
version: 1.4.2
last_updated: 2026-10-04
status: oracle-approved-local-implementation
owner: product-owner
reviewer: oracle-agent
reviewed_on: 2026-10-04
review_rounds: 8
tags: [architecture, modularity, performance, development-plan]
---

# pyMOFL improvement and enhancement proposal

**Guide Version:** Development Plan Creation Guide 2.7, 2026-09-27.

**Mode:** Vertical-Slice; each correction reaches the existing public library entry point.

**Plan Type:** Public Library, existing-system corrections and Technical Probes.

**Planning Horizon:** Rolling-Wave. V0–V8 are expanded and oracle reviewed, including the independent supported V5.1.1–V5.1.3 and V7.2.1–V7.2.3 sub-slices and V8.1 documentation maintenance. The owner approved the remaining generated failure/boundary tests on 2026-10-04; integrated local verification is recorded in implementation-progress.md. The owner accepted the recommended decisions on 2026-10-03; missing source/data evidence still gates dependent numerical work.

**Execution Strategy:** Sequential. One implementation writer owns shared registry, configuration, lockfile, and evidence changes. The oracle independently reviews this proposal; no implementation workers are authorized by this document.

**Execution Baseline:** `e2c56aa90598898cd148b850f18329c532591496`, package 0.3.0, branch `main`, repository root as working directory.

**PRD Trace:** LOCAL-R01–LOCAL-R08 below, derived from the owner's request to improve and review pyMOFL and propose enhancements. No external PRD was supplied; proposed contracts require product-owner acceptance before implementation.

**Real Data Policy:** Reuse named repository reference captures with documented provenance; acquire and approve missing external reference data before dependent validation.

**Generated Data Authorization:** Owner approved DEC-01 on 2026-10-03: controlled V2.1/V2.2 batch failure fixtures and identical V2.3 reruns; recorded seeded NumPy inputs for V4.1 RNG tests and V6.1 performance probes. Record the actual NumPy version, generator and seed at use. This does not authorize fabricated authoritative benchmark reference outputs or blanket future fixture generation.

**Remaining Generated-Input Gates:** Resolved on 2026-10-04 by the owner's explicit response, "Synthetic test data is approved." This authorizes the named V5.0/V7.0 test scopes, including the separately requested fixed-dimension checks and identical verification reruns. It does not replace official reference outputs or authorize new runtime dependencies. Commit/push/version/tag delivery was separately authorized on 2026-10-04.

**Provider Policy:** Keep optional CLIs, reference engines, report sinks, optimizer adapters, and acceleration libraries outside objective mathematics. Use ports only at actual replaceable external seams; NumPy arrays remain the established numerical API.

**Data and Provider Readiness:** Checked-in CEC 2005 captures are available. Pinned local CEC2014 captures are now verified as recorded in reference-validation.md; other external CEC golden datasets and COCO validation remain unavailable/unverified. Data or compatibility decisions block only their dependent slices.

**Slice Ordering Rationale:** Repair installed-package optionality first because it affects every user and requires no new numerical dataset. Correct exception dispatch next after controlled-fault evidence is authorized. Resolve suite and RNG semantics before changing them. Strengthen reference evidence before adopting numerical optimizations or adding suite claims.

**Fact Policy:** All newly proposed requirement facts start Proposed; oracle approval does not make them Active. Product acceptance controls semantics; implementation verification controls evidence results. A Fact Change requires product-owner acceptance; evidence maintenance preserves the claim and demonstrates equivalence.

**Outstanding Blockers:** No source/data authorization blocker remains in the selected implementation scope. Required CEC2014 execution, generated failure/boundary checks, coverage and local equivalents of the provisioned workflow are verified; final integrated oracle review approved the selected local implementation on 2026-10-04. Remote CI was unobserved at the local implementation checkpoint; commit/push/tag delivery is now authorized. Other-year CEC/COCO evidence and research-only features remain outside the selected scope. DEC-01–DEC-04 are resolved.

**Authority:** The owner's implementation request and 2026-10-03 acceptance authorize the selected corrections, four enhancements, and official reference acquisition. The owner subsequently authorized committing and pushing these changes to main, a version bump and tagging on 2026-10-04. Package-index publishing, new runtime dependencies and changes to `cec2005_suite.json` remain outside that scope. Oracle approval is independent design/code review, not evidence that functionality exists.

## Plan compliance matrix

| Invariant | Evidence in this proposal | Status | Dependent scope and resolution |
| :--- | :--- | :--- | :--- |
| Requirement traceability | LOCAL requirements, ledger, and task trace fields | Pass | Draft requirements remain Proposed |
| Modular SOLID, DRY, KISS design | Architecture invariants and feature boundaries | Pass | Preserve existing extension points; no new framework |
| Fact coverage | F01–F12 and E01–E12 | Pass | All proposed, no unsupported Active claims |
| Real-data provenance | DATA-01 provenance and pinned CEC2014/selected SPSO captures | Pass for selected scope; other references unverified | Selected V5/V7 execution uses approved captures and owner-authorized failure/boundary checks |
| No unauthorized synthetic evidence | V0.2.1 (DEC-01) and owner decision record | Pass | Named V2/V4/V6 evidence and remaining V5/V7 synthetic test scopes explicitly authorized |
| TDD and protecting evidence | Separate Test/Implement pairs; C01/C02 characterization bindings | Pass | V1.1 is the RED task for V1.3, protected by V1.2; V2.1 precedes V2.2 |
| Stable phases and coverage | Common gate and phase exit criteria | Pass | Observed integrated results are recorded separately in implementation-progress.md |
| Provider boundaries | Provider matrix and proposed architecture checks | Pass | Feature-specific checks expanded only after feature selection |
| Sequential execution readiness | Baseline check, dependencies, single writer, rollback | Pass for approved slices | V0–V8 local implementation and named gates verified; final integrated review approved 2026-10-04 |
| Performance claims | Measured review, limitations, V6 evidence requirements | Pass | No unmeasured speedup promised |
| Missing reference coverage | Selected DATA-02 acquired before required validation | Selected scope provisioned; local gates verified | Source/rights/environment/manifest pinned; missing other-year datasets confer no additional validation claim |
| Semantics and feature authority | Owner decision record for DEC-02–DEC-04 | Pass | Suite/RNG semantics and four features selected; current scoped evidence retained |

## Requirement fact ledger

The owner for every row is the product-owner capacity; lifecycle is Proposed. F01–F12 are Tier 1 because they describe proposed observable library or tooling contracts. These are acceptance proposals, not claims that current code satisfies them.

| Fact ID | Given / when / then statement | Applies when | Kind | Requirement | Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PYMOFL.INSTALL.CORE.v1 (F01) | Given a core-only installed artifact, when a user imports the library and loads a bundled benchmark, then optional CLI or documentation dependencies are unnecessary. | Declared supported environment; package data present | Behavior | LOCAL-R01 | E01 |
| PYMOFL.EXTENSIONS.DISCOVERY.v1 (F02) | Given existing built-in aliases and explicitly registered components, when automatic discovery or loading runs, then aliases remain usable and explicit registration and package scanning remain available. | Baseline supported registration paths | Compatibility | LOCAL-R01 | E02 |
| PYMOFL.BATCH.ERROR_ONCE.v1 (F03) | Given a component that fails during batch evaluation, when a composition invokes it, then the failure propagates without evaluating that component a second time. | Base functions and output transforms, with and without a valid output buffer | Behavior | LOCAL-R02 | E03 |
| PYMOFL.BATCH.PRESERVE.v1 (F04) | Given a valid deterministic evaluation and supported output buffer, when composition dispatch changes, then values, output identity, transform order, raw-input penalties, and caller input remain compatible. | Previously supported buffer/function combinations | Compatibility | LOCAL-R02 | E04 |
| PYMOFL.SUITE.LOOKUP.v1 (F05) | Given an owner-approved indexing and mutation contract, when suite contents are edited or queried, then canonical IDs and numeric selectors resolve consistently with that contract. | Approved mutation and selector semantics | Behavior | LOCAL-R03 | E05 |
| PYMOFL.NOISE.OWNERSHIP.v1 (F06) | Given an owner-approved RNG mode and seed contract, when noisy functions run, then RNG ownership, draw order, reproducibility limits, and compatibility are observable and documented. | Approved legacy/isolated modes only | Behavior | LOCAL-R04 | E06 |
| PYMOFL.VALIDATION.COMPLETE.v1 (F07) | Given a declared reference-validation manifest, when the required job runs, then every required case executes or fails explicitly, and documented deviations stay visible. | Provisioned pinned references; separate integration job | Reliability | LOCAL-R05 | E07 |
| PYMOFL.PERFORMANCE.EVIDENCE.v1 (F08) | Given approved workload inputs, when an optimization is considered, then the decision includes same-environment numerical, latency, variability, and memory evidence against a named baseline. | Performance candidates, not unrelated documentation edits | Reliability | LOCAL-R06 | E08 |
| PYMOFL.CHUNKING.DETERMINISTIC.v1 (F09) | Given a function explicitly declared deterministic and batch-independent, when approved chunked evaluation is requested, then row order and input ownership are preserved and working batch size is bounded. | Selected feature; supported dtype/shape/aliasing contract | Behavior | LOCAL-R07 / FEAT-01 | E09 |
| PYMOFL.VALIDATION.REPORT.v1 (F10) | Given a validation manifest and observed results, when a selected validation command emits a report, then unavailable, passed, failed, and deviation cases are distinguished without implying execution that did not occur. | Selected optional CLI feature | Behavior | LOCAL-R07 / FEAT-02 | E10 |
| PYMOFL.SPSO.DEFINITION.v1 (F11) | Given approved SPSO definitions and references, when a selected SPSO suite is loaded, then existing functions are composed with the specified bounds, quantization, penalties, and IDs. | Selected suite versions and reference-backed cases | Behavior | LOCAL-R07 / FEAT-03 | E11 |
| PYMOFL.MANIFEST.REPLAY.v1 (F12) | Given a selected reproducibility export contract, when a deterministic benchmark definition is exported and reconstructed, then its identity, parameters, data hashes, and supported environment metadata identify the same definition. | Replay of deterministic definitions; no promise of noisy state restoration | Behavior | LOCAL-R07 / FEAT-04 | E12 |

C01 and C02 are Proposed, Tier-2 Characterization checks rather than requirement-owned Tier-1 facts. C01 captures existing alias-to-class identities and explicit scan/registration signatures. C02 captures current valid deterministic composition values, supported output identity, order, and raw-input penalties. Neither may be promoted silently to a public guarantee or used to preserve the defects being corrected.

## Evidence index

The task descriptions preserve the approved execution sequence and historical gates. Current E01–E12 results and scoped limitations are recorded in [implementation evidence](./implementation-progress.md) and [coverage review](./evidence/coverage-review.md). The 2026-10-04 authorization resolves the named generated-input gates; those tests now execute in integrated local verification. Protect assertion helpers, fixture hashes, expected values, tolerances, marker configuration, and CI inclusion along with the test itself. Do not broaden tolerances, replace expected values with current output, or skip failures to pass a gate.

| ID | Facts / tier | Evidence path and literal command | Oracle and fixture/configuration dependencies |
| :--- | :--- | :--- | :--- |
| E01 | F01 / 1 | `scripts/verify_installed_package.py`; `uv run --locked python scripts/verify_installed_package.py --wheel /tmp/pymofl-artifacts/pymofl-0.4.0-py3-none-any.whl` | Two isolated installed environments outside checkout, core and CLI; DATA-01 packaged suite case; observed import/dependency presence |
| E02 | F02 / 1; C01 / 2 | `tests/test_registry.py`; `uv run --locked pytest tests/test_registry.py tests/test_api.py` | Alias snapshot derived from source and loaded class identities; explicit `scan_package(pkg_name)` and registration behavior; no fabricated plugin inputs |
| E03 | F03 / 1 | Extend existing `tests/core/test_composed_function.py`; `uv run --locked pytest tests/core/test_composed_function.py -k internal_type_error` | DEC-01 authorized controlled fault fixture; same failure object/cause and one invocation; base and output stages; DATA-01 input |
| E04 | F04 / 1; C02 / 2 | Existing `tests/core/test_composed_function.py`, `tests/core/test_batch_vectorization.py`, plus named DATA-01 cases; `uv run --locked pytest tests/core/test_composed_function.py tests/core/test_batch_vectorization.py` | DATA-01 independent reference outputs where applicable; protecting deterministic characterization; existing generated fixtures remain inherited baseline only |
| E05 | F05 / 1 | Extend existing `tests/test_api.py`; `uv run --locked pytest tests/test_api.py -k suite` | DEC-02 semantics; real constructed suite entries; complete approved mutation matrix |
| E06 | F06 / 1 | Existing noisy transform test modules plus `tests/functions/transformations/test_noise_ownership.py`; `uv run --locked pytest tests/functions/transformations/test_noise_ownership.py` | DEC-03 RNG mode; DEC-01 scope-bound seeded fixture authorization or approved captured streams; state isolation and documented draw ordering |
| E07 | F07 / 1 | `tests/test_reference_validation.py`, `scripts/verify_reference_manifest.py`; `PYMOFL_REFERENCE_CAPTURE_ROOT=/tmp/pymofl-implementation/retained-captures-final uv run --locked pytest tests/test_reference_validation.py`; `uv run --locked python scripts/verify_reference_manifest.py --capture-root /tmp/pymofl-implementation/retained-captures-final --manifest /tmp/pymofl-reference-manifest.json --report /tmp/pymofl-reference-report.json` | Actual producer manifest first; pinned CEC2014 source/hashes/90 files/450 cases. V5.2–V5.4 rejection, locally executed required-job and coverage evidence verified; remote CI and other CEC/COCO reference scopes remain unverified |
| E08 | F08 / 1 | `scripts/profile_workloads.py`, `docs/performance-review.md`, `docs/evidence/performance-v6.json`; `uv run --locked python scripts/profile_workloads.py --captures /tmp/pymofl-implementation/retained-captures-final --report /tmp/pymofl-v6-review.json --trials 9 --repeats 3` | Actual V6 diagnostic contract: 48 workloads, authorized PCG64 inputs and approved captures; raw trials, code/input hashes, CPU/BLAS and memory caveats. Fresh report path required; no wall-clock gate or adopted numerical optimization |
| E09 | F09 / 1 | `tests/test_evaluation.py`; `uv run --locked pytest tests/test_evaluation.py` | DEC-04 selected contract; unchanged DATA-01 rows; supported and V7.1.4 rejection/failure checks pass; helper has 100% statements/branches |
| E10 | F10 / 1 | `tests/cli/test_validation_report.py`; `PYMOFL_REFERENCE_CAPTURE_ROOT=/tmp/pymofl-implementation/retained-captures-final uv run --locked --extra cli pytest tests/cli/test_validation_report.py` | Actual DATA-02 producer manifests/reports and installed CLI success/JSON/quiet behavior; V7.2.4 failure/status/exit checks pass; helper/CLI adapter have 100% statements/branches |
| E11 | F11 / 1 | `tests/benchmark_suites/test_spso_validation.py`; `PYMOFL_SPSO_REFERENCE_PATH=/tmp/pymofl-implementation/spso-retained-captures uv run --locked pytest tests/benchmark_suites/test_spso_validation.py` | Approved official selected sources/captures; V7.3.4 source-backed half-step/axis checks (546 cases) and V7.3.5 guard checks pass; no repurposed CEC expectations as SPSO evidence |
| E12 | F12 / 1 | `tests/test_definition.py`; `PYMOFL_REFERENCE_CAPTURE_ROOT=/tmp/pymofl-implementation/retained-captures-final PYMOFL_SPSO_REFERENCE_PATH=/tmp/pymofl-implementation/spso-retained-captures uv run --locked pytest tests/test_definition.py` | Actual producer records and replay of 109 real deterministic requests at selected dimensions/instance; 61 retained-coordinate and 48 metadata-only breadth checks. Two explicit CEC2005 noise exclusions. V7.4.4 corrupt/incompatible records pass; definition has 100% statements/branches |

No pre-existing project-scoped fact register was found. A selected implementation establishes this ledger/index as the initial scoped register and checks any subsequently discovered inherited register before changing facts.

## Real data manifest and decisions

| ID | Source and access | Owner / approval | Sensitivity | Refresh and dependency |
| :--- | :--- | :--- | :--- | :--- |
| DATA-01 | Existing `tests/validation_data/cec/2005/f01.json` through `f25.json`; provenance described by the adjacent README | Product owner; existing checked-in test use, source-generation claims not independently revalidated | Public scientific vectors and expected outputs | Capture hashes before reuse; refresh if reference revision or suite data changes; V0, V1, V2, V6 |
| DATA-02 | Approved external CEC2014 captures selected through `CEC_BENCHMARKS_PATH`; other CEC captures and optional COCO engine remain unavailable/unverified | Owner authorized official-source acquisition/preparation; CEC2014 scope approved through V4.5; other sources require separate retained readiness evidence | Public scientific data; no credentials in fixtures | CEC2014 acquisition gate passed; required execution remains V5; refresh on suite/reference changes; V5 and selected FEAT-02 |
| DATA-03 | Actual installed wheel, package metadata, registry entries, and suite configurations at the baseline | Product owner; repository inspection within task scope | Public code/metadata | Refresh each integrated revision; V0–V3 |
| DEC-01 | Owner-approved named fixture/stream scope | Product-owner Human Decision in V0.2.1 | Name generator/fault mechanism, purpose, covered task IDs, seed/capture policy | V0.2.1 authorizes V2; V0.2.2 records V4.1 streams and V4.3 reruns; V6 authorization is recorded before its probe dispatch |
| DEC-02 | Numeric selectors and mutable suite behavior | Owner accepted canonical selectors/current mutable lookup on 2026-10-03 | No sensitive data | Current list contents and metadata determine lookup; ambiguous named matches raise ValueError; all list mutators remain supported |
| DEC-03 | Legacy global RNG versus explicit isolated opt-in mode; exact draw sequence requirements | Owner accepted default-preserving Generator injection on 2026-10-03 | No sensitive data | New default, seed algorithm, stream injection/state export are separate decisions; don't equate equal distributions with equal seeded results |
| DEC-04 | Select features, latency/memory budgets, acceptable tolerances and dependency changes | Owner selected FEAT-01–FEAT-04 on 2026-10-03; target budgets follow measurement | No sensitive data | Implement four selected features in separate slices; defer hard performance targets until measured; no new runtime dependency approval |

Unanswered future decisions remain blocked; elapsed time and oracle approval are not approvals for generated data, API breaks, or runtime dependencies. The explicit owner acceptance below resolves the named decisions only.

### Owner decision record: 2026-10-03

**Approver:** Trusted product owner. **Evidence:** Owner responded “I agree with all recommendations” to the five-decision walkthrough in this task.

| Decision | Accepted scope | Implementation/evidence boundary |
| :--- | :--- | :--- |
| DEC-01 | Controlled internal TypeError with call counters at base/output batch stages, buffer/no-buffer, V2.1/V2.2; identical V2.3 reruns. Recorded seeded NumPy RNG test inputs for V4.1 and performance inputs for V6.1 | pytest monkeypatch restores replaced callables; captured DATA-01 numerical inputs stay unchanged in V2. Record versions, generator and seeds for V4/V6; generated records are not authoritative references |
| DEC-02 | Numeric strings denote canonical function numbers; integers retain Python list indexing. Mutable lists remain mutable; name/number lookup uses current contents and rejects ambiguous matches | V3.1/V3.2/V3.3 cover all list mutation operations and mutable function metadata; no index-maintenance framework |
| DEC-03 | Preserve each transform's existing default RNG behavior; add explicit per-instance Generator injection | Legacy CEC NoiseTransform continues global RNG/seed behavior without injection; newer Gaussian/Uniform/Cauchy transforms retain their existing local default_rng(seed). Reject simultaneous seed and Generator. Document same-call-pattern replay and scalar/batch/chunk limits |
| DEC-04 | Select FEAT-01–FEAT-04, delivered separately. Measure latency and peak memory before numerical optimization, preserving correctness; choose hard targets from baselines | No arbitrary universal speed target; source-dependent validation/SPSO wait for references; FEAT-05–FEAT-07 remain research-only |
| DATA-02 acquisition | Acquire and prepare references from official benchmark implementations/technical reports; pin revisions, rights and capture procedures; investigate discrepancies before adoption | Authorization is for acquisition/preparation, not a claim that data already exists or matches. Core evaluation never downloads data or depends on a reference engine |

The four RNG implementations already differ: only the CEC NoiseTransform uses global np.random; the three COCO noise transforms already own default_rng streams. Preserving those existing defaults refines the walkthrough's general description without a migration.

## Provider boundary matrix

| Seam | Existing/proposed boundary | Domain exposure | Contract and replacement | Architectural check |
| :--- | :--- | :--- | :--- | :--- |
| NumPy numerical API | Existing functions/transforms accepting arrays | Intentional array types, float64 where current contract requires it | Preserve signatures and numerical contracts; do not add a generic array-provider abstraction | Core code must not acquire CLI, optimizer, reference-engine, or accelerator imports |
| Optional CLI | Existing `cli/` adapts public loading/evaluation | Typer/Rich remain in CLI presentation | Core import works without extras; CLI extra works installed | E01 core environment and inspection of automatic discovery roots |
| External references | Proposed test/validation adapter reuses existing golden loader | Function IDs, dimensions, input vectors, expected outputs, deviation status | Engine-specific types/exceptions handled by adapter; swapping oracle doesn't change objective mathematics | Selected validation implementation adds import-boundary tests forbidding reference-engine imports in core/functions/compositions |
| Result export | Proposed optional reporting helpers | Plain versioned report/manifest records | Start with local JSON; no storage framework or cloud client | Reporting must not be required by evaluation or imported by concrete objectives |
| Optimizer integration | Deferred companion/example, not a core optimizer engine | Existing evaluation API, optional objective callable | Optimizer-specific types stay in adapters/examples; core never imports PSO repos | If selected, architecture test forbids optimizer imports in numerical modules |
| Acceleration | Deferred optional adapter/extra | Preserve selected numerical API; explicitly report unsupported operations | Keep NumPy reference implementation; define capability and numerical tolerances before dependencies | Backend-specific imports isolated in selected adapter modules |

## Implementation phases

All paths and interfaces marked proposed are design targets. Every phase re-verifies the actual revision and predecessor evidence before work. One writer updates facts, assertions, fixture manifests, lockfiles, and wiring. Independent review assesses the integrated change where extension or numerical contracts change. Preserve repository branch/CI policy; phase closure is stage changes for human review, with commits/pushes governed by a later implementation instruction.

### Common verification and review gate

Run from the repository root with the intended supported interpreter and explicit extras:

```bash
uv sync --locked --extra dev --extra cli
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked ty check src/pyMOFL/
uv run --locked pytest
uv run --locked pytest --cov=pyMOFL --cov-branch --cov-report=json:/tmp/pymofl-coverage.json
git diff --check
```

Exercise the existing supported CI minors 3.12 and 3.13; the earlier review also checked 3.14 locally, which is additional evidence and does not replace 3.13. Record interpreter and `uv --version`. No lock regeneration during verification. Package checks additionally build wheel and source distribution and test the wheel outside checkout without development dependencies, first core-only and then CLI-enabled. A separate provisioned integration job runs required reference validation; default tests must not fetch live services or assert wall-clock timing.

Coverage uses the measured legacy baseline: 92.62% statement coverage, 76.65% branch coverage, and 89.47% combined coverage on Python 3.12. No regression on unchanged facts; require at least 90% branch coverage on changed executable behavior and at least 95% on changed domain logic, and justify an explicit scoped exception if unreachable branches prevent that threshold. No executable-code coverage threshold applies to documentation-only changes. New fact cases execute in their required jobs. Review values, assertion helpers, fixtures, tolerances, markers, and integration together; a green suite alone is insufficient.

### V0: Foundation — baseline, data, and contract inventory

**Role:** Foundation. **Dependencies:** None. **Execution baseline:** exact baseline plus task-authorized documentation only. **Integration/review:** implementation writer; product owner accepts requirements; oracle independently reviews the proposal.

**Target capability:** Preserve the existing library and establish evidence before editing it. The walking skeleton already exists in `load()`, `FunctionFactory`, `ComposedFunction`, and `tests/test_api.py`; don't rebuild it.

**Facts enabled/protected:** F01–F08; characterization C01/C02. **Verification:** common gate. **Demo:** `uv run --locked pytest tests/test_api.py`.

**Observable outcome:** Named baseline, declared data approvals, and unchanged numerical/extension behavior; no production code added.

**Rollback:** Remove only task-owned inventory/evidence changes, preserving user work. **Exit:** record results and unresolved decisions, preserve green default suite, stage changes for human review.

#### Task V0.1: Verify the baseline and extension surface

**Type:** Verify.

**PRD Trace:** LOCAL-R01, LOCAL-R02, LOCAL-R08; technical enabler for preserving existing library contracts.

**Fact / Evidence:** C01/C02, Tier 2 Characterization, Proposed; E02/E04 protecting bindings, results recorded separately from Tier-1 acceptance.

**Expected Failure Signature:** N/A — verification of existing behavior; do not force RED.

**Makes Green:** N/A.

**Real Data Dependency:** DATA-01, DATA-03; unchanged inherited randomized tests only for baseline, not new facts.

**Provider Boundary:** Existing numerical API and optional CLI.

**Work Packet:** Sequential.

**Write Scope:** Proposed focused characterization additions in `tests/test_registry.py` and existing `tests/core/test_composed_function.py`; baseline logs outside tracked source.

**Inputs / Contracts:** Current source, `pyproject.toml`, existing tests; alias identity and valid composition contract only.

**Validation / Handoff:** Run common gate and the E02/E04 commands after their evidence exists; record baseline revisions, skip reasons, references and artifact versions. Stop if current state differs materially from the recorded baseline.

**Depends On:** None.

**Facts Protected:** C01/C02; do not encode known retry/import/lookup defects as required behavior.

**Description:** Inspect registered aliases, explicit scanning, composition order, supported buffer paths, scalar/batch behavior, raw penalties, and input ownership. Reuse real captured inputs; an unavailable characterization case becomes an explicit data gate rather than an invented fixture.

**Acceptance Criteria:** Characterization results and data sources recorded; no production change; every subsequent refactor can cite protecting evidence; default suite remains green. Verify `.gitignore` and repository setup already exist rather than reinitializing Git.

#### Task V0.2: Verify reference provenance and inventory unresolved decisions

**Type:** Verify.

**PRD Trace:** LOCAL-R03–LOCAL-R07; DP-05 and DP-09.

**Fact / Evidence:** DATA-01 provenance/hashes; availability of DATA-02; proposed F05–F12 evidence stays Unknown.

**Expected Failure Signature:** N/A — verify actual availability; unavailable data is an honest blocker.

**Makes Green:** N/A.

**Real Data Dependency:** DATA-01–DATA-03; DEC-01–DEC-04 remain human-owned.

**Provider Boundary:** External reference acquisition and optional features.

**Work Packet:** Sequential.

**Write Scope:** This proposal's data/decision records and named evidence manifests; no new generators or changed constants.

**Inputs / Contracts:** Actual fixture README, fixture hashes, owners' explicit answers, suite/RNG current behavior.

**Validation / Handoff:** Record exactly what is approved or unavailable; product owner is the decision authority. V2 waits for controlled-fault evidence authorization; V3/V4/V5/V7 wait for their own contracts/data. V0.2.1 separately records DEC-01 authorization. Later rolling-wave revisions expand DEC-02–DEC-04 and acquisition tasks before their dependent implementation.

**Depends On:** V0.1.

**Facts Protected:** No existing fact/fixture is weakened to remove a blocker.

**Description:** Confirm checked-in captures and inventory missing external references. DEC-01 must name controlled fault mechanism, purposes, task coverage V2.1/V2.2, and cleanup, through V0.2.1; any seeded performance/noise authorization needs separate named tasks in a subsequent revision.

**Acceptance Criteria:** Provenance limitations explicit; unanswered decisions not treated as accepted; dependency gates target only affected slices; all fixtures public/minimized with no credentials or production records.

#### Task V0.2.1: Human Decision — authorize controlled batch-failure evidence

**Type:** Human Decision.

**PRD Trace:** LOCAL-R02; DP-05 controlled generated-fixture authorization.

**Fact / Evidence:** F03/F04, Tier 1 → E03/E04; authorization does not make evidence Green.

**Expected Failure Signature:** N/A — this task records a human decision rather than running a test.

**Makes Green:** N/A.

**Real Data Dependency:** DATA-01 real captured input; proposed behavior-only fault fixture authorized only by this task, with no fabricated representative numeric or error-response payload.

**Provider Boundary:** N/A — controlled in-process function/transform failure, not a new external provider.

**Work Packet:** Sequential.

**Write Scope:** DEC-01 record in this proposal and evidence metadata; no production implementation or generated fixture before approval.

**Inputs / Contracts:** V0.2 inventory and the proposed E03 contract; product-owner explicit answer required.

**Validation / Handoff:** Record Approve, Reject, or Unanswered, owner capacity, timestamp, mechanism/version, scope and exact covered task IDs. Reject/Unanswered leaves V2 blocked; V1 and independent documentation can proceed. No response is not approval.

**Depends On:** V0.2.

**Facts Protected:** C02 and existing captured inputs; approval cannot authorize numerical fixture replacement or broader RNG/performance data generation.

**Question / Options:** May V2.1/V2.2 use a controlled internal failure on a real captured input to prove single invocation? Approve the named mechanism, or reject and retain the proposed correction as blocked until suitable approved failure evidence is available.

**Scope Requested:** Simulate an internal `TypeError` and track calls at the base and output-transform batch stages, with and without a currently valid output buffer; no realistic invented error payload is asserted.

**Generator / Mechanism:** Locked pytest 9.0.2 `monkeypatch` or equivalent task-local callable replacement, an invocation counter and a behavior-only exception instance. Re-confirm the exact framework version at execution; use DATA-01 input values unchanged.

**Determinism:** No PRNG, seed, random output or numeric generator. The mechanism deterministically raises on the named call and restores every replaced callable after the test.

**Covered Tasks:** V2.1 creates evidence; V2.2 consumes that evidence. V2.3 may rerun the identical registered E03/E04 checks but may not add generated cases under this authorization.

**Facts Affected:** F03/F04 (Tier 1); C02 (protecting Tier 2).

**Owner / Approver:** Product-owner capacity, exercised by the trusted human; the oracle and implementation writer cannot approve.

**Default if Unanswered:** Not authorized. The owner answered Approve on 2026-10-03 for the exact scope below.

**Why Real Data Cannot Serve:** Existing benchmark reference captures contain successful evaluations, not controlled internal failure at both composition stages; the fixture simulates failure behavior, without substituting generated benchmark data for missing external references.

**Authorization Record:** Approved by the trusted product owner on 2026-10-03 through acceptance of the decision walkthrough. Exact mechanism and covered V2.1/V2.2/V2.3 tasks are stated above; numerical input remains DATA-01.

**Description:** Separate controlled-fault fixture authorization from provenance verification. Once authorized, register each fixture with V0.2.1 provenance at the point of use; any generated fixture file uses a clearly generated path, not the real-reference directory.

**Acceptance Criteria:** Explicit human decision and exact authorized mechanism/tasks recorded, or narrow Blocked status retained; no blanket future generation approval; default suite and original data untouched.

#### Task V0.2.2: Human Decision record — seeded RNG evidence

**Type:** Human Decision record (already answered; no new approval requested).

**PRD Trace:** LOCAL-R04. **Fact / Evidence:** F06/E06. **Expected Failure Signature:** N/A. **Makes Green:** N/A.

**Real Data Dependency:** DATA-01 and Generated (authorized by this task): seeded NumPy streams.

**Provider Boundary:** In-process noise transforms, not external numerical references. **Work Packet:** sequential. **Write Scope:** decision/provenance record only. **Depends On:** owner acceptance on 2026-10-03. **Facts Protected:** existing RNG defaults, F03/F04.

**Inputs / Contracts:** Owner approved recorded seeded NumPy inputs for RNG testing in the five-decision walkthrough. This records that acceptance; it does not request it again.

**Scope Requested / Accepted:** V4.1 creates stream ownership/default/sequence evidence; V4.2 consumes exactly that evidence; V4.3 reruns identical evidence only. New unrelated fixture cases are excluded.

**Generator / Mechanism:** NumPy Generator(PCG64(20261003)) for explicit/local modes; legacy RandomState(MT19937) seeded 20261003 for default compatibility checks. The original approved wording covers seeded NumPy testing of each existing mode; naming both algorithms avoids silently changing legacy noise. Record installed NumPy version. Test values reuse DATA-01 captured outputs; expected draws come from the independently instantiated same algorithm/seed.

**Determinism:** Same algorithm/seed/call pattern. Restore process-global np.random state in finally/pytest fixture cleanup. No claim of scalar/batch/chunk invariance for multi-draw transforms.

**Owner / Approver:** Trusted product owner; accepted all five recommendations on 2026-10-03. **Authorization Record:** Approved for exactly the above named consumers. **Default if Unanswered:** Not authorized; this decision is answered.

**Description:** Externalize the accepted test authorization and stream metadata before V4 dispatch; never label generated values as authoritative external captures.

**Validation / Handoff:** Verify exact pytest/NumPy versions and provenance at use. **Acceptance Criteria:** named stream/task coverage recorded; legacy state restored; no false benchmark-reference claims.

### V1: Walking skeleton — installed core package with preserved extensions

**Role:** Capability, using the existing load/evaluate path. **Dependencies:** V0.1/V0.2; owner acceptance of LOCAL-R01. **Execution baseline:** verified V0 output. **Integration/review:** one writer; independent review of alias inventory and installed artifacts.

**Target capability:** Core-only installed import and bundled-function loading without optional dependencies.

**Facts introduced:** F01/F02. **Facts protected:** C01/C02 and current public evaluation API.

**Verification:** common gate, E01/E02. **Demo:** E01 command against the built wheel.

**Observable outcome:** Core-only and CLI-enabled installed environments both work with the same aliases.

**Rollback:** Restore discovery and packaging changes as one reviewed slice; no data migration. **Exit:** protecting evidence green, artifact checks pass outside checkout, changed-code coverage policy met, stage changes for human review.

#### Task V1.1: Test core-only installation

**Type:** Test.

**PRD Trace:** LOCAL-R01.

**Fact / Evidence:** F01, Tier 1 → E01; installed import and real bundled suite load.

**Expected Failure Signature:** Core-only installed `import pyMOFL` fails because automatic scanning reaches optional `pyMOFL.cli` and imports missing `typer`; this is the intended packaging defect, not missing test tooling.

**Makes Green:** N/A.

**Real Data Dependency:** DATA-01 f01 captured 10D optimum and expected value; DATA-03 built artifact.

**Provider Boundary:** Installed artifact versus optional CLI dependencies.

**Work Packet:** Sequential.

**Write Scope:** Proposed `scripts/verify_installed_package.py`; focused installed-artifact checks, not default pytest network setup.

**Inputs / Contracts:** Built wheel and declared metadata; core and CLI extras tested separately; subprocesses run outside checkout with no source PYTHONPATH leakage.

**Validation / Handoff:** `uv build --no-sources --out-dir /tmp/pymofl-artifacts`, then E01 command. Provision/install dependencies as explicit setup outside the default test suite; script records versions, creates unique temporary environments, and cleans them on failure and success.

**Depends On:** V0.1, V0.2.

**Facts Protected:** C01/C02.

**Description:** Implement an installed-artifact checker that imports the package, resolves built-in aliases, loads CEC 2005 F1 from packaged data, and verifies its captured optimum. Also run CLI smoke checks in the CLI environment; don't fix the defect in this task.

**Acceptance Criteria:** The core import defect is demonstrably RED at the baseline; wheel path matches actual package metadata; no optional packages accidentally supplied to the core environment; resource cleanup and bounded subprocess timeouts work; evidence index records Expected Red.

#### Task V1.2: Verify explicit registration and discovery compatibility

**Type:** Verify.

**PRD Trace:** LOCAL-R01.

**Fact / Evidence:** F02, Tier 1 → E02; C01, Tier 2 characterization.

**Expected Failure Signature:** N/A — existing supported extension behavior may already pass.

**Makes Green:** N/A.

**Real Data Dependency:** DATA-03 actual built-in aliases and explicit scans of existing component packages; additional third-party fixture capture needs a scoped approval if unavailable.

**Provider Boundary:** Registry extension seam; optional packages remain separate.

**Work Packet:** Sequential.

**Write Scope:** Proposed `tests/test_registry.py` and focused existing API tests.

**Inputs / Contracts:** Current `register`, `get`, `scan_package(pkg_name)` and `FunctionRegistry.register_base` behavior/signatures; current dynamically replaced components remain allowed.

**Validation / Handoff:** Run E02, comparing named aliases to actual class identities before and after explicit scans. Stop on alias collisions requiring a changed public contract rather than choosing a new precedence silently.

**Depends On:** V1.1.

**Facts Protected:** C01, C02.

**Description:** Pin supported discovery paths without inventing a plugin framework. Verify import order doesn't remove aliases and already-imported explicit components remain available to loading.

**Acceptance Criteria:** Full baseline alias inventory preserved; generic explicit scanning is retained; failures are not swallowed; tests state the extension cases actually covered rather than claiming all third-party ecosystems are tested.

#### Task V1.3: Implement bounded automatic discovery

**Type:** Implement.

**PRD Trace:** LOCAL-R01.

**Fact / Evidence:** N/A — implementation binds F01/F02 to E01/E02.

**Expected Failure Signature:** N/A.

**Makes Green:** E01; preserve E02.

**Real Data Dependency:** DATA-01, DATA-03.

**Provider Boundary:** Automatic component discovery versus explicit package scans and optional CLI/docs imports.

**Work Packet:** Sequential.

**Write Scope:** `src/pyMOFL/__init__.py`, `src/pyMOFL/registry.py`, `src/pyMOFL/loader.py`; touch `factories/function_factory.py` only if alias protection demonstrates necessity.

**Inputs / Contracts:** V1.1/V1.2 evidence; existing public generic scanner and mutable registry. Scope the automatic roots using the actual decorated modules; preserve explicit calls.

**Validation / Handoff:** Run E01/E02 and common gate; compare integrated alias identities and installed core/CLI behavior.

**Depends On:** V1.1, V1.2.

**Facts Protected:** F02, C01/C02; no evaluation signature/formula changes.

**Description:** Restrict library-owned automatic discovery to registration-bearing modules without importing optional presentation packages. Consider one-time built-in discovery only where explicit registration still updates lookup and explicit scans retain behavior; keep locks/cache mechanisms minimal and based on evidence.

**Acceptance Criteria:** E01 green, E02 preserved, no catch-all suppression of import failures, no fabricated static alias inventory, no new runtime dependency or import-time network/process operation. Preserve the generic explicit scanner rather than silently changing `scan_package(pkg_name)` semantics.

#### Task V1.4: Verify delivery and fact sufficiency

**Type:** Fact Sufficiency Review and Verify.

**PRD Trace:** LOCAL-R01, LOCAL-R08.

**Fact / Evidence:** F01/F02, Tier 1 → E01/E02; protecting C01/C02.

**Expected Failure Signature:** N/A — verify completed integrated behavior.

**Makes Green:** N/A.

**Real Data Dependency:** DATA-01, DATA-03; declared dependency lock and supported range.

**Provider Boundary:** Installed package/core/CLI and CI setup.

**Work Packet:** Sequential.

**Write Scope:** `.github/workflows/ci.yml` and the artifact verification script; evidence updates; no unrelated runtime migration or API expansion.

**Inputs / Contracts:** V1.3 integrated revision; local and CI checks use identical commands with explicit extras and locked verification. CI actions resolve to reviewed full 40-hex commit SHAs with readable version comments; setup-uv pins the same reviewed uv release recorded locally (review environment used 0.12.22; revalidate before choosing it). Preserve read-only token permissions and unprivileged PR jobs.

**Validation / Handoff:** Run common gate, wheel/sdist build, E01/E02, supported 3.12/3.13 checks, and a recorded lower-bound dependency environment compatible with metadata. Build-backend constraints/version recorded separately from uv runtime lock; no claim `uv build --no-sources` alone pins build dependencies.

**Depends On:** V1.3.

**Facts Protected:** F01/F02, C01/C02.

**Description:** Add installed-artifact CI checks and lock freshness enforcement without replacing established tooling. Resolve action identities and full SHAs from official repositories during implementation, review their provenance, pin a reviewed uv release, and arrange update proposals through existing maintenance tooling; do not invent SHAs or automatically choose a newer major release. In an isolated local patch, reintroduce optional-package scanning and confirm E01 fails; undo only that task-owned sensitivity mutation. Review the complete diff and whether the tests could catch a lost alias or source-checkout import leak.

**Acceptance Criteria:** Supported matrix and artifact gates green; lower-bound or unsatisfied-range evidence reported honestly; every adopted action uses a reviewed full SHA and version comment; uv release matches the recorded local pin; read-only CI permissions preserved; no orphan RED evidence or stale-lock implicit updates; no agent-approved exception to action pinning; minimal focused diff reviewed; stage changes for human review.

### V2: Capability — batch failures propagate once

**Role:** Capability. **Dependencies:** V1; DEC-01 authorization recorded by V0.2.1 for V2.1/V2.2 controlled-fault evidence. **Execution baseline:** verified V1 output. **Integration/review:** one writer; independent review of extension substitutability and exception behavior.

**Target capability:** Safe dispatch for existing batch methods with mixed output-buffer support.

**Facts introduced:** F03/F04. **Facts protected:** F01/F02, C02. **Verification:** common gate and E03/E04. **Demo:** `uv run --locked pytest tests/core/test_composed_function.py -k internal_type_error`.

**Observable outcome:** Internal failures are propagated after one call; deterministic values, buffers, penalties, and extension compatibility remain valid.

**Rollback:** Restore dispatch helper and callers together; no numerical formula or serialized-data migration. **Exit:** DEC-01 satisfied, E03/E04 and integrated gate green, evidence sensitivity verified, stage changes for human review.

#### Task V2.1: Test internal failures and characterize supported buffers

**Type:** Test, with protecting Verify cases.

**PRD Trace:** LOCAL-R02.

**Fact / Evidence:** F03/F04, Tier 1 → E03/E04; C02 characterization.

**Expected Failure Signature:** A deliberately authorized internal `TypeError` at the base or scalar-transform stage results in two invocations at baseline, violating the single-invocation assertion; the failure must originate inside evaluation rather than from argument binding.

**Makes Green:** N/A.

**Real Data Dependency:** DATA-01 inputs; Generated (authorized by Task V0.2.1) for the named behavior-only fault fixture only after its record is approved. Any additional artificial edge-case data requires a separate authorization.

**Provider Boundary:** Existing function/transform methods are in-process extension seams, not providers requiring a new framework.

**Work Packet:** Sequential.

**Write Scope:** Existing `tests/core/test_composed_function.py`; named real-data cases added to `tests/core/test_batch_vectorization.py` as necessary.

**Inputs / Contracts:** Valid `out` support or no `out`, exception identity/cause, one invocation, current mutable component assignments, existing deterministic order and raw-input penalty semantics.

**Validation / Handoff:** Run E03 and E04. Record expected RED only for named E03 assertions; protecting cases remain green. Unknown buffer shape/dtype/aliasing behavior is inventoried, not silently hardened in this slice.

**Depends On:** V1.4, V0.2.1; DEC-01 approval is required, not inferred from completing provenance verification.

**Facts Protected:** F01/F02 and C02; preserve default `OptimizationFunction.evaluate_batch` row-loop fallback and subclasses without `out`.

**Description:** Cover base and output-stage internal failures with and without a valid buffer, then cover supported callable signatures and component replacement. Use targeted fault injection solely as explicitly authorized contract evidence; don't introduce representative fabricated benchmark inputs.

**Acceptance Criteria:** Intended RED is observed; fault cases clean up patches; protecting valid cases pass; source-derived expectations and independent captured outputs distinguish characterization from mathematical correctness.

#### Task V2.2: Implement single-invocation batch dispatch

**Type:** Implement.

**PRD Trace:** LOCAL-R02.

**Fact / Evidence:** N/A — implementation serves F03/F04.

**Expected Failure Signature:** N/A.

**Makes Green:** E03; preserve E04.

**Real Data Dependency:** DATA-01; Generated (authorized by Task V0.2.1) for consumption of the named V2.1 fault evidence only.

**Provider Boundary:** Existing numerical function and scalar-transform API.

**Work Packet:** Sequential.

**Write Scope:** `src/pyMOFL/functions/transformations/composed.py`; one small private invocation helper at the narrowest shared location only if used by both dispatch sites.

**Inputs / Contracts:** V2.1 contracts; callable keyword support, wrapper/custom methods, component replacement, and fallback methods without `out`.

**Validation / Handoff:** Run E03/E04 and common gate. Review accepted keywords before execution; a child exception must never be interpreted as capability negotiation.

**Depends On:** V2.1.

**Facts Protected:** F01/F02/F04 and C02.

**Description:** Prefer a small helper that selects a supported invocation before calling once, then copies into a supplied output buffer only when necessary. Do not add `out` to every subclass or require a new interface from third-party components. Any capability cache must honor replacement of publicly mutable components and callables; avoid caching if its complexity is unjustified.

**Acceptance Criteria:** E03 green; no blanket evaluation `TypeError` retry; existing bufferless methods remain substitutable; exception causes preserved; transform/penalty order and ownership unchanged; no new dtype coercion or invalid-buffer policy outside the approved contract.

#### Task V2.3: Verify the integrated dispatch contract

**Type:** Fact Sufficiency Review and Verify.

**PRD Trace:** LOCAL-R02, LOCAL-R08.

**Fact / Evidence:** F03/F04, Tier 1 → E03/E04; F01/F02 regression checks.

**Expected Failure Signature:** N/A — verification of completed behavior.

**Makes Green:** N/A.

**Real Data Dependency:** DATA-01, DATA-03; rerun only Generated (authorized by Task V0.2.1) E03/E04 evidence created in V2.1, with no additional generated inputs.

**Provider Boundary:** In-process extension compatibility; no new external provider.

**Work Packet:** Sequential.

**Write Scope:** Evidence bindings and focused documentation; no unrelated objective edits.

**Inputs / Contracts:** Actual V2.2 integrated diff and complete protecting fact set.

**Validation / Handoff:** Common gate, E01–E04. Isolated reintroduction of broad retry must fail E03; revert only that sensitivity mutation. Record callable cases not covered and block incompatible generalizations.

**Depends On:** V2.2.

**Facts Protected:** F01–F04 and C01/C02.

**Description:** Verify single-call errors and valid evaluation behavior through the public composed API, not merely a private helper. Review integrated changes for open/closed and substitutability requirements.

**Acceptance Criteria:** Gate and sensitivity check pass; no numerical formula change; default tests green; changed-domain coverage threshold met; stage changes for human review.

### V3: Live canonical suite lookup

**Role:** Capability.

**Dependencies:** V1 and accepted DEC-02.

**Baseline:** staged V1 plus current tests.

**Review:** sequential writer and independent oracle.

**Facts:** introduce F05/E05; protect F01/F02, integer/slice list behavior.

**Provider Boundary:** existing BenchmarkSuite public API, no providers.

**Real Data Dependency:** actual configured suite functions/IDs.

**Demo:** `uv run --locked pytest tests/test_api.py -k suite`.

**Rollback:** revert only V3 lookup/tests/docs; numeric-string migration must be documented.

**Exit:** mutation/selector evidence and common gate green; changed behavior coverage at policy; reviewed and staged.

#### Task V3.1: Test canonical selectors and current mutable contents

**Description:** Exercise canonical selectors and current mutable contents before the production change.

**Type:** Test.

**PRD Trace:** LOCAL-R03.

**Fact / Evidence:** F05/E05, Tier 1; result Unknown until run.

**Expected Failure Signature:** `suite['1']` maps to a different function after reorder, stale lookup returns removed entries, or duplicate canonical IDs silently resolve.

**Makes Green:** N/A.

**Depends On:** V1.4, accepted DEC-02.

**Write Scope:** existing `tests/test_api.py`.

**Work Packet:** sequential.

**Real Data Dependency:** actual BBOB/CEC function instances/IDs; no fabricated numerical data.

**Inputs / Contracts:** canonical numbers independent of position; integers/slices unchanged; case-insensitive exact IDs/names and short codes; no match KeyError; ambiguous match ValueError (get() must not disguise ambiguity); metadata edits visible.

**Provider Boundary:** BenchmarkSuite.

**Facts Protected:** F01/F02.

**Validation / Handoff:** run E05 before implementation, retain RED; cover append/extend/insert/remove/pop/clear, item/slice assignment/deletion, reverse/sort, += and *=.

**Acceptance Criteria:** sensitivity to stale indexes and ambiguity demonstrated; tests restore no shared suite state.

#### Task V3.2: Implement lookup against current entries

**Description:** Implement lookup against current entries. Implement only the task’s named public contract and evidence.

**Type:** Implement.

**PRD Trace:** LOCAL-R03.

**Fact / Evidence:** F05/E05.

**Expected Failure Signature:** V3.1 RED.

**Makes Green:** E05.

**Depends On:** V3.1.

**Write Scope:** `loader.py::BenchmarkSuite` and migration documentation.

**Work Packet:** sequential.

**Real Data Dependency:** DATA-03 actual suites.

**Provider Boundary:** public mutable list API.

**Inputs / Contracts:** inspect current function_id and name at lookup; normalize numeric strings to canonical short code, never list position; reject more than one matching entry even if the same object occurs twice. Exact canonical full IDs must not accidentally match unrelated full IDs through their common short code. No parallel mutable index or list mutator overrides.

**Facts Protected:** F01/F02 and Python list semantics.

**Validation / Handoff:** E05 and full API tests.

**Acceptance Criteria:** all mutation cases pass; new behavior documented; no factory/config/numerical change.

#### Task V3.3: Verify integrated suite behavior

**Description:** Verify integrated suite behavior. Implement only the task’s named public contract and evidence.

**Type:** Verify.

**PRD Trace:** LOCAL-R03/LOCAL-R08.

**Fact / Evidence:** F05/E05; protect F01/F02.

**Expected Failure Signature:** N/A.

**Makes Green:** N/A.

**Depends On:** V3.2.

**Write Scope:** evidence record and scoped docs.

**Work Packet:** sequential.

**Real Data Dependency:** unchanged actual suite entries and default suite.

**Provider Boundary:** public load/get_suite and optionality.

**Inputs / Contracts:** V3.1 matrix plus existing loading API.

**Validation / Handoff:** common gate, changed branch coverage, oracle review of ambiguity and mutable metadata.

**Acceptance Criteria:** evidence/results retained, phase green and staged.

### V4: Explicit noise stream ownership

**Role:** Capability.

**Dependencies:** V2 and V3.3, accepted DEC-01/DEC-03.

**Baseline:** integrated V3.

**Review:** sequential writer and independent oracle.

**Facts:** introduce F06/E06; protect F01–F04 and existing formulas/sequences.

**Provider Boundary:** existing noise transforms and TransformBuilder.

**Real Data Dependency:** approved V4.1 seeded PCG64 streams, actual captured objective values; record NumPy and seed (20261003).

**Demo:** E06 command.

**Rollback:** remove new opt-in keyword/wiring/tests only; retain all old defaults.

**Exit:** common gate, coverage policy, seed/ownership evidence, docs, independent review and staging.

#### Task V4.1: Test legacy defaults and explicit stream injection

**Description:** Exercise legacy defaults and explicit stream injection before the production change.

**Type:** Test.

**PRD Trace:** LOCAL-R04.

**Fact / Evidence:** F06/E06.

**Expected Failure Signature:** rng keyword rejected, explicit stream not used, or wrong seed/rng conflict behavior.

**Makes Green:** N/A.

**Depends On:** V2.3, V3.3, V0.2.2 and accepted decisions.

**Write Scope:** proposed `tests/functions/transformations/test_noise_ownership.py` plus existing factory tests if needed.

**Work Packet:** sequential.

**Real Data Dependency:** DATA-01 and Generated (authorized by Task V0.2.2): named seeded NumPy streams. Restore global state after legacy checks.

**Provider Boundary:** four actual transforms and TransformBuilder.

**Inputs / Contracts:** no global-state change under injection; supplied Generator identity and state progression; separate equal-seed streams replay same call pattern; seed+rng raises ValueError before reseeding; defaults preserve old global/local modes. Scalar and batch formulas remain exact; Uniform/Cauchy have documented block draw ordering, not chunk/scalar invariance.

**Facts Protected:** F03/F04 and legacy noise behavior.

**Validation / Handoff:** E06 intended RED and existing four noise modules protecting checks.

**Acceptance Criteria:** precise generators/seeds/source values recorded, no statistical/flaky assertions or new reference claims.

#### Task V4.2: Add keyword-only Generator injection

**Description:** Add keyword-only Generator injection. Implement only the task’s named public contract and evidence.

**Type:** Implement.

**PRD Trace:** LOCAL-R04.

**Fact / Evidence:** F06/E06.

**Expected Failure Signature:** V4.1 RED.

**Makes Green:** E06.

**Depends On:** V4.1.

**Write Scope:** existing noise.py, gaussian_noise.py, uniform_noise.py, cauchy_noise.py, TransformBuilder and scoped API docs.

**Work Packet:** sequential.

**Real Data Dependency:** DATA-01 and Generated (authorized by Task V0.2.2), identical V4.1 evidence.

**Provider Boundary:** transform construction seam.

**Inputs / Contracts:** keyword-only rng; reject seed+rng; NoiseTransform uses injected standard_normal while legacy path retains randn and global seeding; other transforms keep default_rng(seed) unless supplied rng. Forward rng through programmatic builder params, without serializing live Generator objects or introducing a global provider.

**Facts Protected:** F01–F04, existing RNG defaults.

**Validation / Handoff:** E06 and existing noise/factory/composition checks.

**Acceptance Criteria:** no formula/order/dtype rewrite, optionality preserved, no stream/state replay promises beyond approved contract.

#### Task V4.3: Verify replay boundaries and ownership documentation

**Description:** Verify replay boundaries and ownership documentation. Implement only the task’s named public contract and evidence.

**Type:** Verify.

**PRD Trace:** LOCAL-R04/LOCAL-R08.

**Fact / Evidence:** F06/E06; protect F01–F05.

**Expected Failure Signature:** N/A.

**Makes Green:** N/A.

**Depends On:** V4.2.

**Write Scope:** evidence/docs.

**Work Packet:** sequential.

**Real Data Dependency:** DATA-01 and Generated (authorized by Task V0.2.2): identical V4.1 evidence, no new cases.

**Provider Boundary:** public transform/factory APIs.

**Inputs / Contracts:** shared Generator consumption is caller-owned; no thread-safety or seed-only noisy-state replay guarantee.

**Validation / Handoff:** common gate, changed-domain coverage, oracle review of sequence and global-state compatibility.

**Acceptance Criteria:** same-call-pattern replay verified, limits documented, phase green and staged.

### V4.5: Official reference acquisition and compatibility gate

**Role:** Data Gate. **Baseline:** integrated V4. **Review:** sequential writer and independent oracle. **Authority:** recorded DATA-02 acquisition approval. **Facts:** enable F07/E07 and F10/E10; protect F01–F06 and all existing expected values. **Exit:** pinned provenance, locally accessible captures, investigated deviations, independent review. No benchmark formula or production validation changes in this phase.

The first required integration manifest will cover CEC2014 F1–F30 at D10, D30 and D50, five captured inputs per function/dimension. These dimensions are supported by the official archive and existing validation tests. This is declared coverage, not a claim of comprehensive validation of every bundled suite. Other downloaded snapshots remain source inventory until separately verified; CEC2013's documented buffer behavior and the other suites are not silently promoted to trusted captures.

#### Task V4.5.1: Inventory official source and capture contracts

**Type:** Data Acquisition. **PRD Trace:** LOCAL-R05. **Fact / Evidence:** N/A; prerequisites for E07. **Expected Failure Signature:** N/A. **Makes Green:** N/A. **Depends On:** V4.3 and DATA-02 acquisition approval. **Work Packet:** Sequential. **Write Scope:** task-owned external source snapshots and provenance documentation. **Real Data Dependency:** official P-N-Suganthan CEC2013, CEC2014, CEC2017-BoundContrained, 2020-Bound-Constrained-Opt-Benchmark and 2022-SO-BO repositories; existing DATA-01 F1 inputs. **Provider Boundary:** external reference engines remain outside numerical core.

**Inputs / Contracts:** pin Git revision, archive and member SHA256, compiler/version/flags, source rights metadata and local-use/redistribution limits. Retain original source; record every Linux compatibility patch separately. Inspect input-file paths, globals and dangerous operations before execution. No license file is presently observed in the selected CEC archives: record rights as unspecified, do not redistribute their source or claim an open-source license. Capture inputs are official driver's shift and zero vectors plus unchanged DATA-01 F1 random, lower and upper rows; no new random reference inputs. The existing reference-loader contract is datasets/CEC2014/func_N_DD/golden.jsonl with func_id/dim/case/x/value fields.

**Facts Protected:** F01–F06, existing captures and formulas. **Description:** Identify and pin official reference inputs and a portable local capture procedure before execution. **Validation / Handoff:** verify revision/archive hashes, inspect driver and evaluator source, record rights and exact patch diff in docs/reference-validation.md. **Acceptance Criteria:** source provenance and 90 function/dimension identifiers are explicit; reference sources stay outside the repository; no unsupported rights or equivalence claims.

#### Task V4.5.2: Capture and cross-check CEC2014 outputs

**Type:** Data Acquisition. **PRD Trace:** LOCAL-R05. **Fact / Evidence:** N/A; supplies E07 oracle data. **Expected Failure Signature:** N/A. **Makes Green:** N/A. **Depends On:** V4.5.1. **Work Packet:** Sequential. **Write Scope:** task-owned external capture harness, datasets and capture manifest; provenance documentation. **Real Data Dependency:** pinned official CEC2014 archive, its shift/rotation/shuffle files, DATA-01 F1 D10/D30/D50 rows. **Provider Boundary:** subprocess C++ executable; no import, download or reference-engine linkage in core.

**Inputs / Contracts:** use a bounded local executable, no fast-math flags; verify each requisite data file exists and source/patch hashes match. Record removal of unused WINDOWS.H and correction of double fscanf format %Lf to %lf as portability patches, preserving formulas. Confirm published F1 shift optimum/bias and source data agreement with bundled constants before using captures. Emit five named cases per function/dimension and per-file hashes; repeat capture byte-for-byte. Investigate discrepancies against pyMOFL and retain actual observations without changing outputs, tolerances or formulas. If source behavior or provenance remains unresolved, keep the affected captures unapproved and gate dependent optimization.

**Facts Protected:** F01–F06 and documented deviations. **Description:** Obtain actual official-engine outputs with complete input provenance and verify parser compatibility. **Validation / Handoff:** load all 450 captured records through load_golden_cases; CEC_BENCHMARKS_PATH=/tmp/pymofl-implementation/official-captures uv run --locked pytest tests/utils/test_golden_loader.py tests/benchmark_suites/test_cec2014_validation.py; record passing cases and any numerical disagreements separately. **Acceptance Criteria:** repeatable captured files, 450 actual records, parser round-trip and reference controls verified; discrepancies remain visible and causally reviewed, never converted to passing by weakening existing assertions.

#### Task V4.5.3: Review reference readiness

**Type:** Verify. **PRD Trace:** LOCAL-R05/LOCAL-R08. **Fact / Evidence:** F07/E07 readiness only, Tier 1; required execution behavior remains V5 work. **Expected Failure Signature:** N/A — data verification. **Makes Green:** N/A. **Depends On:** V4.5.2. **Work Packet:** Sequential. **Write Scope:** docs/reference-validation.md, docs/implementation-progress.md and task-owned retained provenance metadata. **Real Data Dependency:** actual V4.5.2 captures/source/patch hashes and observed test results. **Provider Boundary:** independent oracle review; no new numerical API.

**Inputs / Contracts:** record source rights as observed, manifest coverage and unavailable suites, environment, source/config/input/capture hashes, control checks, discrepancies and exact reproduction commands. External captures are local verification data, not automatically redistributable package assets. **Facts Protected:** F01–F06 and all historical reference claims. **Description:** Decide readiness from retained evidence before reference-sensitive work; oracle approval does not imply comprehensive suite accuracy. **Validation / Handoff:** oracle inspects capture procedure and discrepancy disposition; default gate remains hermetic. **Acceptance Criteria:** an approved pinned required-case manifest can be consumed by V5, or specific unresolved cases remain explicitly gated; no numerical optimization dispatch against unapproved oracle cases.

### V5: Required reference manifest execution

**Role:** Capability. **Baseline:** integrated V4 and approved V4.5 captures. **Facts:** F07/E07, protecting F01–F06. **Review:** sequential writer and independent oracle. **Exit:** all 450 declared CEC2014 records execute scalar/batch or fail explicitly; no required skip; reproducible setup, coverage and common gate pass. Other suites remain unavailable/unverified, not passed.

The required manifest pins source identity, input provenance, 90 function/dimension IDs and five case names per ID. Explicit setup records actual platform/compiler and generated capture hashes. Validation checks the pinned source contract and those hashes before consuming records; it does not falsely promise cross-platform bit-identical libm outputs. The reference engine runs only in explicit setup, with controls on every platform, never on import or ordinary validation.

#### Task V5.0: Human Decision — invalid reference-copy authorization

**Type:** Human Decision. **PRD Trace:** LOCAL-R05. **Fact / Evidence:** F07/E07. **Expected Failure Signature:** N/A. **Makes Green:** N/A. **Depends On:** V4.5.3. **Work Packet:** Sequential. **Write Scope:** authorization record only. **Real Data Dependency:** actual DATA-01 and V4.5 records; copies do not replace real reference authority. **Provider Boundary:** pytest temporary directories.

**Decision Needed:** authorize V5.2/V5.4 failure-test copies? **Why real data cannot serve:** approved captures are valid; unchanged records do not exercise deliberate duplicate/invalid/nonfinite/corrupt provenance rejection. **Scope Requested:** missing/duplicate cases, altered function IDs/dimensions/shapes, NaN/Infinity, incorrect source/file hashes only. **Generator:** deterministic Python stdlib/NumPy/pytest edits of named actual records; record installed versions. **Determinism:** no random generation; exact mutation names/inputs retained. **Covered Tasks:** V5.2 and identical V5.4 reruns/sensitivity. **Facts Affected:** F07/E07, Tier 1. **Owner/Approver:** product owner. **Default if unanswered:** Not authorized; dependent mutations stay pending.

**Inputs / Contracts:** original approved files remain unchanged; temporary invalid copies are explicitly labeled adversarial/nonrepresentative and cleaned up by pytest. No invented outputs count as valid numerical evidence. **Facts Protected:** F01–F06 and capture provenance. **Description:** Required scope-bound DP-05 authorization, requested through asynchronous user input on 2026-10-03. **Validation / Handoff:** record actual response before dependent fixture creation. **Acceptance Criteria:** explicit accepted scope and covered IDs, or independent unchanged-data work only.

#### Task V5.1: Retain reproducible external capture setup

**Type:** Evidence Maintenance. **PRD Trace:** LOCAL-R05. **Fact / Evidence:** F07/E07 prerequisite only, Unknown. **Expected Failure Signature:** N/A. **Makes Green:** N/A. **Depends On:** V4.5.3. **Work Packet:** Sequential. **Write Scope:** proposed scripts/capture_cec2014.py and reference documentation. **Real Data Dependency:** approved actual V4.5 source/archive and DATA-01 inputs. **Provider Boundary:** explicit subprocess setup from local archive to external directory, no implicit network.

**Inputs / Contracts:** retain reviewed patches/driver/flags with argument paths, pinned archive SHA256, safe extraction, bounded subprocesses, complete data-file length/shape checks, input/source/patch/data/output hashes, compiler/platform record, repeated capture and F1 controls. Never commit external sources, vectors or executable. **Facts Protected:** F01–F06 and reference identity. **Description:** Retain the verified procedure after temporary evidence expires. **Validation / Handoff:** reproduce 450 records in a fresh external directory; compare all capture hashes and loader round-trips; missing/bad archive fails before compilation. **Acceptance Criteria:** identical local approved records and metadata, no numerical or import behavior change, no unsupported rights claims.

#### Task V5.2: Verify authorized required-data rejection cases

**Dispatch Gate:** Task V5.0 accepted on 2026-10-04. Supported production creation/execution is separately verified through V5.1.1–V5.1.3 below. This task retains its original named generated-copy authorization scope.

**Type:** Verify. **PRD Trace:** LOCAL-R05. **Fact / Evidence:** F07, Tier 1 → E07. **Expected Failure Signature:** N/A; any observed defect requires a separately recorded Test/Implement pair before repair. Existing optional success on absent required captures remains historical baseline evidence, never the required-entry result. **Makes Green:** N/A. **Depends On:** V5.1.3 and accepted V5.0. **Work Packet:** Sequential. **Write Scope:** tests/test_reference_validation.py and required-runner tests. **Real Data Dependency:** approved actual V4.5 records and only the named authorized adversarial copies. No fabricated successful numerical values or new random inputs. **Provider Boundary:** optional public helper and required script.

**Inputs / Contracts:** versioned manifest, source identity and file hashes; missing file/case, duplicate case, wrong metadata/shape/nonfinite data, bad hash or wrong source fail. Real scalar and batch evaluation execute each valid case at declared tolerance. Statuses distinguish unavailable/failed/passed/deviation. A declared deviation requires a retained reason and matching ID, still executes and never suppresses malformed/missing data or operational errors. Sensitivity uses omission/metadata edits of real records; default tests neither fetch nor compile references. **Facts Protected:** F01–F06, formulas, original tolerances. **Description:** Specify completeness and honest observed results. **Validation / Handoff:** uv run --locked pytest tests/test_reference_validation.py; record original missing-required successful optional run separately from subsequent required-entry assertions. **Acceptance Criteria:** requirement-owned completeness assertions, source/hash provenance and status sensitivity, no import failure represented as behavioral RED.

#### Task V5.3: Verify required manifest helper against authorized failure evidence

**Type:** Verify. **PRD Trace:** LOCAL-R05. **Fact / Evidence:** F07/E07. **Expected Failure Signature:** N/A; an observed defect requires its own Test/Implement pair before repair. **Makes Green:** N/A. **Depends On:** V5.2 and V5.1.3. **Work Packet:** Sequential. **Write Scope:** tests/evidence and narrowly required repairs after a separately recorded pair. **Real Data Dependency:** approved V4.5 captures/provenance and DATA-01 protection. **Provider Boundary:** plain versioned results, NumPy/public loading; reference engines and CLI dependencies stay outside.

**Inputs / Contracts:** consume local manifest/capture root, verify source/hash/coverage and execute all valid scalar/batch cases through public loading. Parsing/provenance/report records stay at this helper boundary, never in objectives. Required script exits zero only for observed required success or explicit documented numerical deviations; unavailable/invalid/error/unexpected differences exit nonzero. Retain existing CEC2014 tolerance. No implicit download, timing gate, plugin framework, new dependency or sensitive raw-input logging. **Facts Protected:** F01–F06 and numerical contracts. **Description:** Enforce required execution and create the narrow result boundary reusable by FEAT-02. **Validation / Handoff:** focused E07 and actual 450-record script integration; inspect actual scalar/batch execution counts. **Acceptance Criteria:** E07 green, missing required data nonzero, source/hash failures never deviations, core optionality and existing math preserved.

#### Task V5.4: Verify required integration and import boundaries

**Type:** Verify. **PRD Trace:** LOCAL-R05/LOCAL-R08. **Fact / Evidence:** F07, Tier 1 → E07. **Expected Failure Signature:** N/A. **Makes Green:** N/A. **Depends On:** V5.3 and accepted V5.0. **Work Packet:** Sequential. **Write Scope:** separate explicit integration workflow, import-boundary assertions and evidence/docs. **Real Data Dependency:** V5.1 setup and approved actual references. **Provider Boundary:** explicitly provisioned integration job; default CI remains hermetic.

**Inputs / Contracts:** separate workflow retrieves pinned source, runs retained setup controls, required checker and existing CEC2014 tests. Remote CI remains unobserved until run. Core/functions/compositions cannot import reference engines, CLI or capture setup. Required absence fails even when old optional tests skip. **Facts Protected:** F01–F06. **Description:** Verify completeness, packaging boundaries and limitations independently. **Validation / Handoff:** supported-minor common gate, changed-domain coverage, artifact checks, oracle review; required missing-case sensitivity on task-owned actual capture copy. **Acceptance Criteria:** 450 observed scalar/batch results, zero required skips, no newly declared CEC2014 deviations, gates pass; F07 becomes Pass only from integrated evidence.

### V5.1 supported production sub-slice (oracle-approved)

The original V5.2/V5.4 adversarial scope retains its IDs; the owner resolved its authorization on 2026-10-04 and current results are recorded above.
The following tasks build the actual manifest/report producer and execute
unchanged approved records independently. This split does not close F07 or
authorize generated failure fixtures.

The optional `pyMOFL.reference_validation` module will expose
`prepare_reference_manifest(capture_root)` and
`validate_reference_manifest(manifest, *, capture_root)`. The producer reads
the retained setup's actual provenance and files, checks pinned CEC2014
source/revision/archive/original/patch/driver and DATA01 identities, and emits
a version-1 strict JSON manifest. Required coverage is independently fixed
at F1–F30 × D10/30/50 × shift/zeros/random/bounds_min/bounds_max, never inferred
from whatever files are present. It includes actual capture and provenance
hashes and setup compiler/platform identities. The approved CEC2014 deviation
policy is independently empty; arbitrary supplied reasons cannot create an
accepted deviation. Any future deviation policy requires its own reviewed
scope. No fabricated manifest fixture.

Validation checks transport/schema, pinned source and actual hashes before
numerical execution. Original/archive hashes remain recorded setup identities
when their bytes are absent; available patched source/patch/driver/executable,
data/captures and reconstructed input hashes are actually checked. Paths must remain confined to the explicitly supplied
capture root; supplied provenance paths are data, not arbitrary file access.
Every valid case is evaluated once through the public scalar path and once
in its function/dimension batch. Retain the existing strict tolerance
`abs(actual-expected) < 1e-6 * max(1, abs(expected))` for each path. Nonfinite
or malformed inputs/results and operational errors cannot become deviations.
All declared cases receive passed/failed/unavailable/deviation status with
scalar/batch execution flags; required success requires every case observed
passed or an explicitly reasoned numerical deviation. Missing/invalid data
and unexpected exceptions yield nonzero required outcome, never skipped/pass.

The report schema version1 records selected suite/source/provenance identity,
current software identity, manifest hash, required case count, actual scalar/
batch counts, status counts and canonical per-case IDs/reasons/difference
metrics. It omits raw coordinate vectors. It records retained dataset hashes
without claiming the external source is the latest release or hashes prove
authorship. `scripts/verify_reference_manifest.py --capture-root ROOT
--manifest FILE --report FILE` is the stdlib entry point, with no CLI dependency
or implicit download/compiler run. Package functions/compositions/core never
import this optional helper, the CLI or capture setup.

#### Task V5.1.1: Test actual manifest creation and valid-data execution

**Type:** Test. **PRD Trace:** LOCAL-R05. **Fact / Evidence:** F07/E07 supported scope only. **Expected Failure Signature:** helper availability/producer capability assertion fails; imports are setup, not RED. **Makes Green:** N/A. **Depends On:** V5.1 and oracle approval of this contract. **Work Packet:** Sequential. **Write Scope:** tests/test_reference_validation.py. **Real Data Dependency:** approved retained V4.5 CEC2014 provenance and450 unchanged records, provisioned explicitly through PYMOFL_REFERENCE_CAPTURE_ROOT; no random inputs, corrupt copies or invented successful outputs. Actual new manifests/reports come from the validated production create path. **Provider Boundary:** optional helper, existing loader, strict JSON and required script.

**Inputs / Contracts:** independently required90 files/450 cases, pinned provenance, actual produced schema/hash metadata, scalar/batch counts and numerical agreement. Default availability check is hermetic; source integration explicitly skips only when not provisioned in ordinary tests, never in the required script. **Facts Protected:** F01–F06, approved capture identities/tolerances. **Description:** Establish real producer/replay/report evidence. **Validation / Handoff:** availability RED then focused source-provisioned check and external script run. **Acceptance Criteria:** 450 scalar and450 batch results, explicit IDs/statuses and no required skips; no negative-path/final coverage claim.

#### Task V5.1.2: Implement optional required-data producer and validator

**Type:** Implement. **PRD Trace:** LOCAL-R05. **Fact / Evidence:** F07/E07 supported scope. **Expected Failure Signature:** V5.1.1 intended RED. **Makes Green:** supported producer/execution assertions. **Depends On:** V5.1.1. **Work Packet:** Sequential. **Write Scope:** src/pyMOFL/reference_validation.py, scripts/verify_reference_manifest.py and reference documentation; no objective formulas, capture edits or new dependency. **Real Data Dependency:** same actual approved captures, producer-generated manifest/report and static source pins already evidenced in V4.5. **Provider Boundary:** pathlib/JSON/hash, NumPy and public loading only.

**Inputs / Contracts:** the above fixed coverage/schema/source/strict tolerance/report contracts; no engine imports, downloads or class deserialization. Guards are implemented now but generated rejection verification stays V5.2/V5.4. **Facts Protected:** F01–F06 and original datasets. **Description:** Add the library-neutral result boundary FEAT02 reuses. **Validation / Handoff:** focused valid-data checks and required real450-case script execution. **Acceptance Criteria:** V5.1.1 green, actual observed reports; no required data silently skipped, no broader failure/coverage closure claimed.

#### Task V5.1.3: Verify supported integration and import boundaries

**Type:** Verify. **PRD Trace:** LOCAL-R05/LOCAL-R08. **Fact / Evidence:** F07/E07 supported scope only. **Expected Failure Signature:** N/A. **Makes Green:** N/A. **Depends On:** V5.1.2. **Work Packet:** Sequential. **Write Scope:** artifact smoke, evidence/reference documentation and ATLAS handoff. **Real Data Dependency:** same approved captures and actual generated reports on supported minors. **Provider Boundary:** independent oracle and installed core/CLI environments.

**Inputs / Contracts:** source/hash/coverage metadata and exact actual counts, optional import boundaries, retained producer commands. **Facts Protected:** F01–F06. **Description:** Independently establish supported capability without promoting untested failure guarantees. **Validation / Handoff:** common gate, real450-case integration, artifact optionality, oracle and ATLAS. **Acceptance Criteria:** no required skipped cases,450 scalar/450 batch executions, valid JSON report and supported approvals; final F07 closure remains V5.2–V5.4.

### V6: Performance evidence and candidate dispositions

**Role:** Technical Probe. **Baseline:** integrated V4/V4.5; V5 is required only before reference-sensitive adoption. **Facts:** F08/E08, protect F01–F07 as applicable. **Review:** sequential writer/oracle. **Exit:** same-environment latency/variance/memory/numerical evidence and explicit accept/reject/block disposition per candidate. A probe does not silently adopt its numerical patch.

#### Task V6.0: Record accepted seeded performance input scope

**Type:** Human Decision (accepted 2026-10-03 owner decision record). **PRD Trace:** LOCAL-R06. **Fact / Evidence:** F08/E08 prerequisite. **Expected Failure Signature:** N/A. **Makes Green:** N/A. **Depends On:** accepted DEC-01/DEC-04. **Work Packet:** Sequential. **Write Scope:** provenance/authorization only. **Real Data Dependency:** owner-authorized seeded NumPy performance inputs. **Provider Boundary:** standalone probe.

**Inputs / Contracts:** Generator(PCG64(20261003)), installed NumPy version recorded; finite uniform vectors within actual workload bounds. Sizes 1/4/100/1000/10000, D10/30/50 where supported. These are explicitly modeled diagnostic workloads, not claimed production distributions or authoritative reference outputs. **Facts Protected:** existing actual captures. **Description:** Bind already accepted V6.1 performance generation to a stable authorization task, including generator/seed/version/size/range at use. **Validation / Handoff:** retain input hashes and generation parameters with reports. **Acceptance Criteria:** V6.1 creates inputs; V6.2 consumes them; V6.3 performs identical reruns only. No new adversarial numerical fixtures or broader generation granted.

#### Task V6.1: Capture workload baselines

**Type:** Probe. **PRD Trace:** LOCAL-R06. **Fact / Evidence:** F08, Tier 1 → E08. **Expected Failure Signature:** N/A — measurement, no speed target. **Makes Green:** N/A. **Depends On:** V4.3, V6.0; V4.5.3 for CEC2014 controls. **Work Packet:** Sequential. **Write Scope:** proposed scripts/profile_workloads.py and performance evidence; no production math. **Real Data Dependency:** actual DATA-01/V4.5 inputs plus Generated (authorized by Task V6.0) finite diagnostic populations. **Provider Boundary:** NumPy/public loading and standalone subprocess measurement.

**Inputs / Contracts:** measure repeated discovery, GNBG F24 MinComposition, CEC2014 F23 weighted composition and BBOB F21 Gallagher; scalar/small/large workloads separately. Record revision, interpreter/NumPy/BLAS/threads, warmups/repeats, trial values/median/range, tracemalloc peaks separately from fresh-process maximum RSS. Construction excluded from numerical timings. Retain inputs or exact generation/hash, no timing threshold in tests. **Facts Protected:** F01–F06 and official CEC2014 controls. **Description:** Measure hot paths and memory growth before candidates. **Validation / Handoff:** same script/environment emits raw measurements; verify actual inputs/outputs immutable. **Acceptance Criteria:** workload limitations and measurement methods explicit, no unmeasured speed claims or production-distribution claims.

#### Task V6.2: Probe isolated candidates and choose dispositions

**Type:** Probe. **PRD Trace:** LOCAL-R06. **Fact / Evidence:** F08/E08. **Expected Failure Signature:** N/A. **Makes Green:** N/A. **Depends On:** V6.1. **Work Packet:** Sequential. **Write Scope:** task-owned isolated candidate code/reports, no production math. **Real Data Dependency:** exactly retained V6.1 inputs; approved CEC2014 reference controls. **Provider Boundary:** temporary numerical candidates, existing public evaluation.

**Inputs / Contracts:** compare running-minimum and streamed weighted distances without dot-product cancellation rewrite. Alternate baseline/candidate trials, compare values and input ownership; collect tracked allocation and RSS distinctions. Discovery V1 evidence is integrated, not another speculative rewrite. Gallagher profiling identifies hotspots only; absent COCO/exceptional-value references gates adoption. Default target: reject a candidate lacking a consistent practical benefit; lower allocations with slower latency motivate FEAT-01 rather than a faster-default claim. **Facts Protected:** formulas, reduction order, flags and all accepted behavior. **Description:** Make individual measured accept/reject/block decisions. **Validation / Handoff:** raw before/after values and trial distributions; oracle reviews disposition and numerical limits. **Acceptance Criteria:** no production numerical adoption without a separate protected Test/Implement pair and V5 evidence; all candidates explicitly disposed.

#### Task V6.3: Verify performance report and retained probe

**Type:** Verify. **PRD Trace:** LOCAL-R06/LOCAL-R08. **Fact / Evidence:** F08, Tier 1 → E08. **Expected Failure Signature:** N/A. **Makes Green:** N/A. **Depends On:** V6.2. **Work Packet:** Sequential. **Write Scope:** performance report/script documentation and implementation-progress.md. **Real Data Dependency:** unchanged actual/authorized V6.1 workloads and reports; identical reruns authorized by V6.0. **Provider Boundary:** standalone tool, no competitor runtime dependencies.

**Inputs / Contracts:** reproducible command, schema/metadata, candidate decisions and limitations; tests never enforce wall-clock thresholds. No unrelated competitor imports or warning suppression. **Facts Protected:** F01–F07 as applicable. **Description:** Retain observable evidence and independently review conclusions. **Validation / Handoff:** oracle plus proportional tool checks; common gate if production files change. **Acceptance Criteria:** reports distinguish tracked allocations/RSS and model/reference results, independent review approves supported claims.

### V7.0: Scope-bound remaining failure/boundary test authorization

**Type:** Human Decision. **PRD Trace:** LOCAL-R07. **Fact / Evidence:** F09–F12/E09–E12, Tier 1. **Expected Failure Signature:** N/A. **Makes Green:** N/A. **Depends On:** selected FEAT-01–FEAT-04. **Work Packet:** Sequential. **Write Scope:** authorization record only. **Real Data Dependency:** actual DATA-01, source-backed SPSO records and records produced through new validated report/manifest APIs. **Provider Boundary:** pytest temporary copies and public library/CLI entry points.

**Decision Needed:** authorize the exact checks below? **Why real data cannot serve:** unchanged valid data does not exercise deliberate invalid request, tie, corrupt-record and controlled-failure classes. **Generator:** deterministic Python/NumPy/pytest operations, installed versions recorded; no random generation or fabricated valid reference outputs. **Determinism:** named literal operation/case retained, originals unchanged and pytest cleanup. **Owner/Approver:** product owner. **Default if unanswered:** Not authorized; positive unchanged-data work proceeds independently.

| Purpose | Exact generation scope | Covered tasks |
| :--- | :--- | :--- |
| Bounded helper rejection/failure | chunk_size zero/negative/nonintegral/bool; false deterministic/batch-independent declarations; invalid input/output/result shape/dtype, readonly or overlapping output elements using real values; controlled child TypeError/counter at a later chunk to observe propagation and partial writes | V7.1.4 and identical verification reruns |
| Validation report/CLI failure | temporary invalid copies with missing/duplicate case, wrong ID/dimension/shape, NaN/Infinity, changed source/file hash, unsupported schema; verify honest statuses and exit codes | V7.2.4 and identical verification reruns |
| SPSO quantization/axis boundary | deterministic half-step neighbors/ties and zero-axis combinations derived from pinned source bounds/steps; actual official engine supplies expected outputs, not pyMOFL | V7.3.4 and identical verification reruns |
| Fixed-dimension factory rejection (separate additional authorization request) | invalid/nonpositive/bool/noninteger fixed_dimension; temporary copies of actual selected configs with conflicting explicit dim/dimension; actual existing composition config passed to fixed mode; actual fixed base with mismatched requested dimension | V7.3.5 and identical verification reruns |
| Definition manifest rejection | mutate actual exported record: unsupported schema, missing/changed identity/parameters/hash, path escape and incompatible software metadata | V7.4.4 and identical verification reruns |

**Inputs / Contracts:** each purpose is independently scoped above; no blanket fixture license. Invalid copies are labeled nonrepresentative. Source vectors/outputs and exported originals remain unchanged; controlled faults never supply invented successful values. **Facts Protected:** F01–F08 and all accepted real captures. **Description:** Record the specific remaining test generators before use, under DP-05; selection of the features alone does not expand V2/V4/V6 fixture permissions. **Validation / Handoff:** record the actual human answer before the named dependent checks; positive/minimized-real-data checks are separate tasks. **Acceptance Criteria:** explicit authorization per named purpose/task or continued gate; no generated negatives silently adopted.

#### V5.0 / V7.0 authorization record — 2026-10-04

**Decision:** Approved by the trusted product owner through the conversation:
"Synthetic test data is approved." **Purpose and scope:** all named rows in
V5.0 and V7.0, including fixed-dimension guard requests; controlled invalid
copies/results/failures, source-derived half-step/axis vectors and corrupted
definition records. **Generator:** deterministic Python stdlib, NumPy and
pytest operations named in those tasks; no random test generation. Initial
verification environment: Python 3.14.8, NumPy 2.4.2, pytest 9.0.2; record
supported-minor versions at verification. **Covered tasks:** V5.2–V5.4,
V7.1.4, V7.2.4, V7.3.4, V7.3.5, V7.4.4 and identical reruns/integrated V8.2
verification. Originals remain untouched; official engines supply expected
SPSO boundary outputs. Adversarial copies never become authoritative reference
data. Cleanup uses task-owned external paths or pytest temporary directories.
The previous unanswered requests are historical; they no longer block dispatch.

**Tool and snapshot sensitivity consumers:** The same 2026-10-04 owner
approval covers deterministic isolated snapshot/metadata guards, source setup
CLI/repeat/diagnostic/result failures, profiler argument/provenance/restoration
failures and installed-artifact operational failures. Actual source/capture
copies and delegated real subprocesses are used; controlled faults are labeled
adversarial and never treated as valid reference observations. Generator,
versions, consumers and preservation requirements are recorded before execution
in `/tmp/pymofl-implementation/operational-sensitivity.md`. Immutable archive
pin checks remain intact.

**Operational sensitivity consumers:** The owner's synthetic-testing approval
also covers the remaining requirement-owned V5.2/V5.4 and V7.2.4 execution
checks: wrappers around actual public-loaded functions that raise a controlled
construction/scalar/batch exception or corrupt an actual result's type, shape,
finiteness or numerical agreement; missing/malformed manifest transport and
fresh-report creation failures. Deterministic pytest monkeypatches and temporary
paths isolate these operations. Actual approved expected values remain
unchanged, no fabricated output counts as successful reference evidence, and
the deviation policy stays empty. These checks verify honest statuses,
execution counts, failure exits and the declared coverage gate.

### V7.1: Bounded deterministic batch helper

**Role:** Capability. **Baseline:** integrated V4/V6 review. **Facts:** F09/E09; protect F01–F08. **Provider Boundary:** optional helper beside public evaluation, no new mandatory component methods. **Review:** sequential writer/oracle. **Exit:** supported-array values/ownership/bounds, supported-minor/common/artifact gate, authorized negative checks and changed-domain coverage pass.

The proposed evaluate_chunks(function, X, chunk_size, *, deterministic, batch_independent, out=None) requires explicit true caller declarations. It never infers purity from class names. X must be a NumPy matrix with integer or floating dtype; child results must be NumPy arrays with integer or floating dtype and shape (chunk_rows,). Both are copied/cast to float64; bool, complex, object and string dtypes are rejected rather than silently coerced. Output is float64 shape (N,). Copy/cast only one input chunk at a time, preserving caller ownership, strided/readonly inputs and row order. A writable float64 (N,) out is optional and cannot overlap X. Empty batches return without child evaluation. Each child receives at most chunk_size rows; arbitrary custom internals are not bounded by the helper. Validate static arguments before calling the child; on child failure prior completed output chunks remain written and the exception propagates. No automatic noisy chunking or rollback allocation of an entire population.

#### Task V7.1.1: Test helper availability and real bounded evaluation

**Type:** Test. **PRD Trace:** LOCAL-R07/FEAT-01. **Fact / Evidence:** F09, Tier 1 → E09. **Expected Failure Signature:** public evaluate_chunks availability assertion fails; existing whole-batch evaluation does not satisfy observed requested row bounds. Import errors do not count as RED. **Makes Green:** N/A. **Depends On:** V4.3 and V6.3. **Work Packet:** Sequential. **Write Scope:** proposed tests/test_evaluation.py. **Real Data Dependency:** unchanged DATA-01 F1 D10 points/outputs, smaller/empty slices, actual array views and captured vectors. **Provider Boundary:** public helper and observation-only delegation to the real loaded evaluator.

**Inputs / Contracts:** public callable availability, row-order values, positive chunk sizes/remainder, input copies observed by delegate, empty inputs, reusable output identity, wrong-size actual captured vector and overlapping actual input view. No generated negative controls, altered reference fields or fault responses. **Facts Protected:** F01–F08 and source numerical values. **Description:** Open the new opt-in capability with actual data and observable batch bounds. **Validation / Handoff:** uv run --locked pytest tests/test_evaluation.py; retain public-availability RED and actual full-batch observation, then run protecting APIs. **Acceptance Criteria:** intended assertion-level RED, source values untouched, no import failure claimed as evidence.

#### Task V7.1.2: Implement the opt-in helper

**Type:** Implement. **PRD Trace:** LOCAL-R07/FEAT-01. **Fact / Evidence:** F09/E09. **Expected Failure Signature:** V7.1.1 RED. **Makes Green:** E09 supported-input assertions. **Depends On:** V7.1.1. **Work Packet:** Sequential. **Write Scope:** proposed src/pyMOFL/evaluation.py, public __init__.py export and API docs. **Real Data Dependency:** unchanged V7.1.1 captures. **Provider Boundary:** existing OptimizationFunction.evaluate_batch, NumPy and stdlib only.

**Inputs / Contracts:** above helper contract; use integral index semantics, reject bool/nonpositive and unsupported declarations, validate shape/dtype/overlap before evaluation; copy/cast per chunk, validate child result shape, propagate failures once. Argument checks precede generated failure verification; V7.0 authorization was resolved on 2026-10-04. **Facts Protected:** F01–F08, extension API and default evaluators. **Description:** Add one helper without changing each benchmark signature or default strategy. **Validation / Handoff:** positive E09 and existing numerical APIs, inspection of working allocation/ownership. **Acceptance Criteria:** bounded rows and source values pass, output identity/empty behavior correct, no noisy-default changes, no new dependencies.

#### Task V7.1.3: Verify supported-data behavior and compatibility

**Type:** Verify. **PRD Trace:** LOCAL-R07/LOCAL-R08. **Fact / Evidence:** F09, Tier 1 → E09 supported scope. **Expected Failure Signature:** N/A. **Makes Green:** N/A. **Depends On:** V7.1.2. **Work Packet:** Sequential. **Write Scope:** helper evidence/docs only. **Real Data Dependency:** same unchanged/minimized V7.1.1 data. **Provider Boundary:** public core/artifact API and independent oracle.

**Inputs / Contracts:** supported/minimized data cases and optionality; negative-control closure is established separately by V7.1.4, now verified. **Facts Protected:** F01–F08. **Description:** Independently verify positive capability and migration limits. **Validation / Handoff:** supported-minor common gate, core artifact checks and oracle review; coverage cannot be called final before V7.1.4. **Acceptance Criteria:** supported cases green, partial writes/declaration responsibility documented; feature closure remains gated until failure checks pass.

#### Task V7.1.4: Verify authorized rejection and partial-failure cases

**Type:** Verify. **PRD Trace:** LOCAL-R07/FEAT-01. **Fact / Evidence:** F09, Tier 1 → E09. **Expected Failure Signature:** N/A — verification of implemented guards; if a defect appears, add its Test/Implement pair before repair. **Makes Green:** N/A. **Depends On:** V7.1.3 and accepted V7.0 bounded-helper scope. **Work Packet:** Sequential. **Write Scope:** tests/test_evaluation.py and evidence. **Real Data Dependency:** Generated (authorized by Task V7.0), exact bounded-helper cases only; real successful values unchanged. **Provider Boundary:** public helper, pytest cleanup and real evaluator delegation.

**Inputs / Contracts:** V7.0's named control/result/fault cases, preserve error/cause and one invocation, observe completed-chunk partial writes and unchanged input. **Facts Protected:** F01–F08 and supported E09. **Description:** Close failure classes without disguising new input authorization as feature selection. **Validation / Handoff:** scoped tests/changed-domain coverage/common gate and oracle. **Acceptance Criteria:** required rejection/failure behavior verified, no fixture leaks, coverage policy passes; only then close FEAT-01.

### V7.2: Optional validation CLI/report adapter (oracle-approved contract)

**Role:** Capability. **Baseline:** supported V5.1.3 result producer. **Facts:** F10/E10; protect F01–F09. **Provider Boundary:** existing optional Typer CLI calls the library-neutral helper; no objective depends on CLI/reporting. **Exit:** stable structured report, observed actual execution and status-aware exits, installed optionality, authorized failure checks and final coverage.

`pymofl validate --capture-root ROOT --manifest FILE --report FILE` consumes an
actual producer manifest and emits the same version1 report as the stdlib
required script. No second validator or duplicated numerical logic. Exit0
requires all required cases passed or explicitly reasoned numerical deviations;
exit1 represents failed/unavailable/invalid required execution. Setup errors
must produce an honest unavailable/failed outcome, never an empty success.
Paths and canonical IDs may be reported; raw vectors and reference engines
stay outside CLI output/imports. The CLI is available only with existing extras,
never a new core requirement. Reports record pinned dataset identity/freshness
limits, not an unverified claim of latest source.

#### Task V7.2.1: Test the actual positive CLI workflow

**Type:** Test. **PRD Trace:** LOCAL-R07/FEAT-02. **Fact / Evidence:** F10/E10 supported scope only. **Expected Failure Signature:** validate command availability assertion fails; import/setup failures are not RED. **Makes Green:** N/A. **Depends On:** V5.1.3 and oracle approval of this contract. **Work Packet:** Sequential. **Write Scope:** tests/cli/test_validation_report.py. **Real Data Dependency:** unchanged approved CEC2014 captures, actual manifest/report produced by V5 helper; explicitly provisioned through PYMOFL_REFERENCE_CAPTURE_ROOT. No hand-built realistic reports or failure copies. **Provider Boundary:** real existing CLI app/runner and stdlib required script.

**Inputs / Contracts:** real450-case report, zero exit, schema/source/hash/count/ID identity equal to library producer; command help always hermetic. **Facts Protected:** F01–F09 and data/reference identity. **Description:** Show end-to-end optional reporting through actual creation/execution. **Validation / Handoff:** command-availability RED then provisioned CLI run. **Acceptance Criteria:**450 scalar/450 batch executions and complete report, no fabricated observed results; failure classes stay V7.2.4.

#### Task V7.2.2: Implement the narrow CLI adapter

**Type:** Implement. **PRD Trace:** LOCAL-R07/FEAT-02. **Fact / Evidence:** F10/E10 supported scope. **Expected Failure Signature:** V7.2.1 intended RED. **Makes Green:** real command/report assertions. **Depends On:** V7.2.1. **Work Packet:** Sequential. **Write Scope:** src/pyMOFL/cli/reference.py, CLI main wiring and usage docs; existing dependencies only. **Real Data Dependency:** same actual producer workflow. **Provider Boundary:** optional Typer adapter over pyMOFL.reference_validation.

**Inputs / Contracts:** above command/report/exit contracts, no duplicate parser/math/status logic. **Facts Protected:** F01–F09. **Description:** Expose library-neutral observed validation through existing CLI. **Validation / Handoff:** focused positive CLI and installed artifact smoke. **Acceptance Criteria:** real report and0 exit on required success, core imports remain CLI-free; no final failure/coverage claim.

#### Task V7.2.3: Verify supported reporting and installed optionality

**Type:** Verify. **PRD Trace:** LOCAL-R07/LOCAL-R08. **Fact / Evidence:** F10/E10 supported scope only. **Expected Failure Signature:** N/A. **Makes Green:** N/A. **Depends On:** V7.2.2. **Work Packet:** Sequential. **Write Scope:** artifact smoke and evidence/usage documentation/ATLAS. **Real Data Dependency:** same approved captures and real generated manifests/reports. **Provider Boundary:** installed core/CLI environments and independent oracle.

**Inputs / Contracts:** strict version1 JSON, actual source/software/dataset metadata and execution counts; no latest-release claim or simulated successes. **Facts Protected:** F01–F09. **Description:** Verify adapter and import modularity independently. **Validation / Handoff:** supported-minor common gate, actual installed CLI run and oracle. **Acceptance Criteria:** observed valid required report and optionality pass; final FEAT02 closure remains V7.2.4.

#### Task V7.2.4: Verify authorized failure reports and exit codes

**Type:** Verify. **PRD Trace:** LOCAL-R07/FEAT-02. **Fact / Evidence:** F10/E10. **Expected Failure Signature:** N/A; defects require a separately recorded Test/Implement pair before repair. **Makes Green:** N/A. **Depends On:** V7.2.3 and accepted V7.0 report/CLI-failure scope. **Work Packet:** Sequential. **Write Scope:** tests/cli/test_validation_report.py and evidence. **Real Data Dependency:** Generated (authorized by V7.0), exact named mutations of actual producer records/capture copies only. **Provider Boundary:** public helper/CLI and temporary directories.

**Inputs / Contracts:** unavailable/invalid/failed/deviation distinction, honest observed counts and nonzero required failure exits; operational/provenance errors never deviations. **Facts Protected:** prior facts, dataset identity, unchanged valid originals. **Description:** Close status/exit boundary classes. **Validation / Handoff:** scoped failure sensitivity, final changed-domain coverage, common gate and oracle. **Acceptance Criteria:** invalid required execution cannot report success; only then close FEAT02.

### V7.3.0: Acquire source-backed SPSO native-coordinate captures

**Type:** Data Acquisition. **PRD Trace:** LOCAL-R07/FEAT-03. **Fact / Evidence:** F11/E11 prerequisite, not yet passed. **Expected Failure Signature:** N/A. **Makes Green:** N/A. **Depends On:** owner-authorized official acquisition, selected FEAT-03. **Work Packet:** Sequential. **Write Scope:** task-owned external source snapshots/capture procedure and provenance documentation only. **Real Data Dependency:** official particleswarm.info standard_pso_2007.zip (SHA256 f9524f7f9568009b4ab5c76cd32d91c255fef978b4ff64891b460cb520f34bd1) and standard_pso_2011_c.zip (SHA256 11692f658158b18aafd97d667eeebdc7527cf21147d530d73d4c7eb795af0557). **Provider Boundary:** external reference-only harness; no optimizer in library.

**Inputs / Contracts:** restrict IDs to 4 Tripod D2, 11 Network D42, 18 Gear D4 and 21 Spring D3; record archive/source/patch/driver/executable hashes, compiler/flags/platform and observed rights. Capture unchanged constructor bounds and source-documented solutions where supplied. No arbitrary intermediate points, half-step neighbors or zero-axis combinations before V7.0 authorization. The harness applies the actual source quantis operation and perf objective-distance contract in native coordinates; setting source SS.normalise=0 bypasses optimizer coordinate normalization and is explicitly recorded. For 2011, remove only unused reference-harness GSL declarations/includes; isolate unselected optimizer call sites with an abort-only stub, never a fabricated numerical value. Inspect actual linked source and quantis extraction before compiling; retain patches. Reject unsupported IDs/dimensions before evaluation. Do not execute optimizer main or redistribute external code/data with unspecified rights.

**Facts Protected:** F01–F10, existing Tripod/Gear/Network/Spring/Quantized meanings and constants. **Description:** Independently establish selected official definitions before declarative configs or source-backed variant classes. Current source inspection shows zero-axis Tripod, Gear squared error, half-up quantization and Spring coordinate/penalty differences; reusing a class is conditional on proven equivalent semantics. **Validation / Handoff:** bounded compiler/process timeouts, repeat captures and compare exact bytes, source-backed controls, UBSan inspection on selected paths, oracle review of portability/stub and capture scope. Compare legacy library outputs only to identify differences, never to create expected values. **Acceptance Criteria:** pinned observed captures and explicit discrepancy disposition for both versions/selected IDs, or named blocked source path; no claim of the complete SPSO suite or optimum certification without evidence. Implementation tasks remain non-dispatchable until expanded and reviewed.

**Native-coordinate adapter clarification:** 2011 problemDef scales quanta before returning when SS.normalise>0. Setting the returned flag to zero alone is insufficient. The reference-only patch disables that final quantum-normalization branch, preserving the declared native step constants; the driver then sets SS.normalise=0 before quantis/perf. Retain and review this exact patch as coordinate adaptation, without changes to objective/constraint or rounding formulas.

**Link boundary:** 2011's included unselected CEC2005 F107 noise case references alea_normal. The reference-only harness uses an abort-only RNG stub, alongside the abort-only PSO stub. Both are unreachable for IDs 4/11/18/21; an unexpected call terminates, never returns fabricated numerical data. A linker failure established this dependency before the stub was added; capture execution resumes only after review of this addition.

### Deferred phase contracts

### V7.4: Definition manifest (approved supported contract)

**Role:** Capability. **Baseline:** selected supported suite APIs and V7.3 supported-data evidence. **Facts:** F12/E12; protect numerical APIs. **Provider Boundary:** optional pyMOFL.definition module, plain versioned JSON-compatible records; no storage service, optimizer or object deserialization framework. **Exit:** source-backed definition/export/reconstruction evidence, compatibility/hash rejection, supported-minor/artifact gate and oracle review.

Proposed export_definition(name_or_id, *, suite, dimension=None, instance=None) constructs a fresh deterministic suite definition through existing public loading. reconstruct_definition(manifest) validates and constructs a fresh instance through the same loader. It never imports a class named by untrusted JSON or deserializes executable objects. Initial supported families: BBOB noiseless, GNBG, CEC2014, deterministic CEC2005 configurations and the selected SPSO distributions; unsupported/noisy configurations are rejected explicitly before construction where their configs identify noise. BBOB records the actual default iid1 when no instance is supplied; GNBG/CEC/SPSO record instance=None and reject a supplied irrelevant selector. Canonical suite/function identities come from actual selected production config/factory resolution, not just echoing an alias. Arbitrary registry/user-created objects, custom keyword arguments, evaluation caches, current mutable instances and RNG state replay are outside this first contract.

Schema version 1 includes canonical suite/function/instance/resolved dimension; original declarative config (or actual BBOB factory-produced config); actual resolved component/transform parameters and bounds from fresh construction; selected bundled config/data path hashes; package numerical code hashes; pyMOFL/NumPy/Python version and platform metadata; and a content integrity hash. Actual runtime defaults must be observed, not inferred from config omission. A narrow inspection of fresh library-owned component data may record parameters; it is evidence only, never an instruction to recreate arbitrary objects. Numeric arrays use explicit dtype/shape and base64 bytes so real infinite bound metadata can be represented without nonstandard JSON NaN/Infinity. Object arrays may contain only explicitly supported enums; unsupported callable/object/Generator state fails. No opaque fallback stringification.

Replay compatibility initially requires matching pyMOFL version/code hashes, NumPy version, Python major/minor and platform machine/system; this software/platform identity is recorded, including the informational Python patch version. It is not a complete BLAS/runtime environment capture. Reconstructed parameter snapshots and selected artifact hashes must match. These are conservative definition-replay checks, not a promise of bit-identical evaluation across changed BLAS/runtime environments. Manifest paths are package-relative and confined to bundled constants/code; reconstruction obtains allowed paths from current production resolution and never opens arbitrary manifest-selected paths. Hashes detect mismatch, not authorship or authenticity. No recorded seed is described as a current RNG state.

#### Task V7.4.1: Test actual definition creation and replay

**Type:** Test. **PRD Trace:** LOCAL-R07/FEAT-04. **Fact / Evidence:** F12, Tier 1 → E12. **Expected Failure Signature:** optional public definition helper availability assertion fails; imports are setup, not behavioral RED. **Makes Green:** N/A. **Depends On:** oracle approval of V7.4 contract and supported V7.3.3. **Work Packet:** Sequential. **Write Scope:** proposed tests/test_definition.py. **Real Data Dependency:** unchanged DATA01 CEC2005 F1/D10 request/points; DATA03 BBOB F1/D2/iid1; actual GNBG F24/D30 used by V6; actual selected SPSO source requests/captures. These canonical existing requests seed the system's new validated export path; later assertions consume its actual produced records, never hand-assembled manifests. **Provider Boundary:** optional public exporter/reconstructor and json serialization.

**Inputs / Contracts:** schema and required observed fields, actual resolved parameter/default provenance, selected config/data/source hashes, strict JSON round-trip, fresh-instance identity and same-definition numerical replay on unchanged available points. For suites without retained point arrays, compare actual constructed parameter snapshots and source-bound/native captured points, not newly generated random data. Assert source/class identity and parameters from actual produced objects/configs. No invalid record mutations before V7.0. **Facts Protected:** F01–F11, original configs/data/streams. **Description:** Open canonical create-and-read provenance, then use produced records for reproducibility. **Validation / Handoff:** assertion-level availability RED, focused unchanged-data tests and source integration where provisioned. **Acceptance Criteria:** no opaque state dump mistaken for a replay instruction, no hand-built realistic manifest, actual resolved defaults observed.

#### Task V7.4.2: Implement optional deterministic manifest helper

**Type:** Implement. **PRD Trace:** LOCAL-R07/FEAT-04. **Fact / Evidence:** F12/E12. **Expected Failure Signature:** V7.4.1 intended RED. **Makes Green:** E12 supported creation/replay. **Depends On:** V7.4.1. **Work Packet:** Sequential. **Write Scope:** proposed src/pyMOFL/definition.py and reproducibility docs; no mandatory new function methods. **Real Data Dependency:** same actual requests, bundled constants/configs and produced records. **Provider Boundary:** existing public load, suite/config/file-reference utilities and factory-produced BBOB configs; stdlib/NumPy only.

**Inputs / Contracts:** above versioned schema and replay restrictions; resolve selected files rather than hashing unrelated datasets; hash current package code without stale cache assumptions; validate JSON at the boundary before using typed records. Reject unsupported/noisy definitions honestly. No generic persistence service or executable serialization. **Facts Protected:** F01–F11 and extension behavior. **Description:** Implement the smallest optional helper needed to observe and replay supported definitions. **Validation / Handoff:** focused producer round-trips and source-backed snapshots; inspect actual file-resolution/default paths. **Acceptance Criteria:** real fields present and verified, unchanged numerical/constructor APIs, no new dependencies or private state-replay claims.

#### Task V7.4.3: Verify supported replay and installed optionality

**Type:** Verify. **PRD Trace:** LOCAL-R07/LOCAL-R08. **Fact / Evidence:** F12 → E12 supported scope. **Expected Failure Signature:** N/A. **Makes Green:** N/A. **Depends On:** V7.4.2. **Work Packet:** Sequential. **Write Scope:** evidence/docs and artifact smoke. **Real Data Dependency:** producer-created records on each actual supported-minor environment, existing source data. **Provider Boundary:** independent oracle and installed core/CLI environments.

**Inputs / Contracts:** supported-minor round-trips, original versus resolved config/default distinctions, artifact/source identity and environment limitations. **Facts Protected:** F01–F11. **Description:** Independently verify the exported record describes the actual definition and does not become a new core dependency. **Validation / Handoff:** common gate, numerical replay on available real cases, independent oracle and ATLAS report. **Acceptance Criteria:** supported scope observed green and limitations explicit; failure/coverage closure remains V7.4.4.

#### Task V7.4.3.1: Verify actual deterministic definition breadth

**Type:** Verify. **PRD Trace:** LOCAL-R07/FEAT-04. **Fact / Evidence:** F12/E12 supported scope only. **Expected Failure Signature:** N/A; an unexpected actual deterministic failure requires a recorded protected Test/Implement repair pair. **Makes Green:** N/A. **Depends On:** V7.4.3 and oracle review of this contract. **Work Packet:** Sequential. **Write Scope:** tests/test_definition.py and evidence/compatibility notes. **Real Data Dependency:** actual selected configuration/factory IDs, existing valid dimensions CEC2005/CEC2014 D10, GNBG D30, BBOB D2/iid1 and selected SPSO entry dimensions. Strict JSON records are produced by the actual exporter, never hand-assembled or mutated. **Provider Boundary:** actual configuration utilities/factories, public export/reconstruction and unchanged captures supplied explicitly.

**Inputs / Contracts:** enumerate real IDs; identify and list noisy configuration exclusions before loading them, according to the supported deterministic contract. Each remaining request must export, strict-JSON round-trip and reconstruct with matching canonical request, original config, resolved parameters and artifacts. Compare unchanged retained numerical rows only when available; otherwise record metadata-only replay. No random/new numerical vectors, invalid requests or fabricated expected values. **Facts Protected:** all original configs, reference values and current RNG/evaluation semantics. **Description:** Establish breadth beyond the seven initial examples without treating unsupported-state rejection as comprehensive support. **Validation / Handoff:** source-derived request/exclusion inventory, supported-minor execution and independent oracle; investigate every unexpected deterministic failure without skips. **Acceptance Criteria:** every inventoried deterministic request replays or has a separately reviewed correction/support restriction; no claim of independent numerical accuracy for metadata-only cases. V7.4.4 and final coverage are separately verified in the current evidence index.

#### Task V7.4.4: Verify authorized incompatible/corrupt records

**Type:** Verify. **PRD Trace:** LOCAL-R07/FEAT-04. **Fact / Evidence:** F12 → E12. **Expected Failure Signature:** N/A; defects require a separate Test/Implement pair before repair. **Makes Green:** N/A. **Depends On:** V7.4.3 and accepted V7.0 definition-rejection scope. **Work Packet:** Sequential. **Write Scope:** tests/test_definition.py and evidence. **Real Data Dependency:** Generated (authorized by V7.0), named mutations of actual exported records only. **Provider Boundary:** public reconstruction and pytest temporary copies.

**Inputs / Contracts:** unsupported schema, missing/changed identity/parameters/hash, path escape and incompatible software metadata; originals unchanged, exact operations/versions recorded. **Facts Protected:** prior source/definition facts, no numerical expected values invented. **Description:** Close integrity and compatibility classes without trusting unvalidated manifest data or arbitrary paths. **Validation / Handoff:** rejection sensitivity, changed-domain coverage/common gate and independent oracle. **Acceptance Criteria:** required invalid/incompatible records cannot silently replay, no fixture leaks; only then close FEAT-04.

#### V7.3 selected-source implementation contract (review before dispatch)

**Role:** Capability. **Baseline:** V7.3.0 approved captures. **Facts:** F11/E11; protect existing library definitions. Expose only the selected four native-coordinate functions in each pinned SPSO version, not the full optimizer test catalog. Source metadata records archive revisions/update history and the 2007 Spring g2 penalty bug. Source-faithful 2007 and corrected 2011 are distinct explicit variants; neither silently replaces the existing generic CompressionSpringFunction.

Reuse NetworkFunction, GearTrainFunction plus PowerTransform(2), existing BiasTransform(-objective), and a new small AbsoluteTransform. Source-equivalence claims are limited to declared search bounds. Network reuse is equivalent after source quantization only for in-bounds binary coordinates; its thresholding differs from raw source integer link sums outside that domain. Bounds remain metadata, without clipping/enforcement or an external-domain equivalence promise. Add SPSOTripodFunction with source sign(0)=0 and SPSOCompressionSpringFunction with native [N,D,d] coordinates and explicit penalty_version="2007"/"2011"; the 2007 g2-positive branch deliberately uses the stress multiplier, documented as historical behavior. No optimum-point certificate for Spring is available: get_global_minimum remains NotImplementedError, while suite metadata records the source target as a best-known objective, not a certified point. Use a new HalfUpQuantizationTransform with per-coordinate steps; step zero passthrough, source threshold 1e-40, round floor(0.5+x/q). Never change legacy Quantized np.rint or existing Tripod/Spring defaults.

Configs declare dimension_policy="fixed_heterogeneous", entry dimensions 2/42/4/3, actual bounds/steps, reference target and scope/discrepancy metadata. load accepts the matching entry dimension or omitted dimension; get_suite omits a single dimension for this heterogeneous suite and rejects a supplied one. Loader applies entry bounds metadata only for this declared policy, without enforcing bounds or changing existing suite bounds. Add explicit spso2007/spso2011 prefix routing. FunctionFactory/TransformBuilder reuse existing pipelines and add only two primitive transform cases; no new suite-factory framework or optimizer runtime.

#### Task V7.3.1: Test selected source cases and explicit suite availability

**Type:** Test. **PRD Trace:** LOCAL-R07/FEAT-03. **Fact / Evidence:** F11, Tier 1 → E11. **Expected Failure Signature:** public suite availability assertion and source-record scalar/batch assertions fail; missing imports are setup, not RED. **Makes Green:** N/A. **Depends On:** V7.3.0 capture approval and implementation-contract oracle review. **Work Packet:** Sequential. **Write Scope:** proposed tests/benchmark_suites/test_spso_validation.py and focused primitive tests. **Real Data Dependency:** actual V7.3.0 20 captured cases and eight metadata records, existing actual bounds/legacy DATA01 numerical values where relevant. **Provider Boundary:** public loading/config/factory evaluation; local external source data supplied explicitly.

**Inputs / Contracts:** both versions, selected IDs, actual constructor bounds/steps, source documented solutions, scalar/batch/source quantization agreement, array ownership and empty slices of real vectors, output buffers using actual captured values. Wrong-dimension checks use another unchanged captured vector; no generated arbitrary ties/axis cases before V7.0. Reuse shared validators only methods that do not generate unauthorized data; do not invoke their inherited random generators. **Facts Protected:** existing Tripod/Network/Gear/Spring/Quantized aliases and meanings. **Description:** State source equivalence for observed selected native cases and honest scope metadata. **Validation / Handoff:** explicit local capture environment, public availability RED then numerical RED; default capture-dependent tests may skip with a precise missing-reference reason, explicit required integration cannot pass on skips. **Acceptance Criteria:** source values untouched, differences between pinned Spring versions asserted, unchanged legacy aliases protected.

#### Task V7.3.2: Implement explicit variants, primitives and declarative suites

**Type:** Implement. **PRD Trace:** LOCAL-R07/FEAT-03. **Fact / Evidence:** F11/E11. **Expected Failure Signature:** V7.3.1 intended assertions. **Makes Green:** E11 supported-source cases. **Depends On:** V7.3.1. **Work Packet:** Sequential. **Write Scope:** proposed benchmark/spso_tripod.py and benchmark/spso_compression_spring.py, transformations/half_up_quantization.py and absolute.py, transformation exports/builder, constants/spso2007/spso2007_suite.json and spso2011/spso2011_suite.json, loader fixed-policy/prefix wiring, FunctionFactory.create_function optional fixed_dimension seam and source-scope docs. **Real Data Dependency:** pinned source definitions and approved unchanged captures; original C/header files remain external. **Provider Boundary:** existing OptimizationFunction/VectorTransform/ScalarTransform/Bounds/FunctionFactory.

**Inputs / Contracts:** above explicit selected-source contract, typed public APIs and NumPy scalar/batch consistency; reuse elementary transforms; independent Python implementation of mathematical definitions, no external optimizer source embedded. **Facts Protected:** F01–F10, no changes to legacy class behavior or cec2005_suite.json. **Description:** Add minimum variant code where source differences require it and metadata-only bounds for new declared fixed suites. **Validation / Handoff:** E11 positive controls, protecting legacy suites, factory/import boundary inspection. **Acceptance Criteria:** both selected suites reconstruct/evaluate approved cases with explicit version/bug/target contracts, no automatic quantization of old functions or extra dependency.

**Factory seam clarification:** ConfigParser injects a default dimension even for existing fixed constructors. Add optional keyword-only fixed_dimension to FunctionFactory.create_function; only declared fixed-suite loader routes opt in. Reject contradictory original config dimensions and composition/hybrid/decomposed delegation for this option; use fixed dimension for file loading, remove dimension fields only from the local copied constructor parameters, and validate the actual base dimension before transforms. Default factory calls and caller configs remain unchanged. Oracle approved this narrow seam after unchanged source tests exposed the constructor mismatch; no generic signature filtering or legacy class constructor edits.

#### Task V7.3.3: Verify source scope and compatibility

**Type:** Verify. **PRD Trace:** LOCAL-R07/LOCAL-R08. **Fact / Evidence:** F11 → E11 supported scope. **Expected Failure Signature:** N/A. **Makes Green:** N/A. **Depends On:** V7.3.2. **Work Packet:** Sequential. **Write Scope:** source evidence/docs, proposed scripts/capture_spso.py retained explicit procedure and artifact checks. **Real Data Dependency:** same unchanged source captures. **Provider Boundary:** independent oracle and explicit local reference integration.

**Inputs / Contracts:** repeated release/UBSan identities, pinned rights/provenance, source-native coordinate adaptation, selected numerical tolerance rtol=1e-12/atol=1e-10, legacy APIs and aliases. **Facts Protected:** F01–F10 and benchmark identity. **Description:** Verify modularity and scope independently, not merely that a suite loads. **Validation / Handoff:** common supported-minor gate and installed artifacts, oracle source/capture/code review. **Acceptance Criteria:** observed source cases pass without required skips, no optimizer/reference dependency in core; broader boundary/coverage closure is established separately by V7.3.4, now verified.

**Retained setup contract:** scripts/capture_spso.py takes an explicit local archive directory containing the two pinned archive filenames and an empty external output directory. Verify archive/source identity before compiling, safely extract C/header files, preserve originals/patches and bounded compiler/process controls. No network, arbitrary new inputs or implicit evaluation dependency. Reproduce the approved JSONL hashes; separate per-build flags remain recorded. Retain the source constructor/min-max/documented-solution scope only; adding boundary input generation remains V7.3.4.

#### Task V7.3.4: Verify authorized half-step and axis boundaries

**Type:** Verify. **PRD Trace:** LOCAL-R07/FEAT-03. **Fact / Evidence:** F11 → E11. **Expected Failure Signature:** N/A; defects require a new Test/Implement pair before repair. **Makes Green:** N/A. **Depends On:** V7.3.3 and accepted V7.0 SPSO boundary scope. **Work Packet:** Sequential. **Write Scope:** external source captures/provenance and boundary tests. **Real Data Dependency:** Generated (authorized by V7.0), exact half-step neighbors/ties and zero-axis combinations, expected outputs from actual official source engines only. **Provider Boundary:** same pinned external harness and public suite APIs.

**Inputs / Contracts:** record each named deterministic boundary operation and installed versions, no random expected values, original captures unchanged. **Facts Protected:** existing values/quantization and source version discrepancies. **Description:** Verify the definitions precisely where legacy generic functions differ. **Validation / Handoff:** source batch/scalar controls, changed-domain coverage, oracle and integrated gate. **Acceptance Criteria:** no false equivalence, no fabricated references, required coverage achieved; only then close FEAT-03.

#### Task V7.3.5: Verify separately authorized fixed-dimension guard classes

**Type:** Verify. **PRD Trace:** LOCAL-R07/FEAT-03. **Fact / Evidence:** F11/E11 and protecting F02/F04. **Expected Failure Signature:** N/A; defects require a separately recorded Test/Implement pair before repair. **Makes Green:** N/A. **Depends On:** V7.3.3 and explicit V7.0 fixed-dimension authorization, requested separately from earlier half-step/axis scope. **Work Packet:** Sequential. **Write Scope:** focused tests for FunctionFactory and evidence; no changes to real source configs/captures. **Real Data Dependency:** Generated (authorized by V7.0), exact additional deterministic guard requests/copies named above, using actual selected SPSO and existing composition configs. No fabricated successful numerical values or custom fake provider/base class. **Provider Boundary:** public FunctionFactory fixed_dimension seam and existing real constructors.

**Inputs / Contracts:** strict positive-integer fixed option, rejection of conflicting explicit fields before construction, forbidden composition delegation, actual base dimension validation and unchanged caller config. **Facts Protected:** existing factory/parser behavior and selected source identity. **Description:** Close the new opt-in seam's rejection branches without inventing realistic fixture entities. **Validation / Handoff:** explicit owner scope first, deterministic Python/NumPy/pytest versions recorded, focused checks/changed-domain coverage and oracle. **Acceptance Criteria:** all named guards observed and originals preserved; this task remains gated independently if unanswered.

### V8: Documentation and integrated delivery

Supported documentation maintenance can proceed after the implemented positive
interfaces integrate. Final delivery still depends on every named failure/data
gate and may not certify partial features or unobserved remote CI.

#### Task V8.1: Reconcile supported documentation and generated catalog

**Type:** Evidence Maintenance. **PRD Trace:** LOCAL-R08 and selected FEAT01–FEAT04. **Fact / Evidence:** protecting F01–F12, supported scope only. **Expected Failure Signature:** N/A; no objective/API behavior changes. **Makes Green:** N/A. **Depends On:** supported V5.1.3, V7.1.3, V7.2.3, V7.3.3 and V7.4.3; oracle review of this task. **Work Packet:** Sequential. **Write Scope:** README.md, docs/index.md, ROADMAP.md and its docs/roadmap.md copy, docs/SPSO_Functions.md, docs/function_catalog.md, scripts/generate_catalog.py and current evidence/compatibility documents; only descriptive noise-state and quantization statements in CODING_GUIDELINES.md and its docs copy. **Real Data Dependency:** actual live registered classes/constructor signatures, approved selected SPSO source facts and observed implementation reports. The existing catalog producer reads real registry metadata; no representative fixtures, new vectors or invented numerical reference outputs. **Provider Boundary:** existing documentation/catalog producer, no new runtime dependency.

**Inputs / Contracts:** user-facing entry points, actual aliases and fixed/source scopes, explicit noisy-state exceptions to statelessness, source provenance/rights/freshness limits and unfinished gates. Constructor defaults are metadata, not proof of scalable/fixed dimensional support. Catalog labels must say defaults or unavailable constructor metadata rather than invent that property. Preserve historical roadmap claims as historical, not newly verified release/reference evidence. **Facts Protected:** all numerical/reference/API identities, user-owned changes and proposal baseline history. **Description:** Remove stale supported-surface descriptions and synchronize the two entry documentation copies; retain a single generator for current catalog metadata. Straightforward documentation corrections need no new GOTCHA/ATLAS architecture artifact. **Validation / Handoff:** generated catalog counts/links and selected source rows inspected against actual registry/code; doc commands/options checked against live interfaces; whitespace/lint/format as affected, independent oracle. **Acceptance Criteria:** current supported documentation matches code and observed scope; no full-feature/release/remote-CI certification, no change to baseline historical evidence or numerical methods.

#### Task V8.2: Verify the full integrated plan and delivery state

**Type:** Verify. **PRD Trace:** LOCAL-R01–LOCAL-R08 and selected FEAT01–FEAT04. **Fact / Evidence:** F01–F12/E01–E12 at their full approved scopes. **Expected Failure Signature:** N/A; defects require separately recorded repair pairs. **Makes Green:** N/A. **Depends On:** V8.1, V5.4, V7.1.4, V7.2.4, V7.3.4, V7.3.5, V7.4.3.1 and V7.4.4; prerequisite human/data gates cannot be bypassed. **Work Packet:** Sequential. **Write Scope:** integrated evidence/handoffs and task-owned staging; no commit/push/release/publication. **Real Data Dependency:** all named approved real captures and actual outputs; only separately accepted generated scopes with their recorded versions/provenance. **Provider Boundary:** full supported-minor gate, actual installed artifacts, explicit required reference workflow and independent oracle.

**Inputs / Contracts:** whole-plan requirement audit, required450-case job with no required skips, error/coverage sensitivity, performance dispositions, all selected public interfaces and modular optionality, GOTCHA/ATLAS handoffs. **Facts Protected:** full register and original numerical expected values/tolerances. **Description:** Establish completion from integrated authoritative evidence, not positive-only subsets. **Validation / Handoff:** declared common gate, changed-domain coverage target, artifact dependency floors/locked versions and independent final review; report any unobserved remote job as such. **Acceptance Criteria:** every explicit plan requirement verified or honestly still incomplete; stage only task-owned changes for review; mark goal complete only when no required work remains.

V3/V4 rows summarize their expanded tasks above. V8.1 documentation and V8.2 local integrated verification are recorded in implementation-progress.md. Historical authorization prerequisites in individual task descriptions were resolved on 2026-10-04; final oracle disposition is recorded separately. Preserve phase/fact IDs across revisions. Each phase uses the integrated predecessor baseline, a single implementation writer and independent review where numerical/API compatibility is affected.

| Phase / role | Dependencies and target | Facts introduced / protected | Verification and demo command | Observable outcome, risks, rollback, and exit |
| :--- | :--- | :--- | :--- | :--- |
| V3 / Capability: suite lookup | V1; DEC-02. Consistent canonical lookup and approved numeric/mutation semantics | Introduce F05; protect F01/F02, canonical IDs and integer list indexing | E05 and common gate; `uv run --locked pytest tests/test_api.py -k suite` | Cover append/extend/insert/remove/pop/clear, item/slice assignment/deletion, sort/reverse and in-place operators for approved mutable behavior, or explicitly approved narrowed API with migration. Duplicate IDs and mutable function metadata need a declared policy. Prefer simplest coherent lookup rather than an index-maintenance framework; reverting a changed numeric contract needs migration notes. Exit: approved selector matrix green, compatibility documented, stage changes for human review. |
| V4 / Capability: RNG ownership | V2; DEC-03; approved captured streams or exact DEC-01 authorization. Explicit isolated opt-in if selected, preserving legacy defaults unless owner approves a break | Introduce F06; protect F03/F04 and current documented noise formulas | E06 and common gate; `uv run --locked pytest tests/functions/transformations/test_noise_ownership.py` | Library instance isolation and per-mode draw semantics tested. Equal distribution is not equal seeded output; multiple transforms/components and scalar/batch order require explicit guarantees/limits. RNG state thread-safety not promised; no hidden global reseeding. Rollback retains a legacy mode until approved migration closes. Exit: documented modes and sequence boundaries green, stage changes for human review. |
| V4.5 / Data Gate: reference acquisition | V4.3 integrated baseline; approved source/case selection. Acquire DATA-02 before dependent V5/FEAT-02 work | Enable F07 and selected F10; protect existing captures and deviation obligations | V4.5.1–V4.5.3 source/capture verification plus `uv run --locked pytest tests/utils/test_golden_loader.py` | Pin source revisions, licenses, environment and required case IDs; validate hashes and parser compatibility. No new production validation behavior in acquisition. Rollback removes only newly acquired task-owned local capture setup; never discard approved existing data. Exit: provenance approved and required captures accessible, unchanged default suite green, stage changes for human review. |
| V5 / Capability: complete required reference validation | V1, V4.5; DATA-02 approved and accessible. Introduce required manifest semantics with separate RED Test and GREEN Implement tasks before later hardening | Introduce F07; protect deterministic reference behavior and existing deviation obligations | E07 plus existing suite/COCO tests in provisioned integration job; `uv run --locked python scripts/verify_reference_manifest.py` | Missing required captures fail the required job, while optional tests stay explicitly optional locally. Don't force known upstream deviations to match by changing formulas or weakening tolerances. Tests validate manifest sensitivity and case execution, not just file presence. Acquisition, adapter, checker and CI tasks split in revision. Rollback removes only new integration wiring, preserves captures/provenance. Exit: required IDs executed, deviations reported, source licenses/hashes pinned, stage changes for human review. |
| V6 / Probe: performance and numerical evidence | V1/V2; V5 evidence for reference-sensitive numerical candidates; approved workload data and owner-selected budgets | Establish F08 evidence; protect F01–F07 as applicable | E08 proposed captured-input mode and common gate; same command emits report/raw measurements | Compare discovery reuse, weights allocation, MinComposition streaming and measured Gallagher/large-scale hot paths individually. Same revision/interpreter/BLAS, representative scalar/small/large batches, numerical equivalence, trial distributions and RSS/allocation distinctions required. No performance threshold in default pytest. Each candidate accepted/rejected/blocked explicitly; adoption needs a separate protected task pair, not a probe silently editing production. Rollback restores isolated candidate patch only. Exit: evidence-backed dispositions and raw inputs/results retained appropriately, stage changes for human review. |
| V7 / Capability: owner-selected enhancement | V5/V6 as relevant; DEC-04; selected FEAT-01–FEAT-04 only; revised full tasks and required data | Introduce selected F09–F12 only; protect evaluation API, existing factories and reference behavior | Selected E09–E12 and common gate; each selected feature needs its own public-entry-point demo | Choose one enhancement per slice; split this phase by decimal IDs before dispatch if several are chosen. No optimizer framework, backend rewrite, multi-objective retrofit or new dependencies automatically included. Version public schema/API when selected; reject unsupported stateful chunking. Rollback removes opt-in helper/schema without changing objective defaults. Exit: selected contracts green, migration/docs clear, stage changes for human review. |
| V8 / Documentation and delivery hardening | All selected predecessor slices integrated | Protect all accepted facts; strengthen LOCAL-R08 delivery evidence | Common gate, E01 artifact smoke, required E07 job when provisioned; `uv run --locked pytest tests/test_api.py` | Reconcile roadmap/catalog/README duplication, document approved exception and dataset freshness, retain supported-version/dependency-range/artifact checks. Documentation-only coverage N/A; no new API behavior. Future GOTCHA/ATLAS artifacts required for nontrivial implementation; this proposal is not a claim they ran. Rollback task-owned docs/CI only. Exit: integrated facts green, complete scope reviewed, stage changes for human review. |

## Requirements and prioritized changes

| Requirement | Priority and outcome | Confirmed code/configuration location | Decision or protection |
| :--- | :--- | :--- | :--- |
| LOCAL-R01 | High: reliable core-only installation and modular discovery | `__init__.py`, `registry.py`, `loader.py`, `pyproject.toml`, CLI package | Core and CLI installed separately; preserve explicit scanning/registration |
| LOCAL-R02 | Medium: batch failures propagate once without buffer regressions | `functions/transformations/composed.py::evaluate_batch` | Preserve optional `out`; default base row-loop exists and remains supported |
| LOCAL-R03 | Medium: coherent suite lookup and mutation | `loader.py::BenchmarkSuite` | DEC-02 before changing numeric selectors or list semantics |
| LOCAL-R04 | Medium: explicit RNG ownership and reproducibility | `functions/transformations/noise.py`, newer noise modules | DEC-03; isolated opt-in is a proposal, not a silent default change |
| LOCAL-R05 | High: honest numerical validation coverage | Suite validation tests, golden loader, external data environment, CI | Pinned approved references; separate required integration gate |
| LOCAL-R06 | Medium: measured optimization choices | Registry scanning, weighted/min compositions, Gallagher evaluation, benchmark script | Real workload data, budgets, numerical and memory evidence |
| LOCAL-R07 | Proposed: useful enhancements with narrow modular seams | Existing loading, composition, CLI and suite factories; new paths labeled proposed | DEC-04 selects actual scope; feature ranking is provisional |
| LOCAL-R08 | Medium: reproducible verification and accurate documentation | `.github/workflows/ci.yml`, CONTRIBUTING, roadmap and docs copies | Explicit locked extras; preserve required checks and source-of-truth boundaries |

### Confirmed baseline findings

The October 3 review inspected code at the stated baseline and ran checks without production edits. Prior logs lived under `/tmp/pymofl-review`; that temporary location is not durable CI evidence. Before implementation V0 reruns the gate and records a retained result and fixture hashes.

| Finding | Evidence and impact | Standards area |
| :--- | :--- | :--- |
| Core install imports optional CLI | A clean wheel install without extras failed at `scan_package()` → `cli/main.py` → missing `typer`; CLI-enabled clean install imported and evaluated a packaged reference optimum successfully | Packaging and Project Boundaries; import ownership |
| Incomplete external reference execution | 527 CEC reference cases skipped because external captures were unavailable; COCO module skipped because `cocoex` was absent. Total baseline 531 skipped, one xfailed | Testing Standards; honest verification |
| Suite lookup stale after mutation | Removing the actual F1 entry still left `suite['f01']` resolving that removed object; `suite['1']` returned F2 at the tested baseline | API contracts; mutation ownership |
| Child `TypeError` retried | A focused diagnostic of `ComposedFunction.evaluate_batch` observed two child calls for an internal failure; broad retry exists at base and scalar-transform dispatch sites | Error Handling; correctness, substitutability |
| Global RNG ownership inconsistent | Legacy `NoiseTransform` calls `np.random.seed`/`randn`; Gaussian/Uniform/Cauchy transforms own a Generator | Numeric workload and concurrency/reproducibility semantics |
| Locked CI missing | Existing CI uses plain `uv sync --all-extras` and `uv run`; a lock exists but freshness is not enforced. Actions use floating version tags and setup-uv has no reviewed uv release pin | Tooling, Lints, and Standards Enforcement |
| Documentation drift | CI already exists; several SPSO-related functions already exist; README/roadmap suite claims are not a reliable inventory | Scope, verified documentation |

Python 3.12 and 3.14 runs each reported 2,848 passed, 531 skipped, one xfailed and one existing Lunacek 1D runtime warning. Ruff format, Ruff lint and ty passed in the locked development environment. Python 3.13 was not executed in that review. Wheel and source distribution built; the wheel contained approximately 401 MiB of uncompressed constants and was approximately 71 MiB compressed. These sizes motivate measurement of installation/data packaging, not an automatic data download system or asset split.

The clean-wheel experiment separately exercised currently resolved compatible runtime dependencies (NumPy 2.5.3 and matplotlib 3.11.2), while locked tests/benchmarks used NumPy 2.4.2 and matplotlib 3.10.8. Do not conflate those environments or claim complete dependency-range coverage.

### Optimization evidence and dispositions

The retained experiment used Python 3.14.8, NumPy 2.4.2, Linux aarch64, `OPENBLAS_NUM_THREADS=1`, nine alternating trials and 30 repetitions per trial. Inputs were the 100 finite 30D vectors already present as CEC 2005 captured optima, random inputs and bounds; no new representative population was generated. Functions were constructed before timing. Peak allocations came from `tracemalloc`, not total process RSS or comprehensive native allocation measurement. Two workloads only; no end-to-end optimizer or large-batch speedup claim.

| Candidate | Median time, baseline → candidate | Trial ranges, baseline / candidate | Peak tracked allocations | Disposition |
| :--- | :--- | :--- | :--- | :--- |
| GNBG F24, four-component running minimum | 386.260 → 385.505 microseconds | 384.039–413.004 / 382.639–392.755 microseconds | 221,112 → 217,744 bytes | No meaningful speed benefit; keep current implementation absent a separately measured memory need |
| CEC 2014 F23, five-component streamed distances | 114.446 → 120.520 microseconds | 113.569–115.096 / 119.530–120.818 microseconds | 251,040 → 104,885 bytes | About 58% lower tracked peak, about 5% slower; investigate opt-in memory control, not a faster-default claim |
| Repeated classical function loading | About 940 microseconds per Sphere load in an unprofiled 30-call observation | No trial distribution captured | Not measured | Repeated discovery is the first construction candidate to measure; no quantified adoption claim |

Both numerical candidates passed `np.testing.assert_array_equal` on the retained inputs. That comparison does not prove signed-zero bit identity, exceptional-value behavior, all dimensions, all flags or complete suite equivalence. Preserve NaN propagation: use [`numpy.minimum`](https://numpy.org/doc/2.4/reference/generated/numpy.minimum.html) rather than NaN-ignoring `fmin` for a selected streaming implementation, and test signed-zero and exceptional values under approved evidence.

Profiling 30 repeated Sphere loads attributed about 0.131 of 0.132 seconds to `scan_package`; profiling overhead differs from unprofiled timing. The ratio identifies a construction bottleneck in this diagnostic, not a promised 99% latency reduction. Reusing immutable built-in discovery may help, but explicit registration, collisions, mutable configuration and invalidation must still work.

The current weighted-distance path materializes `(N, K, D)` differences and squares. Bounded chunks can control this memory growth. Replacing it with a dot-product identity can suffer cancellation near optima; don't adopt that algebraic rewrite merely to remove allocation. Preserve reduction order where possible, zero-distance behavior, weight underflow fallback, ties, biases and non-continuous mapping. Cache loaded data only if construction profiling justifies it and file freshness plus caller mutation/ownership are defined; no shared mutable array cache by default.

The existing competitive benchmark document lacks enough environment/trial metadata for a repeatable speedup claim, and its script extrapolates scalar competitors from subsets. V6 must record comparable interfaces, function/instance/data identity, numerical agreement, scalar and batch measurements separately, fixed threading, repeats/warmups, variability and memory method. No new Numba/JAX dependency is warranted by these measurements. See the existing [performance table](./performance_benchmarks.md) as historical evidence, not a new acceptance threshold.

## Architecture invariants and feature proposals

### Modular architecture invariants

1. Objectives own mathematical evaluation, bounds remain metadata, transforms own explicit coordinate/output operations, and compositions own ordering and combination. Penalties use raw input where current composition semantics require it.
2. Preserve `OptimizationFunction` scalar API and its existing default batch fallback. Numerical result shape, dtype handling, error paths, input mutation, buffers and tolerances are explicit per selected contract, not standardized by a sweeping coercion change.
3. Keep `FunctionFactory`, builders and declarative suite data as the construction mechanism. Extract a small shared helper only for proven repeated responsibilities; do not create a universal factory, provider framework, plugin manager or deep new inheritance tree.
4. Open/closed and substitutability mean existing registered functions and custom transforms remain usable without mandatory new `out`, backend, capability, or metadata methods. Use optional adapters/helpers when new behavior isn't part of every function's contract.
5. Keep NumPy as the reference implementation. CLI/reporting/validation/optimizer/backend extras do not become numerical core dependencies. No network, optimizer execution or data download on import/evaluation.
6. Preserve current formulas and documented upstream deviations. Numerical performance edits require protected evidence and isolated review; no changes to benchmark constants, bounds enforcement, quantization, RNG algorithms or default ordering as collateral refactoring.
7. Share real duplicated contract dispatch or report schemas at the narrowest common boundary. Similar-looking formulas that differ by benchmark definition remain distinct. Avoid abstractions whose only benefit is making code look uniform.

### Feature menu and selection contracts

The owner selected FEAT-01–FEAT-04 on 2026-10-03; F09–F12 contracts are their acceptance targets. Source-dependent work remains gated on verified references. FEAT-05–FEAT-07 remain research-only with no implementation authorization.

| Feature | Placement and value | Minimum selected contract / modular boundary | Decision and non-goals |
| :--- | :--- | :--- | :--- |
| FEAT-01: bounded deterministic batch helper | Proposed small helper beside numerical evaluation utilities; limits working batch size without changing every function signature | Opt-in, preserve row order/input ownership; specify empty batches, chunk remainder, positive integral chunk size, float64/result shape, valid buffers, overlap/aliasing and exception/partial-write behavior. Metadata or an explicit caller declaration identifies deterministic batch independence; never infer it from class names. Chunking bounds helper input size, not arbitrary custom internals | First version excludes noisy/stateful/batch-dependent functions unless a later explicit semantic contract covers them; deterministic equality may require approved tolerance across changed BLAS batch shapes. No automatic chunking inside every evaluator |
| FEAT-02: validation CLI and structured report | Optional CLI adapter over a library-neutral validation/report helper; exposes observed coverage and deviations | Explicit pinned manifest, stable selected JSON schema, statuses, exit codes, required/optional distinction and dataset freshness. Errors report identifiers/paths without sensitive records; no fabricated passed results | Depends on DATA-02; validation never fetches data implicitly during ordinary test/evaluation runs; no objective math imported from a reference engine |
| FEAT-03: SPSO suite definitions | New declarative suite config reuses Tripod, Network, Gear Train and Compression Spring implementations | Pin suite version/reference, function numbering, dimensions, bounds, quantization, feasibility/penalty and captured outputs; shared builders only when actual duplication warrants it | Existing functions alone do not prove official SPSO equivalence. Acquire source/rights/reference evidence before new claims; do not silently encode historical C#/SPSO bugs as authoritative |
| FEAT-04: reproducibility definition manifest | Small optional versioned export/reconstruction helper over existing configs and data hashes | Identity, suite/function/instance/dimension, actual resolved parameters, constants/config hashes, software/environment and compatibility version. Distinguish original config from defaults resolved by a factory | Deterministic definition replay only initially; noisy RNG state replay is a separate DEC-03 feature, not implied by recording a seed. No storage/server dependency |
| FEAT-05: experiment runner / optimizer bridge | Prefer examples or a companion package that calls pyMOFL's existing evaluation API | If selected later: objective-call protocol, evaluation budgets including batch/failed-call accounting, seeds, target definitions, callbacks/result schema and constraint reporting | Core remains a function library; do not embed PSO algorithms, parallel execution engine, or import the C# repositories into numerical code. Needs a separate owner-approved plan |
| FEAT-06: multi-objective benchmarks | Research a separate vector-objective contract and suite boundary | Define objective shape, Pareto/reference fronts, constraints, metadata and reference evidence before BBOB-biobj support | Do not change scalar `evaluate()` return types or retrofit every subclass; separate design and source acquisition |
| FEAT-07: optional accelerated backend | Evidence-driven optional adapter/extra after V6 | Backend capability limits, dtype/rounding, RNG, compilation/warmup, transfers, supported transforms and numerical tolerance tested against NumPy | No backend abstraction or dependencies until representative workload and owner approval justify them; retain readable NumPy reference |

### Compatibility decisions before deferred work

Numeric string alias corrections can break users even when current behavior is inconsistent. DEC-02 now selects canonical function numbers for numeric strings, ordinary integer indexing, and errors for ambiguous matches; canonical names remain usable. Immutable conversion is not a routine fix to a mutable list subclass. An approved mutable design must consider every mutator, duplicate IDs and metadata changes without duplicating an index-maintenance mechanism at each call site.

Noise transforms are stateful; documentation calling every transform pure/stateless is inaccurate for these objects. DEC-03 distinguishes distributional agreement, per-instance seed repeatability, global-state compatibility, scalar-versus-batch equivalence and chunk invariance. Uniform/Cauchy batch methods draw whole blocks per random variable, so changing chunk sizes can change which draw is paired with each row; multiple noise stages and components introduce additional ordering differences. Preserve each existing RNG default as accepted in DEC-03 and do not promise chunk-invariant noise by merely injecting a Generator.

Output-buffer dispatch must support bufferless subclasses and respect existing publicly mutable composed fields. Signature inspection/binding or a conservative no-buffer adapter can be investigated, but opaque callable/wrapper behavior needs evidence; broad `TypeError` catching is not a correct capability test. Don't keep a stale constructor-time capability cache for later replaced components. API changes to validate invalid output buffers require their own accepted contract, not incidental hardening in V2.

## Standards review and limits

Standards sources are `/home/firestrand/Dropbox/Projects/Software-Standards/Python Code Standards.md` (revision 2026-09-29), `development-plan-creation-guide.md` (2.7), and `Technical Markdown Style Guide.md` (read 2026-10-03; no revision metadata observed). Local [coding guidelines](../CODING_GUIDELINES.md) and [repository guidance](../CLAUDE.md) refine project conventions. Source code wins when descriptive documentation is stale. The Python standard supplies technical constraints; the development guide supplies plan structure; the Markdown guide supplies document formatting, not additional product requirements.

| Applicable group | Current baseline assessment | Proposal requirement |
| :--- | :--- | :--- |
| Supported versions/tooling | Partial: metadata/configuration align on 3.12 minimum; 3.12/3.13 CI exists; 3.13 not locally checked; locked commands, full action SHAs and uv release pin absent | Preserve minimum, run advertised matrix, record uv/build tooling, explicit locked extras, reviewed action SHAs and same reviewed uv release locally/CI; do not migrate runtime just for this plan |
| Public typing/contracts | Partial: ty passes but registry APIs are not fully annotated and some numerical boundaries use broad types | Type touched public/shared APIs honestly; don't claim whole-repository compliance or globally disable diagnostics |
| Packaging | Fail: core-only wheel import failed; src layout, lock, wheel data and py.typed exist | Isolated core/CLI installed artifact checks; dependency range and build constraints tested separately |
| Failure handling | Fail in identified batch dispatch: internal exception retried | Single invocation, cause preservation, focused cleanup and subprocess timeouts |
| Numerical/performance evidence | Partial: broad tests and historical table; representative large-workload/memory evidence absent | Named approved inputs, same environment, full results and numerical/exception contracts before adoption |
| Tests/reference validation | Partial: baseline green but external references skipped | Keep hermetic default suite; provision required references in separate explicit integration gate; no weakened expected values/tolerances |
| Dependencies/security | Unverified beyond inspected local changes; no formal dependency security audit was run | No new runtime libraries without owner acceptance; no secrets/private data, unexpected import-time I/O or generated production-shaped fixtures |
| Concurrency/async/services | N/A to current scoped corrections | Don't add concurrency infrastructure; stateful instances are not declared thread-safe |
| Documentation and scoped exceptions | Partial: roadmap/guidance drift; no approved new exceptions | Reconcile touched claims; scoped SHOULD rationale and owner-approved MUST exceptions if necessary; no agent-approved exceptions |

Approval of this document is a design/plan review. It does not certify the current code against every standard, approve feature selection, or verify implementations. Required future GOTCHA specs and ATLAS reports apply to selected nontrivial implementation, while straightforward fixes follow the repository's proportional verification rules. Release/upload/commit/push authority remains with the invoking implementation instruction.

## Version history

| Version | Date | Change |
| :--- | :--- | :--- |
| 1.0.0 | 2026-10-03 | Initial evidence-backed proposal, modular architecture constraints, rolling-wave correction plan and gated feature menu |
| 1.1.0 | 2026-10-03 | Separate explicit controlled-fault authorization and dependency; require CI action/uv pinning; split reference acquisition from the new validation capability; clarify changed-behavior coverage |
| 1.2.0 | 2026-10-03 | Record owner acceptance of DEC-01–DEC-04 and official reference acquisition; expand suite/RNG slices; preserve the distinct existing noise defaults |
| 1.3.0 | 2026-10-03 | Expand reviewed official CEC2014 acquisition, gated required validation and authorized performance probes; retain local readiness and optimization dispositions without promoting absent evidence |
| 1.4.0 | 2026-10-03 | Align evidence commands/status with actual implementations; separate oracle-approved real producer/validation and optional CLI supported work from unchanged named generated-failure gates; expand selected feature contracts |
| 1.4.1 | 2026-10-04 | Record owner synthetic-data approval; complete selected rejection/boundary gates, scoped coverage and local required workflow; reconcile evidence and record final oracle approval without claiming remote CI or release |
| 1.4.2 | 2026-10-04 | Record owner authorization to commit/push main, bump to 0.4.0 and tag v0.4.0; preserve original baseline/evidence checkpoints |
