---
title: pyMOFL implementation coverage review
version: 1.0.0
last_updated: 2026-10-04
status: oracle-reviewed
owner: product-owner
reviewer: oracle-agent
tags: [verification, coverage, provenance]
---

# Implementation coverage review

The approved proposal requires no legacy regression, at least 90% changed
behavior branch coverage and 95% changed domain logic, with an explicit scoped
justification for unreachable branches. Measurements use coverage.py without
new exclusions, changed tolerances or weakened assertions. The existing eight
excluded statements are inherited. Raw reports remain outside the repository
under `/tmp/pymofl-implementation`.

## Library and artifact measurements

`closure-coverage312.json` measures the full provisioned Python 3.12 suite:
3,427 passed, 272 unavailable other-reference skips, 35 inherited expected
failures and one inherited Lunacek D1 warning. Python 3.13 and 3.14 full runs
have the same results. NumPy is 2.4.2 in these full-suite environments.

| Measure | Baseline | Current |
| :--- | :--- | :--- |
| Statements | 92.62% | 93.64% |
| Branches | 76.65% | 79.78% |
| Combined | 89.47% | 90.82% |

New evaluator, definition, reference validator, CLI adapter, selected SPSO
bases, Absolute, HalfUpQuantization and RNG resolver have 100% measured
statements and branches. Changed loader coverage is 67 executable lines and
all 32 arcs originating on changed branch lines, independently mapped by the
oracle against the unchanged baseline. Whole legacy modules retain inherited
uncovered branches; those do not count as changed behavior.

`smoke-coverage.json` independently measures the exact installed-artifact
embedded smoke: 52 statements and ten branches, all covered across real
core/CLI wheel environments, with and without provisioned captures. This is
secondary instrumentation with coverage.py installed as a development tool;
the separate clean artifact gates establish optionality and dependency floors.
The parent checker measurement does not purport to trace its subprocesses.

## Retained tools

`tools-coverage.json` combines actual source setup, catalog generation,
required-report execution, artifact environments, controlled operational faults
and the profiler parent plus all 48 workers. Instrumented profiling uses one
trial/repetition plus an actual two-trial alternation check. Those timings do
not replace the original nine-trial performance evidence.

| Tool | Statements | Raw branches | Disposition |
| :--- | :--- | :--- | :--- |
| CEC2014 capture | 93.27% | 82.50% (33/40) | Seven pinned-input defenses justified below; all reachable measured branches covered |
| SPSO capture | 98.23% | 95.24% (40/42) | Two pinned-input defenses justified below |
| SPSO boundary capture | 100% | 100% | Pass |
| Profile workloads | 100% | 100% | Pass, parent and workers measured |
| Installed checker | 98.73% | 95.83% | Pass; foreign wheel rejection remains outside measured operational variants |
| Required manifest runner | 100% | 100% | Pass |
| Catalog generator | 100% | 96.43% | Pass; remaining arc is inherited categorization iteration |

Fault outputs are labeled adversarial. Real processes execute first; wrappers
then corrupt results or raise controlled failures. Copies of genuine pinned
sources/captures isolate missing/invalid input and metadata checks. Failures
never become successful reference evidence. The final external harness records
44 checks, including exact profiler CLI diagnostics, plus three actual
installed CLI operational failures. Original sources/expected outputs are
unchanged.

## Scoped unreachable pinned-input defenses

This record applies the already approved proposal's unreachable-branch clause.
It creates no coverage pragma/config exclusions, suppresses no diagnostics,
changes no guards and does not approve a new standards deviation. Raw coverage
above remains visible. The oracle independently approved this exact scoped record on 2026-10-04.

CEC2014 archive identity is
`1a210560398ca7a50be6adf1e5e90602222519ef23b6e31aba8847e109761876`.
Local DATA01 identity is
`5e7268c46c288a287e1909b5986518631ea87f08e16cb2ace0517cc84ef0142a`.
Both are checked before the listed defenses. Actual pinned contents were
inspected and all 450 outputs captured/repeated and source-controlled.

| File and missing arc | Immutable precondition and rationale |
| :--- | :--- |
| `scripts/capture_cec2014.py` 51→52 | Verified archive has no absolute/traversal members; changing one changes the verified archive bytes |
| Same file 61→62 | Verified source has exactly one Windows include and five format patch sites |
| Same file 123→126 | Verified shift files contain required groups and coordinates |
| Same file 127→130 | Verified shift file coordinates are finite |
| Same file 134→137 | Verified matrix files have the fixed required sizes and finite values |
| Same file 146→151 | Verified shuffle groups are complete permutations |
| Same file 159→160 | Verified source shifts and DATA01 selected rows have fixed D10/30/50 sizes |
| `scripts/capture_spso.py` 74→75 | Verified archives have unique safe selected C/header filenames |
| Same file 94→95 | Verified 2011 source has exactly one native-coordinate patch site |

SPSO archive identities are
`f9524f7f9568009b4ab5c76cd32d91c255fef978b4ff64891b460cb520f34bd1`
and `11692f658158b18aafd97d667eeebdc7527cf21147d530d73d4c7eb795af0557`.
The scope expires whenever pins, selected inputs, patch procedure or line
mapping change. Rehash and independently review again in that event.

A foreign/corrupt archive cannot exercise these downstream branches without
bypassing immutable pin verification. We retain both protections and do not
fabricate an archive claimed to have an approved hash. Reachable output,
archive, local DATA01, repeat, diagnostics, incomplete/nonfinite/control-value,
release/UBSan disagreement and subprocess failures are measured separately;
none receives this justification.
