---
title: pyMOFL performance probe and optimization decisions
version: 1.0.0
last_updated: 2026-10-03
status: reviewed
owner: product-owner
tags: [performance, profiling, optimization]
---

# Performance probe and optimization decisions

The V6 probe recorded 48 workloads with no production numerical edits. It used
Python 3.14.8, NumPy 2.4.2, Linux aarch64, OPENBLAS_NUM_THREADS=1, two warmup
calls, nine alternating baseline/candidate trials and three calls per trial.
Construction was excluded. Scalar timings were separate; each large population
had scalar/batch agreement checked on its first four rows. Full candidate/baseline
outputs and input ownership were checked before and after measurement.

Seeded diagnostic inputs use PCG64(20261003), under owner-approved Task V6.0,
at 1/4/100/1000/10000 rows and D10/30/50. Finite ranges come from actual suite
configuration bounds (GNBG [-5,5], CEC2014 [-100,100]) and actual BBOB bounds
([-5,5]). These are diagnostic uniform populations, not production distributions.
Three further CEC2014 F23 workloads reuse all five approved reference inputs at
each dimension and check the unchanged reference tolerance.

The [retained report](./evidence/performance-v6.json) includes raw trial values,
environment/BLAS metadata, input/result/config hashes and 195 working Python
source hashes. HEAD alone cannot identify this uncommitted worktree. The measured
script SHA256 is `1506a01d70efe902a6230c516cf9331c246c4f77715c1fd592539bc8af11a231`;
the raw report SHA256 is `fead9dced7882104acd0dcd8626ba2fb6078427d120f34592ed63e16a7459e0c`.
The retained script adds provenance/caveats without changing numerical candidates
or timing loops; both source identities are recorded.

## Measurement limits

Candidate calls include temporary MethodType/attribute patching and restoration;
baseline calls do not. The Min candidate additionally pays conservative uncached
signature inspection through ComposedFunction. These costs affect latency and
tracked allocations at every size. Results characterize candidate execution under
this harness, not isolated kernel performance, and cannot alone justify adoption.

Peak bytes are tracemalloc-tracked allocations, not comprehensive native memory.
RSS is recorded as each fresh worker's lifetime maximum, including imports, inputs
and both variants; it is not a per-variant RSS delta. Gallagher has no candidate:
its two timing labels call the same baseline and differ only through measurement
noise. Numerical agreement on finite workloads does not prove NaN, signed-zero,
all flags/dimensions or complete reference-suite equivalence. No timing thresholds
were added to tests.

## Decisions

- Discovery reuse: retain the independently verified V1 change. Its nine-trial
  construction probe measured median 939.35 → 3.65 microseconds per Sphere load,
  with alias/late-registration protection. This is construction latency only.
- MinComposition running minimum: reject a default rewrite on current evidence.
  Component evaluation dominates, tracked memory savings are modest, and the
  harness does not demonstrate a consistent practical improvement. Keep readable
  existing mathematics; no exceptional-value contract is inferred from this probe.
- Weighted streamed distances: reject a universal default replacement. At D30,
  100 rows measured 115.09 → 121.93 microseconds and 251,040 → 104,376 tracked
  peak bytes. At 10,000 rows it measured 9,241.86 → 7,422.23 microseconds and
  24,401,352 → 10,241,976 bytes. Larger workloads merit further investigation,
  but small-batch regression, harness costs and incomplete reference/edge evidence
  gate any production adoption. Implement selected opt-in bounded chunking first;
  do not replace distances with a cancellation-prone dot-product identity.
- Gallagher: retain the current implementation; profile only. D30/10,000 rows
  measured median 114,030.70 microseconds with 7,587,704 tracked peak bytes.
  Missing COCO reference and exceptional-value evidence prevent backend or
  numerical rewrite claims. No accelerator dependency is added.

Hard performance targets were deferred by the owner until measurement. No new
algorithm was adopted here, so no arbitrary global speed target is needed for
acceptance. A later targeted optimization needs a measured workload budget,
equivalent measurement adapters and its own protected Test/Implement pair.

Reproduce with an approved local capture root and a fresh report path:

```bash
OPENBLAS_NUM_THREADS=1 uv run --locked python scripts/profile_workloads.py --captures /tmp/pymofl-implementation/official-captures --report /tmp/pymofl-performance-rerun.json
```

The script is a Linux diagnostic tool; its RSS units and resource API are scoped
to that platform. It does not fetch references or import competitor packages. It restores
candidate methods in finally blocks; worker processes have 180-second timeouts.
These local observations do not establish performance on another CPU/BLAS/runtime
or an end-to-end optimizer.
