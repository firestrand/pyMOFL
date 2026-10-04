# Changelog

## 0.4.1 — 2026-10-04

- Fix the explicit reference-validation workflow: initialize runner temporary paths in a shell step through `GITHUB_ENV`, where runner variables are available.
- Preserve the published v0.4.0 tag; this patch includes the corrected workflow.

## 0.4.0 — 2026-10-04

### Added

- Opt-in `evaluate_chunks` for explicitly deterministic, batch-independent functions, with bounded input batches and documented ownership and partial-write behavior.
- Optional reference manifest validation, strict JSON reports, and `pymofl validate` with honest execution counts and failure exits.
- Selected native-coordinate SPSO 2007/2011 definitions: Tripod, Network, Gear Train and Compression Spring; source half-up quantization and the historical 2007 Spring penalty variant are explicit.
- Deterministic definition export/replay with configuration, artifact and software identity checks; unsupported mutable/noisy state fails explicitly.
- Reproducible source capture, workload profiling and installed-package checks, plus an explicit pinned-source reference-validation workflow.

### Fixed

- Core-only installed imports no longer require CLI or documentation extras; built-in discovery is cached while explicit scanning and registration remain available.
- Batch dispatch propagates internal exceptions after one invocation and preserves valid buffers, evaluation order and input ownership.
- Suite lookup reflects current mutable contents and rejects ambiguous matches.
- Noise transforms accept explicit NumPy generators while preserving existing default RNG behavior.

### Compatibility and evidence

- Numeric strings select canonical function numbers; integer suite indexes retain Python list behavior.
- Definition records require a matching package version; regenerate records for 0.4.0 rather than replaying records from a different release.
- Generic benchmark defaults and numerical optimization strategies remain unchanged. Measured broad min/weight rewrites were not adopted.
- Selected CEC2014 validation covers F1–F30 at D10/30/50, five cases each. SPSO evidence covers the four named definitions, original controls and 546 source boundary cases; it does not establish all-suite accuracy.
- Documentation includes API compatibility, source/provenance limitations, performance findings and reviewed coverage evidence.
