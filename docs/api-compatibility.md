---
title: pyMOFL API compatibility notes
version: 1.0.0
last_updated: 2026-10-04
status: locally-verified
owner: product-owner
tags: [api, compatibility, reproducibility]
---

# API compatibility notes

## Batch failures and output buffers

Compositions invoke each base/output batch stage once. An internal `TypeError`
propagates unchanged, including its cause, instead of triggering a second call.
Bufferless implementations still work: the composition copies their result into
the supplied output buffer. Opaque callables use this bufferless path.

Dispatch reads current component methods, so replacing a base function or an
output transform remains supported. Signature decisions are weakly cached only
for current class-declared Python methods; instance replacements are inspected
live. This does not cache evaluation results or hold component instances alive.

## Mutable suite lookup migration

Integer indexing and slices keep ordinary Python list semantics. Numeric strings
now identify canonical benchmark numbers, independently of list position:

```python
import pyMOFL

suite = pyMOFL.get_suite("bbob", dimension=2, instance=1)
assert suite["1"] is suite["f01"]  # F1
assert suite[1] is suite["f02"]    # second list entry
suite.reverse()
assert suite["1"].function_id == "bbob_f01"
```

Callers that used numeric strings as positions should pass integer indexes.
The former zero-based and one-based string aliases overlapped; that behavior
could silently select another entry and is intentionally removed.

Canonical IDs, qualified short IDs, names, and short codes resolve against
current entries and current metadata after all list mutations. Unknown names
raise `KeyError`. Multiple matching entries raise `ValueError`, including a
repeated copy of the same function. `suite.get()` returns its default for missing
names/out-of-range indexes, but lets ambiguity errors propagate. Full IDs from a
different benchmark family do not match merely because their numbers agree.

## Explicit random streams

All four noise transforms accept keyword-only `rng=np.random.Generator(...)`.
Providing both `seed` and `rng` raises `ValueError`; an incompatible RNG type
raises `TypeError`. Programmatic `TransformBuilder` parameters can also contain
the Generator object. Live Generator objects are not JSON configuration values.

```python
import numpy as np
from pyMOFL.functions.transformations import NoiseTransform

noise = NoiseTransform(rng=np.random.default_rng(20261003))
value = noise(100.0)
```

The older CEC `NoiseTransform` retains global `np.random` behavior without an
explicit Generator; its legacy `seed` still reseeds that global state. Gaussian,
Uniform, and Cauchy transforms retain their existing per-instance
`default_rng(seed)` defaults. Explicit injection changes no global RNG state.

Separate streams with equal algorithm/seed replay the same sequence when call
shapes and order agree. Sharing a Generator intentionally shares its advancing
state. Uniform and Cauchy batch methods draw blocks for each random variable, so
scalar calls and differently chunked batches need not assign the same draws to
each row. Multiple noisy stages also affect ordering. Recording only a seed is
not a guarantee of noisy-state replay or thread-safe concurrent evaluation.
## Opt-in bounded evaluation

`pyMOFL.evaluate_chunks(function, X, chunk_size, *, deterministic,
batch_independent, out=None)` limits rows per child call. The caller must
establish both properties; declarations do not detect state or make noisy
functions reproducible across different call patterns. Existing evaluation
methods keep their behavior.

Input and child results must be integer or floating NumPy arrays; each input
chunk is copied/cast to float64, and results are copied/cast into a float64
output. Boolean, complex, object and string arrays are rejected. Output buffers
must be writable, shape `(N,)`, dtype float64, with elements that do not overlap
each other or the input. Ordinary strided output buffers are supported.
Empty input returns without evaluating. Exceptions propagate once; completed
output chunks remain written on failure. Only helper input copies are bounded;
custom objective internals and the returned `(N,)` array still consume memory.

For a function known to satisfy both properties, call
`pyMOFL.evaluate_chunks(function, X, 100, deterministic=True, batch_independent=True)`.

## Selected SPSO distributions

`pyMOFL.get_suite("spso2011")` and `pyMOFL.get_suite("spso2007")` provide four
selected source definitions: F04 Tripod (D2), F11 Network (D42), F18 Gear (D4),
and F21 Spring (D3). Omit a suite-wide dimension. Individual loads such as
`pyMOFL.load("spso2011_f21")` use the entry's fixed dimension; an explicit
dimension must match it.

These are native-coordinate objective distances from the pinned source target.
Bounds describe the source-equivalence domain and do not clip or enforce it.
Spring uses [N,D,d], source constraints and multiplicative penalties. Its 2007
variant explicitly preserves the distribution's historical g2 multiplier bug;
2011 uses the corrected multiplier. Spring has no certified optimum point in
the acquired source, so get_global_minimum raises NotImplementedError.
Existing generic Tripod, Network, GearTrain, CompressionSpring and Quantized
aliases retain their behavior. [Source review](./spso-reference-review.md)
records versions, actual captures, rights limits and discrepancies.

`FunctionFactory.create_function(config, fixed_dimension=D)` explicitly adapts
a fixed constructor: conflicting original dimensions or composition delegation
are rejected; the constructed base dimension is verified. Existing calls
without this keyword retain their dimension-injection behavior. The selected
suite loader passes it from entry metadata.

## Deterministic definition records

The optional `pyMOFL.definition` module exports a fresh supported suite
definition and reconstructs it through the existing loader:

```python
import json
from pyMOFL.definition import export_definition, reconstruct_definition

record = export_definition("cec2005_f01", suite="cec2005", dimension=10)
replayed = reconstruct_definition(json.loads(json.dumps(record, allow_nan=False)))
```

Version 1 supports deterministic CEC2005, CEC2014, GNBG, BBOB noiseless and
the selected SPSO suites. It records the original configuration separately
from observed parameters and bounds of the freshly constructed definition.
BBOB records actual instance 1 when omitted; its generated configuration has
no bundled data artifacts. Other families reject an instance selector. GNBG
provenance follows the existing factory's selected configuration.

Numeric array observations carry dtype, shape and base64 bytes, including
infinite bound metadata. Stored parameters and paths are audit observations;
reconstruction regenerates the definition from the canonical request and
checks its current configuration, parameters and package-derived hashes.
It never imports classes or opens paths selected by the record.

Replay requires matching numerical code, selected artifacts, package/NumPy
versions, Python major/minor and platform system/machine. Python patch version
is informational. These identities do not guarantee identical results across
changed BLAS or runtime environments, and hashes do not establish authenticity.
Noisy definitions, current mutable instances, RNG state and custom constructor
arguments are outside this contract. Core import does not load this helper.
Supported and owner-authorized corrupt/incompatible-record checks pass. The
definition helper has 100% measured statements and branches in the integrated
Python 3.12 suite; these tests establish the named replay contract.

Additional breadth checks replay all 109 deterministic configuration entries
at the selected dimensions: CEC2005/CEC2014 D10, GNBG D30, BBOB D2/instance1,
and the selected SPSO native dimensions. The two explicit noisy CEC2005
entries (F04 and F17) are excluded. With retained captures provisioned, 61
requests also replay unchanged coordinates; the other 48 verify metadata only.
This does not establish independent benchmark accuracy or all dimensions and
instances. Unsupported state continues to fail explicitly.
