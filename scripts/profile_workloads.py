"""Measure diagnostic workloads and isolated candidates; never change library defaults.

Run with OPENBLAS_NUM_THREADS=1. RSS is a fresh worker's process maximum,
including interpreter/import/inputs; tracemalloc measures tracked allocations.
Generated inputs are authorized by improvement-proposal.md Task V6.0.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import inspect
import json
import os
import platform
import pstats
import resource
import statistics
import subprocess
import sys
import textwrap
import time
import tracemalloc
from pathlib import Path
from types import MethodType

import numpy as np

import pyMOFL
from pyMOFL.compositions.min_composition import MinComposition
from pyMOFL.compositions.weighted_composition import WeightedComposition

WORKLOADS = {"min": ("gnbg", 24), "weights": ("cec2014", 23), "gallagher": ("bbob", 21)}


def _running_minimum(self: MinComposition, X: np.ndarray) -> np.ndarray:
    X = self._validate_batch_input(X)
    result = self.components[0].evaluate_batch(X).copy()
    for component in self.components[1:]:
        np.minimum(result, component.evaluate_batch(X), out=result)
    return result


def _streamed_weight_method():
    """Change only the distance allocation in an isolated copy of the live method."""
    source = textwrap.dedent(inspect.getsource(WeightedComposition._compute_weights_batch))
    original = "diff = X[:, None, :] - self._optima_arr[None, :, :]"
    reduction = "d2 = np.sum(diff**2, axis=2)"
    if source.count(original) != 1 or source.count(reduction) != 1:
        raise ValueError("Live weighted implementation no longer matches the probe baseline")
    source = source.replace(original, "d2 = np.empty((X.shape[0], n), dtype=np.float64)")
    source = source.replace(
        reduction,
        "for i in range(n):\n        diff = X - self._optima_arr[i]\n"
        "        d2[:, i] = np.sum(diff**2, axis=1)",
    )
    namespace = {"np": np}
    exec(compile(source, "<isolated-streamed-weights-probe>", "exec"), namespace)
    return namespace["_compute_weights_batch"]


def _workload(name: str, dimension: int):
    suite, fid = WORKLOADS[name]
    return pyMOFL.load(fid, dimension=dimension, suite=suite, instance=1)


def _candidate(function, name: str):
    target = function
    while hasattr(target, "base_function"):
        target = target.base_function
    if name == "min":
        if not isinstance(target, MinComposition):
            raise TypeError("Expected the actual GNBG MinComposition")
        attribute = "evaluate_batch"
        method = _running_minimum
    elif name == "weights":
        if not isinstance(target, WeightedComposition):
            raise TypeError("Expected the actual CEC2014 WeightedComposition")
        attribute = "_compute_weights_batch"
        method = _streamed_weight_method()
    else:
        return function.evaluate_batch

    def evaluate(X: np.ndarray) -> np.ndarray:
        original = getattr(target, attribute)
        setattr(target, attribute, MethodType(method, target))
        try:
            return function.evaluate_batch(X)
        finally:
            setattr(target, attribute, original)

    return evaluate


def _inputs(function, args) -> tuple[np.ndarray, dict]:
    if args.captured:
        path = args.captures / f"datasets/CEC2014/func_23_D{args.dimension}/golden.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        X = np.array([record["x"] for record in records], dtype=np.float64)
        return X, {
            "kind": "approved CEC2014 captures",
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    if args.worker in {"min", "weights"}:
        root = Path(__file__).resolve().parents[1] / "src/pyMOFL/constants"
        if args.worker == "min":
            config = root / "gnbg/gnbg_suite.json"
            number = 24
        else:
            config = root / "cec/2014/cec2014_suite.json"
            number = 23
        entries = json.loads(config.read_text())["functions"]
        entry = next(e for e in entries if f"_f{number:02d}" in e["id"])
        bounds = entry["search_space"]["default_bounds"]
        low = np.full(args.dimension, bounds["min"], dtype=np.float64)
        high = np.full(args.dimension, bounds["max"], dtype=np.float64)
        bound_source = {
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "entry": entry["id"],
        }
    else:
        low = function.operational_bounds.low
        high = function.operational_bounds.high
        bound_source = {"instance": 1, "source": "actual BBOB operational bounds"}
    generator = np.random.Generator(np.random.PCG64(20261003))
    X = generator.uniform(low, high, (args.rows, args.dimension))
    return X, {
        "kind": "authorized diagnostic model, not production/reference data",
        "authorization": "V6.0",
        "generator": "PCG64",
        "seed": 20261003,
        "range_low": low.tolist(),
        "range_high": high.tolist(),
        "bound_source": bound_source,
    }


def _measure(evaluate, X: np.ndarray, repeats: int) -> float:
    start = time.perf_counter_ns()
    for _ in range(repeats):
        evaluate(X)
    return (time.perf_counter_ns() - start) / (1000 * repeats)


def _tracked_peak(evaluate, X: np.ndarray) -> int:
    tracemalloc.start()
    try:
        evaluate(X)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def _worker(args) -> dict:
    function = _workload(args.worker, args.dimension)
    X, provenance = _inputs(function, args)
    original_X = X.copy()
    baseline = function.evaluate_batch
    candidate = _candidate(function, args.worker)
    expected = baseline(X)
    if args.captured:
        path = args.captures / f"datasets/CEC2014/func_23_D{args.dimension}/golden.jsonl"
        capture_provenance = json.loads((args.captures / "provenance.json").read_text())
        relative = path.relative_to(args.captures).as_posix()
        entry = next(item for item in capture_provenance["captures"] if item["path"] == relative)
        if (
            capture_provenance["archive_sha256"]
            != "1a210560398ca7a50be6adf1e5e90602222519ef23b6e31aba8847e109761876"
            or entry["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("Captured reference provenance no longer matches")
        reference = np.array([json.loads(line)["value"] for line in path.read_text().splitlines()])
        if not np.all(np.abs(expected - reference) < 1e-6 * np.maximum(1.0, np.abs(reference))):
            raise ValueError("CEC2014 captured reference comparison failed")
    np.testing.assert_array_equal(candidate(X), expected)
    np.testing.assert_array_equal(function.evaluate_batch(X), expected)
    np.testing.assert_array_equal(X, original_X)
    scalar = np.array([function.evaluate(row) for row in X[: min(4, len(X))]])
    np.testing.assert_allclose(scalar, expected[: len(scalar)], rtol=1e-12, atol=1e-8)
    baseline(X)
    candidate(X)
    timings = {"baseline": [], "candidate": [], "scalar": []}
    for trial in range(args.trials):
        ordered = [("baseline", baseline), ("candidate", candidate)]
        if trial % 2:
            ordered.reverse()
        for label, evaluate in ordered:
            timings[label].append(_measure(evaluate, X, args.repeats))
        timings["scalar"].append(
            _measure(lambda values: function.evaluate(values[0]), X, args.repeats)
        )
    profile = cProfile.Profile()
    profile.runcall(baseline, X)
    stats = pstats.Stats(profile)
    hot = sorted(stats.stats.items(), key=lambda pair: pair[1][3], reverse=True)[:8]
    np.testing.assert_array_equal(candidate(X), expected)
    np.testing.assert_array_equal(X, original_X)
    return {
        "workload": args.worker,
        "dimension": args.dimension,
        "rows": len(X),
        "input_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
        "inputs": provenance,
        "result_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
        "trial_microseconds": timings,
        "summary_microseconds": {
            key: {"median": statistics.median(values), "min": min(values), "max": max(values)}
            for key, values in timings.items()
        },
        "tracked_peak_bytes": {
            "baseline": _tracked_peak(baseline, X),
            "candidate": _tracked_peak(candidate, X),
        },
        "worker_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rss_scope": "Linux fresh worker lifetime; includes both variants/imports/inputs; not variant-specific delta",
        "value_comparison": "array_equal on finite inputs; not NaN or signed-zero equivalence",
        "hotspots": [
            {"symbol": f"{Path(key[0]).name}:{key[1]}:{key[2]}", "cumulative_seconds": values[3]}
            for key, values in hot
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", choices=WORKLOADS)
    parser.add_argument("--dimension", type=int, default=30)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--captured", action="store_true")
    args = parser.parse_args()
    if args.trials <= 0 or args.repeats <= 0 or args.rows <= 0:
        parser.error("trials/repeats/rows must be positive")
    if args.worker:
        print(json.dumps(_worker(args), allow_nan=False))
        return 0
    if args.report is None:
        parser.error("--report is required for the full probe")
    if args.report.exists():
        parser.error("Report already exists; choose a fresh path")
    report = {
        "schema_version": 1,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "working_source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted((Path(__file__).resolve().parents[1] / "src/pyMOFL").rglob("*.py"))
        },
        "measurement_caveats": [
            "Candidate patch/MethodType/finally-restoration overhead differs from baseline; intrinsic kernel timing/allocation is not isolated.",
            "Min additionally pays uncached signature dispatch at every size.",
            "Gallagher candidate equals baseline; differences are measurement noise.",
            "RSS is a combined-worker lifetime maximum, not per-variant native memory.",
            "Probe alone does not authorize production numerical adoption.",
        ],
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "threads": {
            key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=30
        ).strip(),
        "worktree": "task-modified; candidate patches isolated in this script",
        "trials": args.trials,
        "repeats": args.repeats,
        "blas": np.show_config(mode="dicts"),
        "results": [],
    }
    for name in WORKLOADS:
        for dimension in (10, 30, 50):
            for rows in (1, 4, 100, 1000, 10000):
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--captures",
                    str(args.captures),
                    "--worker",
                    name,
                    "--dimension",
                    str(dimension),
                    "--rows",
                    str(rows),
                    "--trials",
                    str(args.trials),
                    "--repeats",
                    str(args.repeats),
                ]
                run = subprocess.run(
                    command, capture_output=True, text=True, check=True, timeout=180
                )
                report["results"].append(json.loads(run.stdout))
            if name == "weights":
                run = subprocess.run(
                    [*command, "--captured"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=180,
                )
                report["results"].append(json.loads(run.stdout))
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Recorded {len(report['results'])} workloads in {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
