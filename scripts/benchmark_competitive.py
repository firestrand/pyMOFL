#!/usr/bin/env python3
"""
Competitive Throughput Benchmark: pyMOFL vs opfunu & cocoex (COCO).

Benchmarks batch throughput (evaluations per second) and evaluation latency across:
- CEC 2014 & CEC 2017 (against opfunu)
- BBOB (against cocoex compiled C library)
- pyMOFL batch vs pyMOFL row-loop baseline

Usage:
    .venv/bin/python scripts/benchmark_competitive.py [--batch-size 1000] [--repeats 5] [--save-markdown]
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import pyMOFL

warnings.filterwarnings("ignore")

# Try importing competitor libraries
try:
    import opfunu

    HAS_OPFUNU = True
except ImportError:
    HAS_OPFUNU = False

try:
    import cocoex

    HAS_COCOEX = True
except ImportError:
    HAS_COCOEX = False


@dataclass
class BenchmarkResult:
    suite: str
    func_name: str
    dimension: int
    batch_size: int
    pymofl_batch_ms: float
    pymofl_buf_ms: float
    pymofl_loop_ms: float
    competitor_name: str | None
    competitor_ms: float | None
    evals_per_sec_pymofl: float
    evals_per_sec_competitor: float | None
    speedup_vs_competitor: float | None
    speedup_vs_loop: float


def run_benchmark_trial(
    f_pymofl,
    f_competitor,
    competitor_name: str | None,
    suite_name: str,
    func_name: str,
    dim: int,
    batch_size: int,
    repeats: int,
) -> BenchmarkResult:
    np.random.seed(42)
    # Generate random points within reasonable bounds
    bounds = getattr(f_pymofl, "operational_bounds", None)
    if bounds is not None and np.all(np.isfinite(bounds.low)) and np.all(np.isfinite(bounds.high)):
        low = np.maximum(bounds.low, -100.0)
        high = np.minimum(bounds.high, 100.0)
    else:
        low = np.full(dim, -5.0)
        high = np.full(dim, 5.0)

    X = np.random.uniform(low, high, size=(batch_size, dim))
    out_buf = np.empty(batch_size, dtype=np.float64)

    # Warmup
    _ = f_pymofl.evaluate_batch(X[:10])
    if f_competitor is not None:
        if competitor_name and "cocoex" in competitor_name:
            _ = f_competitor(X[0])
        elif competitor_name and "opfunu" in competitor_name:
            _ = f_competitor.evaluate(X[0])

    # 1. pyMOFL evaluate_batch
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = f_pymofl.evaluate_batch(X)
    t_pymofl_batch = ((time.perf_counter() - t0) / repeats) * 1000.0  # ms

    # 2. pyMOFL evaluate_batch with out= buffer
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = f_pymofl.evaluate_batch(X, out=out_buf)
    t_pymofl_buf = ((time.perf_counter() - t0) / repeats) * 1000.0  # ms

    # 3. pyMOFL single-row loop baseline
    # For large batches or slow compositions, run on subset if needed
    loop_subset = min(batch_size, 200)
    t0 = time.perf_counter()
    for _ in range(max(1, repeats // 2)):
        _ = [f_pymofl.evaluate(row) for row in X[:loop_subset]]
    t_pymofl_loop_sub = (time.perf_counter() - t0) / max(1, repeats // 2)
    t_pymofl_loop = (t_pymofl_loop_sub * (batch_size / loop_subset)) * 1000.0  # ms

    # 4. Competitor timing
    t_competitor = None
    if f_competitor is not None:
        comp_subset = min(batch_size, 200)
        t0 = time.perf_counter()
        if competitor_name and "cocoex" in competitor_name:
            for _ in range(max(1, repeats // 2)):
                _ = [f_competitor(row) for row in X[:comp_subset]]
        elif competitor_name and "opfunu" in competitor_name:
            for _ in range(max(1, repeats // 2)):
                _ = [f_competitor.evaluate(row) for row in X[:comp_subset]]
        t_comp_sub = (time.perf_counter() - t0) / max(1, repeats // 2)
        t_competitor = (t_comp_sub * (batch_size / comp_subset)) * 1000.0  # ms

    evals_per_sec_pymofl = batch_size / (t_pymofl_batch / 1000.0)
    evals_per_sec_comp = (
        (batch_size / (t_competitor / 1000.0)) if t_competitor is not None else None
    )

    speedup_vs_comp = (t_competitor / t_pymofl_batch) if t_competitor is not None else None
    speedup_vs_loop = t_pymofl_loop / t_pymofl_batch

    return BenchmarkResult(
        suite=suite_name,
        func_name=func_name,
        dimension=dim,
        batch_size=batch_size,
        pymofl_batch_ms=t_pymofl_batch,
        pymofl_buf_ms=t_pymofl_buf,
        pymofl_loop_ms=t_pymofl_loop,
        competitor_name=competitor_name,
        competitor_ms=t_competitor,
        evals_per_sec_pymofl=evals_per_sec_pymofl,
        evals_per_sec_competitor=evals_per_sec_comp,
        speedup_vs_competitor=speedup_vs_comp,
        speedup_vs_loop=speedup_vs_loop,
    )


def run_all_benchmarks(batch_size: int, repeats: int) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    # -------------------------------------------------------------
    # 1. CEC 2014 Benchmarks (vs opfunu)
    # -------------------------------------------------------------
    cec14_targets = [
        ("cec14_f01", "F1: Rotated High Conditioned Elliptic", 10, "F12014"),
        ("cec14_f01", "F1: Rotated High Conditioned Elliptic", 30, "F12014"),
        ("cec14_f04", "F4: Shifted & Rotated Rosenbrock", 10, "F42014"),
        ("cec14_f04", "F4: Shifted & Rotated Rosenbrock", 30, "F42014"),
        ("cec14_f17", "F17: Hybrid Function 1", 10, "F172014"),
        ("cec14_f17", "F17: Hybrid Function 1", 30, "F172014"),
        ("cec14_f23", "F23: Composition Function 1", 10, "F232014"),
        ("cec14_f23", "F23: Composition Function 1", 30, "F232014"),
    ]

    for fid, label, dim, opfunu_cls in cec14_targets:
        print(f"Benchmarking CEC 2014 {label} (D={dim})...", flush=True)
        f_pymofl = pyMOFL.load(f"cec2014_{fid.split('_')[1]}", dimension=dim)
        f_comp = None
        if HAS_OPFUNU:
            try:
                cls = getattr(opfunu.cec_based.cec2014, opfunu_cls)
                f_comp = cls(ndim=dim)
            except Exception as e:
                print(f"  opfunu load failed for {opfunu_cls}: {e}")

        res = run_benchmark_trial(
            f_pymofl=f_pymofl,
            f_competitor=f_comp,
            competitor_name="opfunu" if f_comp is not None else None,
            suite_name="CEC 2014",
            func_name=f"{label} ({fid})",
            dim=dim,
            batch_size=batch_size,
            repeats=repeats,
        )
        results.append(res)

    # -------------------------------------------------------------
    # 2. CEC 2017 Benchmarks (vs opfunu)
    # -------------------------------------------------------------
    cec17_targets = [
        ("cec17_f01", "F1: Shifted & Rotated Bent Cigar", 10, "F12017"),
        ("cec17_f01", "F1: Shifted & Rotated Bent Cigar", 30, "F12017"),
        ("cec17_f05", "F5: Shifted & Rotated Rastrigin", 10, "F52017"),
        ("cec17_f05", "F5: Shifted & Rotated Rastrigin", 30, "F52017"),
        ("cec17_f11", "F11: Hybrid Function 1", 10, "F112017"),
        ("cec17_f11", "F11: Hybrid Function 1", 30, "F112017"),
        ("cec17_f21", "F21: Composition Function 1", 10, "F212017"),
        ("cec17_f21", "F21: Composition Function 1", 30, "F212017"),
    ]

    for fid, label, dim, opfunu_cls in cec17_targets:
        print(f"Benchmarking CEC 2017 {label} (D={dim})...", flush=True)
        f_pymofl = pyMOFL.load(f"cec2017_{fid.split('_')[1]}", dimension=dim)
        f_comp = None
        if HAS_OPFUNU:
            try:
                cls = getattr(opfunu.cec_based.cec2017, opfunu_cls)
                f_comp = cls(ndim=dim)
            except Exception as e:
                print(f"  opfunu load failed for {opfunu_cls}: {e}")

        res = run_benchmark_trial(
            f_pymofl=f_pymofl,
            f_competitor=f_comp,
            competitor_name="opfunu" if f_comp is not None else None,
            suite_name="CEC 2017",
            func_name=f"{label} ({fid})",
            dim=dim,
            batch_size=batch_size,
            repeats=repeats,
        )
        results.append(res)

    # -------------------------------------------------------------
    # 3. BBOB Benchmarks (vs cocoex official C-extension)
    # -------------------------------------------------------------
    bbob_targets = [
        (1, "F1: Sphere", 10),
        (1, "F1: Sphere", 20),
        (8, "F8: Rosenbrock", 10),
        (8, "F8: Rosenbrock", 20),
        (15, "F15: Rastrigin", 10),
        (15, "F15: Rastrigin", 20),
        (21, "F21: Gallagher 101 Peaks", 10),
        (21, "F21: Gallagher 101 Peaks", 20),
    ]

    coco_suite = cocoex.Suite("bbob", "", "") if HAS_COCOEX else None

    for f_idx, label, dim in bbob_targets:
        print(f"Benchmarking BBOB {label} (D={dim})...", flush=True)
        f_pymofl = pyMOFL.load(f"bbob_f{f_idx:02d}", dimension=dim, instance=1)
        f_comp = None
        if coco_suite is not None:
            try:
                f_comp = coco_suite.get_problem_by_function_dimension_instance(f_idx, dim, 1)
            except Exception as e:
                print(f"  cocoex load failed for F{f_idx} D{dim}: {e}")

        res = run_benchmark_trial(
            f_pymofl=f_pymofl,
            f_competitor=f_comp,
            competitor_name="cocoex (C)" if f_comp is not None else None,
            suite_name="BBOB",
            func_name=label,
            dim=dim,
            batch_size=batch_size,
            repeats=repeats,
        )
        results.append(res)

    return results


def format_markdown_table(results: list[BenchmarkResult], batch_size: int) -> str:
    lines = []
    lines.append(
        f"### Competitive Benchmark: pyMOFL vs Competitors (Batch Size $N={batch_size}$)\n"
    )
    lines.append(
        "| Suite | Problem Function | Dim | Competitor | Competitor Latency | pyMOFL Batch | pyMOFL Throughput | Speedup vs Competitor | Speedup vs Python Loop |"
    )
    lines.append("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for r in results:
        comp_str = r.competitor_name or "N/A"
        comp_lat = f"{r.competitor_ms:6.2f} ms" if r.competitor_ms is not None else "-"
        pymofl_lat = f"{r.pymofl_batch_ms:6.2f} ms"
        throughput = f"{r.evals_per_sec_pymofl / 1e6:5.2f} M/s"
        speedup_comp = (
            f"**{r.speedup_vs_competitor:5.1f}x**" if r.speedup_vs_competitor is not None else "-"
        )
        speedup_loop = f"{r.speedup_vs_loop:5.1f}x"

        lines.append(
            f"| **{r.suite}** | {r.func_name} | {r.dimension} | {comp_str} | {comp_lat} | **{pymofl_lat}** | {throughput} | {speedup_comp} | {speedup_loop} |"
        )

    return "\n".join(lines)


def print_ascii_table(results: list[BenchmarkResult], batch_size: int) -> None:
    header = (
        f"{'Suite':<10} | {'Function':<34} | {'Dim':<4} | {'Competitor':<12} | "
        f"{'Comp (ms)':<9} | {'pyMOFL (ms)':<11} | {'Throughput':<11} | {'Comp Speedup':<12} | {'Loop Speedup':<12}"
    )
    sep = "=" * len(header)
    print("\n" + sep)
    print(f"COMPETITIVE THROUGHPUT BENCHMARK (Batch Size N = {batch_size})")
    print(sep)
    print(header)
    print("-" * len(header))

    current_suite = ""
    for r in results:
        if r.suite != current_suite:
            if current_suite != "":
                print("-" * len(header))
            current_suite = r.suite

        comp_str = r.competitor_name or "N/A"
        comp_lat = f"{r.competitor_ms:6.2f} ms" if r.competitor_ms is not None else "-"
        pymofl_lat = f"{r.pymofl_batch_ms:6.2f} ms"
        throughput = f"{r.evals_per_sec_pymofl / 1e6:5.2f} M/s"
        speedup_comp = (
            f"{r.speedup_vs_competitor:5.1f}x" if r.speedup_vs_competitor is not None else "-"
        )
        speedup_loop = f"{r.speedup_vs_loop:5.1f}x"

        print(
            f"{r.suite:<10} | {r.func_name[:34]:<34} | {r.dimension:<4} | {comp_str:<12} | "
            f"{comp_lat:<9} | {pymofl_lat:<11} | {throughput:<11} | {speedup_comp:<12} | {speedup_loop:<12}"
        )
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="pyMOFL Competitive Benchmark Suite")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size N (default: 1000)")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repeats (default: 5)")
    parser.add_argument(
        "--save-markdown", action="store_true", help="Save markdown table to docs/benchmarks.md"
    )
    args = parser.parse_args()

    print(f"Starting competitive benchmarks (N={args.batch_size}, repeats={args.repeats})...")
    results = run_all_benchmarks(batch_size=args.batch_size, repeats=args.repeats)

    print_ascii_table(results, batch_size=args.batch_size)

    if args.save_markdown:
        md = format_markdown_table(results, batch_size=args.batch_size)
        out_path = Path("docs/performance_benchmarks.md")
        out_path.write_text(md)
        print(f"\nSaved Markdown benchmark report to {out_path.resolve()}")


if __name__ == "__main__":
    main()
