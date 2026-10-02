"""Evaluation command for pyMOFL benchmark functions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import pyMOFL
from pyMOFL.cli.common import (
    clean_function_name,
    find_suite_metadata,
    format_float,
    format_vector,
    resolve_optimum,
)
from pyMOFL.core.function import OptimizationFunction

console = Console()


def eval_fn(
    ctx: typer.Context,
    function: Annotated[
        str,
        typer.Argument(
            help="Function identifier, alias, or suite code (e.g. 'sphere', 'cec17_f01', 'bbob_f01')"
        ),
    ],
    dimension: Annotated[
        int | None,
        typer.Option("--dimension", "-d", help="Dimension to instantiate"),
    ] = None,
    suite: Annotated[
        str | None,
        typer.Option("--suite", "-s", help="Suite identifier if ambiguous"),
    ] = None,
    instance: Annotated[
        int | None,
        typer.Option("--instance", "-i", help="Instance ID for randomized suites"),
    ] = None,
    x: Annotated[
        str | None,
        typer.Option("--x", "-x", help="Comma-separated coordinate values (e.g. '1.0, 2.0, 3.0')"),
    ] = None,
    optimum: Annotated[
        bool,
        typer.Option("--optimum", help="Evaluate at the known global optimum point x*"),
    ] = False,
    zeros: Annotated[
        bool,
        typer.Option("--zeros", help="Evaluate at x = 0 (origin)"),
    ] = False,
    ones: Annotated[
        bool,
        typer.Option("--ones", help="Evaluate at x = 1"),
    ] = False,
    random: Annotated[
        bool,
        typer.Option("--random", help="Evaluate at a single uniform random point within bounds"),
    ] = False,
    random_batch: Annotated[
        int | None,
        typer.Option(
            "--random-batch",
            help="Evaluate a batch of N random points and report summary statistics",
        ),
    ] = None,
    points_file: Annotated[
        Path | None,
        typer.Option(
            "--points-file", help="Path to text, CSV, or JSON file containing input points"
        ),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit detailed JSON representation"),
    ] = False,
) -> None:
    """Evaluate a benchmark function at specified point(s) or randomized batches."""
    output_json = output_json or bool(getattr(ctx, "obj", {}).get("json", False))

    # Parse point x if provided as string to infer dimension
    parsed_x: np.ndarray | None = None
    if x is not None:
        try:
            parts = [float(v.strip()) for v in x.split(",") if v.strip()]
            if not parts:
                raise ValueError("Empty coordinate string")
            parsed_x = np.asarray(parts, dtype=np.float64)
            if dimension is None:
                dimension = len(parsed_x)
        except Exception as exc:
            err = f"Invalid coordinate string '--x {x}': {exc}"
            if output_json:
                console.print_json(data={"error": err})
                raise typer.Exit(code=1) from exc
            console.print(f"[bold red]Error:[/] {err}")
            raise typer.Exit(code=1) from exc

    # Instantiate function
    func: OptimizationFunction
    try:
        func = pyMOFL.load(function, dimension=dimension, suite=suite, instance=instance)
    except ValueError as exc:
        if dimension is None and "requires a 'dimension' parameter" in str(exc):
            func = pyMOFL.load(function, dimension=10, suite=suite, instance=instance)
        else:
            if output_json:
                console.print_json(data={"error": str(exc)})
                raise typer.Exit(code=1) from exc
            console.print(f"[bold red]Error:[/] Could not load function '{function}': {exc}")
            raise typer.Exit(code=1) from exc
    except Exception as exc:
        if output_json:
            console.print_json(data={"error": str(exc)})
            raise typer.Exit(code=1) from exc
        console.print(f"[bold red]Error:[/] Could not load function '{function}': {exc}")
        raise typer.Exit(code=1) from exc

    dim = func.dimension

    # Retrieve known optimum value and point
    _s_name, suite_entry = find_suite_metadata(function, explicit_suite=suite)
    known_x_opt, known_f_opt = resolve_optimum(func, suite_entry=suite_entry)

    display_name = clean_function_name(
        getattr(func, "name", None), getattr(func, "function_id", function), func.__class__.__name__
    )

    # 1. Batch mode: random_batch or points_file
    if random_batch is not None or points_file is not None:
        X_batch: np.ndarray
        if random_batch is not None:
            if random_batch <= 0:
                raise typer.BadParameter("--random-batch must be > 0")
            low = np.where(
                np.isfinite(func.operational_bounds.low), func.operational_bounds.low, -100.0
            )
            high = np.where(
                np.isfinite(func.operational_bounds.high), func.operational_bounds.high, 100.0
            )
            X_batch = np.random.uniform(low, high, size=(random_batch, dim))
        else:
            assert points_file is not None
            if not points_file.exists():
                raise typer.BadParameter(f"Points file not found: {points_file}")
            content = points_file.read_text().strip()
            if content.startswith("["):
                raw_data = json.loads(content)
                X_batch = np.asarray(raw_data, dtype=np.float64)
            else:
                rows = []
                for line in content.splitlines():
                    clean_line = line.strip()
                    if clean_line and not clean_line.startswith("#"):
                        rows.append(
                            [float(val.strip()) for val in clean_line.split(",") if val.strip()]
                        )
                X_batch = np.asarray(rows, dtype=np.float64)

            if X_batch.ndim == 1:
                X_batch = X_batch.reshape(1, -1)
            if X_batch.shape[1] != dim:
                raise typer.BadParameter(
                    f"Points file has dimension {X_batch.shape[1]}, expected {dim}"
                )

        t0 = time.perf_counter()
        Y_batch = func.evaluate_batch(X_batch)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        n_pts = len(X_batch)
        throughput = (n_pts / (dt_ms / 1000.0)) if dt_ms > 0 else float("inf")
        best_idx = int(np.argmin(Y_batch))
        min_val = float(Y_batch[best_idx])
        max_val = float(np.max(Y_batch))
        mean_val = float(np.mean(Y_batch))
        std_val = float(np.std(Y_batch))
        best_x = X_batch[best_idx]

        if output_json:
            console.print_json(
                data={
                    "function": getattr(func, "function_id", function),
                    "name": display_name,
                    "dimension": dim,
                    "count": n_pts,
                    "batch_latency_ms": round(dt_ms, 4),
                    "throughput_evals_per_sec": round(throughput, 2),
                    "best_value": min_val,
                    "best_point": best_x.tolist(),
                    "mean_value": mean_val,
                    "max_value": max_val,
                    "std_value": std_val,
                    "known_optimum_value": known_f_opt,
                }
            )
            return

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold cyan", width=22)
        table.add_column("Value")
        table.add_row("Evaluated Points", f"{n_pts:,}")
        table.add_row("Best f(x) (Min)", f"[bold green]{format_float(min_val)}[/]")
        table.add_row("Mean f(x)", format_float(mean_val))
        table.add_row("Max f(x)", format_float(max_val))
        table.add_row("Std Deviation", format_float(std_val))
        table.add_row("Best Point x_best", format_vector(best_x))
        if known_f_opt is not None:
            table.add_row("Known Optimum f(x*)", format_float(known_f_opt))
            table.add_row("Best Error |f - f*|", format_float(abs(min_val - known_f_opt)))
        table.add_row("Batch Time", f"{dt_ms:.3f} ms")
        table.add_row("Throughput", f"[bold yellow]{throughput / 1e6:.2f} M evals/sec[/]")

        panel = Panel(
            table,
            title=f"[bold]Batch Evaluation: [cyan]{display_name}[/bold]",
            subtitle=f"D={dim} | N={n_pts:,}",
            border_style="cyan",
        )
        console.print(panel)
        return

    # 2. Single point mode
    point: np.ndarray
    point_description: str
    if parsed_x is not None:
        point = parsed_x
        point_description = "Specified vector (--x)"
    elif optimum:
        if known_x_opt is None:
            raise typer.BadParameter(
                f"Global optimum location x* is not analytically defined for '{function}'"
            )
        point = known_x_opt
        point_description = "Global optimum x*"
    elif ones:
        point = np.ones(dim, dtype=np.float64)
        point_description = "Vector of ones (1.0)"
    elif random:
        low = np.where(
            np.isfinite(func.operational_bounds.low), func.operational_bounds.low, -100.0
        )
        high = np.where(
            np.isfinite(func.operational_bounds.high), func.operational_bounds.high, 100.0
        )
        point = np.random.uniform(low, high)
        point_description = "Uniform random in bounds"
    else:
        point = np.zeros(dim, dtype=np.float64)
        point_description = "Vector of zeros (0.0) [default]"

    if len(point) != dim:
        raise typer.BadParameter(f"Input point dimension is {len(point)}, expected {dim}")

    t0 = time.perf_counter()
    y_val = float(func(point))
    dt_ms = (time.perf_counter() - t0) * 1000.0

    err_to_opt: float | None = None
    if known_f_opt is not None:
        err_to_opt = abs(y_val - known_f_opt)

    if output_json:
        console.print_json(
            data={
                "function": getattr(func, "function_id", None) or function,
                "name": display_name,
                "dimension": dim,
                "point": point.tolist(),
                "point_description": point_description,
                "value": y_val,
                "latency_ms": round(dt_ms, 5),
                "known_optimum_value": known_f_opt,
                "error_to_optimum": err_to_opt,
            }
        )
        return

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Property", style="bold cyan", width=22)
    table.add_column("Value")
    table.add_row("Function", f"[bold]{display_name}[/]")
    table.add_row("Dimension", f"{dim}")
    table.add_row("Input Point x", f"{format_vector(point)} ({point_description})")
    table.add_row("Output f(x)", f"[bold green]{format_float(y_val)}[/]")

    if known_f_opt is not None:
        table.add_row("Known Optimum f(x*)", format_float(known_f_opt))
        assert err_to_opt is not None
        table.add_row("Error |f(x) - f(x*)|", f"[bold]{format_float(err_to_opt)}[/]")

    table.add_row("Latency", f"{dt_ms:.4f} ms")

    panel = Panel(
        table,
        title=f"[bold]pyMOFL Evaluation: [cyan]{display_name}[/bold]",
        subtitle=f"f(x) = {format_float(y_val)}",
        border_style="green",
    )
    console.print(panel)
