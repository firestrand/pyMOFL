"""Inspection command for pyMOFL benchmark functions."""

from __future__ import annotations

from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import pyMOFL
from pyMOFL.cli.common import (
    clean_function_name,
    find_suite_metadata,
    format_bounds,
    format_float,
    format_vector,
    resolve_optimum,
)
from pyMOFL.core.function import OptimizationFunction

console = Console()


def get_function_info_dict(
    name_or_id: str,
    dimension: int | None = None,
    suite: str | None = None,
    instance: int | None = None,
) -> dict[str, Any]:
    """Inspect and return full metadata dictionary for an optimization function."""
    func: OptimizationFunction
    try:
        func = pyMOFL.load(name_or_id, dimension=dimension, suite=suite, instance=instance)
    except ValueError as exc:
        if dimension is None and "requires a 'dimension' parameter" in str(exc):
            func = pyMOFL.load(name_or_id, dimension=10, suite=suite, instance=instance)
        else:
            raise exc

    suite_name, suite_entry = find_suite_metadata(name_or_id, explicit_suite=suite)
    cls = func.__class__
    cls_name = cls.__name__
    module_name = cls.__module__

    func_id = getattr(func, "function_id", None) or name_or_id
    raw_name = getattr(func, "name", None) or (suite_entry.get("name") if suite_entry else None)
    human_name = clean_function_name(raw_name, str(func_id), cls_name)

    # Category
    category = "Classical / Primitive"
    if suite_entry and suite_entry.get("category"):
        category = str(suite_entry["category"])
    elif suite_name:
        category = suite_name.upper()
    elif "benchmark" in module_name:
        category = "Benchmark"

    # Supported dimensions
    supported_dims = "Scalable (D ≥ 1)"
    if suite_entry and "dimensions" in suite_entry:
        dim_info = suite_entry["dimensions"]
        if isinstance(dim_info, dict) and "supported" in dim_info:
            supported_dims = ", ".join(str(d) for d in dim_info["supported"])
    elif cls_name in {"GearTrainFunction", "TripodFunction", "CompressionSpringFunction"}:
        supported_dims = f"Fixed (D={func.dimension})"

    # Global minimum & optimum
    x_opt, f_opt = resolve_optimum(func, suite_entry=suite_entry)

    # Description
    docstring = (cls.__doc__ or "").strip()
    doc_lines = [line.strip() for line in docstring.split("\n") if line.strip()]
    description = (suite_entry.get("description") if suite_entry else None) or (
        doc_lines[0] if doc_lines else human_name
    )

    # Check for known formula or references
    formula = None
    for line in doc_lines:
        if "f(x)" in line or "=" in line:
            formula = line
            break

    oper_bounds = func.operational_bounds
    init_bounds = func.initialization_bounds

    return {
        "id": func_id,
        "name": human_name,
        "class": cls_name,
        "module": module_name,
        "suite": suite_name or "classical",
        "category": category,
        "dimension": func.dimension,
        "supported_dimensions": supported_dims,
        "operational_bounds": {
            "low": oper_bounds.low.tolist() if hasattr(oper_bounds.low, "tolist") else [],
            "high": oper_bounds.high.tolist() if hasattr(oper_bounds.high, "tolist") else [],
            "formatted": format_bounds(oper_bounds.low, oper_bounds.high, func.dimension),
            "quantization": str(oper_bounds.qtype),
        },
        "initialization_bounds": {
            "low": init_bounds.low.tolist() if hasattr(init_bounds.low, "tolist") else [],
            "high": init_bounds.high.tolist() if hasattr(init_bounds.high, "tolist") else [],
            "formatted": format_bounds(init_bounds.low, init_bounds.high, func.dimension),
        },
        "global_minimum": {
            "value": f_opt,
            "point": x_opt.tolist() if x_opt is not None else None,
            "point_formatted": format_vector(x_opt) if x_opt is not None else "Not specified",
        },
        "description": description,
        "formula": formula,
    }


def info(
    ctx: typer.Context,
    function: Annotated[
        str,
        typer.Argument(
            help="Function identifier, alias, or suite code (e.g. 'sphere', 'cec17_f01', 'bbob_f01')"
        ),
    ],
    dimension: Annotated[
        int | None,
        typer.Option(
            "--dimension", "-d", help="Instantiated dimension (defaults to 10 or default)"
        ),
    ] = None,
    suite: Annotated[
        str | None,
        typer.Option("--suite", "-s", help="Suite identifier (e.g. 'cec2017', 'bbob')"),
    ] = None,
    instance: Annotated[
        int | None,
        typer.Option(
            "--instance", "-i", help="Instance identifier for randomized suites (e.g. BBOB)"
        ),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit detailed JSON representation"),
    ] = False,
) -> None:
    """Inspect metadata, formulation, bounds, and optima for a benchmark function."""
    output_json = output_json or bool(getattr(ctx, "obj", {}).get("json", False))

    try:
        data = get_function_info_dict(
            name_or_id=function,
            dimension=dimension,
            suite=suite,
            instance=instance,
        )
    except Exception as exc:
        if output_json:
            console.print_json(data={"error": str(exc), "function": function})
            raise typer.Exit(code=1) from exc
        console.print(f"[bold red]Error:[/] Could not inspect function '{function}': {exc}")
        raise typer.Exit(code=1) from exc

    if output_json:
        console.print_json(data=data)
        return

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Property", style="bold cyan", width=22)
    table.add_column("Value")

    table.add_row("Function ID", f"[bold]{data['id'] or function}[/]")
    table.add_row("Name", str(data["name"]))
    table.add_row("Suite", f"[green]{data['suite']}[/]")
    table.add_row("Category", str(data["category"]))
    table.add_row("Implementation", f"{data['class']} ({data['module']})")
    table.add_row("Dimension", f"[bold yellow]{data['dimension']}[/]")
    table.add_row("Supported Dims", str(data["supported_dimensions"]))
    table.add_row("Operational Bounds", str(data["operational_bounds"]["formatted"]))
    table.add_row("Initialization Bounds", str(data["initialization_bounds"]["formatted"]))

    opt_val = data["global_minimum"]["value"]
    opt_val_str = (
        f"[bold green]{format_float(opt_val)}[/]"
        if opt_val is not None
        else "[dim]Not known / Shifted[/]"
    )
    table.add_row("Global Minimum f(x*)", opt_val_str)
    table.add_row("Optimum Location x*", str(data["global_minimum"]["point_formatted"]))

    if data.get("formula"):
        table.add_row("Formula / Excerpt", f"[italic]{data['formula']}[/]")
    table.add_row("Description", str(data["description"]))

    panel = Panel(
        table,
        title=f"[bold]pyMOFL Function Info: [cyan]{data['name']}[/bold]",
        subtitle=f"Suite: {data['suite']} | D={data['dimension']}",
        border_style="cyan",
    )
    console.print(panel)
