"""Catalog and discovery commands for pyMOFL functions and benchmark suites."""

from __future__ import annotations

import contextlib
import inspect
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from pyMOFL.core.function import OptimizationFunction
from pyMOFL.loader import _discover_suite_configs, _resolve_suite_path
from pyMOFL.registry import _COMPONENTS, scan_package
from pyMOFL.utils.suite_config import load_suite_config, supported_dimensions

console = Console()


def _get_classical_functions() -> list[dict[str, Any]]:
    """Gather all registered classical optimization function classes."""
    scan_package()
    items: list[dict[str, Any]] = []
    seen_classes: set[type] = set()

    for alias, cls in sorted(_COMPONENTS.items(), key=lambda t: str(t[0])):
        if isinstance(cls, type) and issubclass(cls, OptimizationFunction):
            if cls in seen_classes:
                continue
            seen_classes.add(cls)

            sig = inspect.signature(cls.__init__)
            dim_param = sig.parameters.get("dimension")
            default_dim = (
                dim_param.default
                if dim_param and dim_param.default != inspect.Parameter.empty
                else None
            )

            dim_str = f"Fixed (D={default_dim})" if default_dim else "Scalable (D ≥ 1)"
            doc = (cls.__doc__ or "").strip().split("\n")[0]

            items.append(
                {
                    "id": alias,
                    "name": cls.__name__.replace("Function", ""),
                    "suite": "classical",
                    "category": "Classical / Primitive",
                    "dimensions": dim_str,
                    "description": doc,
                }
            )
    return items


def list_functions(
    ctx: typer.Context,
    suite: Annotated[
        str | None,
        typer.Option(
            "--suite", "-s", help="Filter by suite ID (e.g. 'cec2017', 'bbob', 'classical')"
        ),
    ] = None,
    category: Annotated[
        str | None,
        typer.Option("--category", "-c", help="Filter by category (e.g. 'unimodal', 'multimodal')"),
    ] = None,
    search: Annotated[
        str | None,
        typer.Option(
            "--search", "-q", help="Filter by substring in function name, ID, or description"
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum functions to display in table"),
    ] = 50,
    all_: Annotated[
        bool,
        typer.Option("--all", "-a", help="Show all matching functions without truncation"),
    ] = False,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON array of functions"),
    ] = False,
) -> None:
    """List and search benchmark functions available in pyMOFL."""
    output_json = output_json or bool(getattr(ctx, "obj", {}).get("json", False))
    quiet = bool(getattr(ctx, "obj", {}).get("quiet", False))

    functions: list[dict[str, Any]] = []

    # 1. Classical only
    if suite and suite.strip().lower() == "classical":
        functions = _get_classical_functions()

    # 2. Specific suite
    elif suite:
        clean_suite = suite.strip().lower()
        suite_path = _resolve_suite_path(clean_suite)
        data = load_suite_config(suite_path)
        raw_funcs = data.get("functions", [])
        suite_id = str(data.get("suite_id", clean_suite))
        for item in raw_funcs:
            if isinstance(item, dict):
                dims = supported_dimensions(item)
                dim_str = ", ".join(str(d) for d in dims) if dims else "Scalable"
                functions.append(
                    {
                        "id": str(item.get("id", "")),
                        "name": str(item.get("name", item.get("id", ""))),
                        "suite": suite_id,
                        "category": str(item.get("category", "General")),
                        "dimensions": dim_str,
                        "description": str(item.get("description", "")),
                    }
                )

    # 3. All suites + classical
    else:
        # Load from all canonical suites
        suites = _discover_suite_configs()
        seen_paths = set()
        for s_key, s_path in sorted(suites.items()):
            if s_path in seen_paths:
                continue
            seen_paths.add(s_path)
            with contextlib.suppress(Exception):
                data = load_suite_config(s_path)
                s_id = str(data.get("suite_id", s_key))
                for item in data.get("functions", []):
                    if isinstance(item, dict):
                        dims = supported_dimensions(item)
                        dim_str = ", ".join(str(d) for d in dims) if dims else "Scalable"
                        functions.append(
                            {
                                "id": str(item.get("id", "")),
                                "name": str(item.get("name", item.get("id", ""))),
                                "suite": s_id,
                                "category": str(item.get("category", "General")),
                                "dimensions": dim_str,
                                "description": str(item.get("description", "")),
                            }
                        )
        # Add classical functions
        functions.extend(_get_classical_functions())

    # Apply filters
    filtered: list[dict[str, Any]] = []
    for f in functions:
        if category and category.lower() not in f["category"].lower():
            continue
        if search:
            q = search.lower()
            text_corpus = f"{f['id']} {f['name']} {f['description']} {f['suite']}".lower()
            if q not in text_corpus:
                continue
        filtered.append(f)

    if output_json:
        console.print_json(data=filtered)
        return

    if not filtered:
        if not quiet:
            console.print("[yellow]No functions matched the requested filters.[/yellow]")
        return

    display_items = filtered if all_ else filtered[:limit]

    table = Table(title=f"pyMOFL Functions ({len(filtered):,} available)")
    table.add_column("Function ID", style="bold cyan")
    table.add_column("Name")
    table.add_column("Suite", style="green")
    table.add_column("Category")
    table.add_column("Dimensions", style="yellow")

    for f in display_items:
        table.add_row(
            f["id"],
            f["name"],
            f["suite"],
            f["category"],
            f["dimensions"],
        )

    console.print(table)

    if len(filtered) > len(display_items):
        rem = len(filtered) - len(display_items)
        console.print(
            f"[dim]... and {rem:,} more functions. Use [bold]--all[/bold] to display all or filter with [bold]--search[/bold] / [bold]--suite[/bold].[/dim]"
        )


def list_suites(
    ctx: typer.Context,
    search: Annotated[
        str | None,
        typer.Option("--search", "-q", help="Filter by substring in suite name or description"),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON array of suites"),
    ] = False,
) -> None:
    """List all benchmark suites supported by pyMOFL."""
    output_json = output_json or bool(getattr(ctx, "obj", {}).get("json", False))

    suites_dict = _discover_suite_configs()
    seen_paths = set()
    suite_records: list[dict[str, Any]] = []

    for s_key, s_path in sorted(suites_dict.items()):
        if s_path in seen_paths:
            continue
        seen_paths.add(s_path)
        try:
            data = load_suite_config(s_path)
            s_funcs = data.get("functions", [])
            # Collect common dimensions
            all_dims: set[int] = set()
            for item in s_funcs:
                if isinstance(item, dict):
                    all_dims.update(supported_dimensions(item))
            dim_str = ", ".join(str(d) for d in sorted(all_dims)) if all_dims else "Arbitrary"

            record = {
                "suite_id": str(data.get("suite_id", s_key)),
                "name": str(data.get("name", s_key)),
                "functions_count": len(s_funcs),
                "dimensions": dim_str,
                "description": str(data.get("description", "")),
            }
            suite_records.append(record)
        except Exception:
            continue

    # Add classical registry entry
    classical_funcs = _get_classical_functions()
    suite_records.append(
        {
            "suite_id": "classical",
            "name": "Classical & Primitive Functions",
            "functions_count": len(classical_funcs),
            "dimensions": "Arbitrary (D ≥ 1)",
            "description": "Classical test problems (Sphere, Rosenbrock, Rastrigin, Ackley, Schwefel, etc.)",
        }
    )

    # Filter
    filtered: list[dict[str, Any]] = []
    for s in suite_records:
        if search:
            q = search.lower()
            if q not in f"{s['suite_id']} {s['name']} {s['description']}".lower():
                continue
        filtered.append(s)

    if output_json:
        console.print_json(data=filtered)
        return

    table = Table(title="pyMOFL Benchmark Suites")
    table.add_column("Suite ID", style="bold cyan")
    table.add_column("Suite Name", style="bold")
    table.add_column("Functions", justify="right", style="yellow")
    table.add_column("Supported Dims")
    table.add_column("Description")

    for s in filtered:
        desc = s["description"]
        if len(desc) > 65:
            desc = desc[:62] + "..."
        table.add_row(
            s["suite_id"],
            s["name"],
            str(s["functions_count"]),
            s["dimensions"],
            desc,
        )

    console.print(table)
