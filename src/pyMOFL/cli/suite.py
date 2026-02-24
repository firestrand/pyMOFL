"""Generic suite utility commands for all benchmark families."""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.table import Table

from pyMOFL.utils import (
    FILE_REFERENCE_EXTENSIONS,
    iter_file_references,
    load_suite_config,
    supported_dimensions,
)

app = typer.Typer(no_args_is_help=True)
console = Console()


def _status(message: str, quiet: bool, style: str | None = None) -> None:
    if quiet:
        return
    if style is None:
        console.print(message)
    else:
        console.print(f"[{style}]{message}[/{style}]")


def _constants_root() -> Path:
    """Return the local constants directory for shipped suite descriptors."""

    return Path(__file__).resolve().parent.parent / "constants"


def _suite_config_candidates() -> builtins.list[Path]:
    return sorted(path for path in _constants_root().rglob("*_suite.json") if path.is_file())


def _load_suite(path: Path) -> dict[str, object]:
    try:
        payload = load_suite_config(path)
    except (TypeError, OSError, ValueError) as exc:
        raise typer.BadParameter(f"Suite config '{path}' is invalid: {exc}") from exc
    return payload


def _find_suite_config(suite: Path | str | None, suite_id: str | None) -> Path:
    if suite is not None:
        suite_path = Path(suite)
        if not suite_path.exists():
            raise typer.BadParameter(f"Suite file does not exist: {suite_path}")
        return suite_path

    candidates = _suite_config_candidates()
    if suite_id is not None:
        matches: list[Path] = []
        for candidate in candidates:
            payload = _load_suite(candidate)
            if str(payload.get("suite_id", "")) == suite_id:
                matches.append(candidate)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            message = ", ".join(str(path) for path in matches)
            raise typer.BadParameter(f"Suite id '{suite_id}' is ambiguous: {message}")
        raise typer.BadParameter(f"No suite found with id '{suite_id}'.")

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise typer.BadParameter("No suite configuration files found under package constants.")

    available = ", ".join(path.stem.replace("_suite", "") for path in candidates)
    raise typer.BadParameter(
        f"Multiple suite configs found. Specify --suite or --suite-id. Available: {available}"
    )


def _validation_failures(
    functions: builtins.list[dict[str, object]],
    suite_dir: Path,
) -> builtins.list[dict[str, object]]:
    missing: list[dict[str, object]] = []
    for function_entry in functions:
        if not isinstance(function_entry, dict):
            continue
        function_id = str(function_entry.get("id", "<unknown>"))
        references = sorted(set(iter_file_references(function_entry, FILE_REFERENCE_EXTENSIONS)))
        dimensions = supported_dimensions(function_entry)
        dimensions_to_check = dimensions or [None]

        for reference in references:
            if "{dim}" in reference:
                for dim in dimensions_to_check:
                    if dim is None:
                        continue
                    candidate = reference.format(dim=dim)
                    if not (suite_dir / candidate).exists():
                        missing.append(
                            {
                                "function_id": function_id,
                                "reference": reference,
                                "dimension": dim,
                                "resolved": candidate,
                            }
                        )
            else:
                if not (suite_dir / reference).exists():
                    missing.append(
                        {
                            "function_id": function_entry.get("id", "<unknown>"),
                            "reference": reference,
                            "dimension": None,
                            "resolved": reference,
                        }
                    )

    return missing


@app.command("list")
def list_functions(
    ctx: typer.Context,
    suite: Annotated[
        Path | None, typer.Option("--suite", help="Path to a suite JSON configuration")
    ] = None,
    suite_id: Annotated[
        str | None, typer.Option("--suite-id", help="Suite identifier (suite_id in config)")
    ] = None,
    category: Annotated[str | None, typer.Option(help="Filter by function category")] = None,
    search: Annotated[str | None, typer.Option(help="Filter by substring in function id")] = None,
) -> None:
    """List benchmark functions defined in a suite configuration."""

    suite_path = _find_suite_config(suite, suite_id)
    suite_data = _load_suite(suite_path)
    functions = suite_data.get("functions", [])
    if not isinstance(functions, builtins.list):
        raise typer.BadParameter("Suite config is missing a valid 'functions' list")

    normalized_functions: builtins.list[dict[str, object]] = []
    for item in functions:
        if isinstance(item, dict):
            normalized_functions.append(cast(dict[str, object], item))

    selected = []
    for function_entry in normalized_functions:
        if not isinstance(function_entry, dict):
            continue
        function_id = str(function_entry.get("id", ""))
        function_category = function_entry.get("category")
        if category is not None and str(function_category) != category:
            continue
        if search is not None and search.lower() not in function_id.lower():
            continue
        dimensions = supported_dimensions(function_entry)
        selected.append(
            {
                "id": function_id,
                "category": str(function_category or ""),
                "dimensions": dimensions,
            }
        )

    output_json = bool(getattr(ctx, "obj", {}).get("json", False))
    if output_json:
        console.print_json(
            data={
                "command": "suite.list",
                "suite": str(suite_path),
                "count": len(selected),
                "functions": selected,
            }
        )
        return

    quiet = bool(getattr(ctx, "obj", {}).get("quiet", False))
    if not selected:
        _status("No functions matched the requested filters.", quiet)
        return

    table = Table(title="Suite Functions")
    table.add_column("Function")
    table.add_column("Category")
    table.add_column("Supported Dims")
    for function_data in selected:
        table.add_row(
            function_data["id"],
            function_data["category"],
            ", ".join(map(str, function_data["dimensions"])) if function_data["dimensions"] else "",
        )
    console.print(table)


@app.command()
def validate(
    ctx: typer.Context,
    suite: Annotated[
        Path | None, typer.Option("--suite", help="Path to a suite JSON configuration")
    ] = None,
    suite_id: Annotated[
        str | None, typer.Option("--suite-id", help="Suite identifier (suite_id in config)")
    ] = None,
    suite_dir: Annotated[
        Path | None,
        typer.Option(help="Directory that contains suite resource files"),
    ] = None,
    strict: Annotated[
        bool, typer.Option("--strict", help="Return non-zero exit code on any missing file")
    ] = False,
    output_json: Annotated[bool, typer.Option("--json", help="Emit compact JSON output")] = False,
) -> None:
    """Validate suite file references against local resource files."""

    suite_path = _find_suite_config(suite, suite_id)
    suite_data = _load_suite(suite_path)
    suite_data_dir = suite_dir or suite_path.parent
    functions = suite_data.get("functions", [])
    if not isinstance(functions, builtins.list):
        raise typer.BadParameter("Suite config is missing a valid 'functions' list")

    normalized_functions: builtins.list[dict[str, object]] = []
    for item in functions:
        if isinstance(item, dict):
            normalized_functions.append(cast(dict[str, object], item))

    missing = _validation_failures(normalized_functions, suite_data_dir)

    output_json = output_json or bool(getattr(ctx, "obj", {}).get("json", False))
    quiet = bool(getattr(ctx, "obj", {}).get("quiet", False))
    if output_json:
        payload = {
            "command": "suite.validate",
            "suite": str(suite_path),
            "suite_dir": str(suite_data_dir),
            "status": "pass" if not missing else "fail",
            "missing_count": len(missing),
            "missing": missing,
        }
        console.print_json(data=payload)
        if strict and missing:
            raise typer.Exit(code=1)
        return

    if not missing:
        _status("All referenced files exist.", quiet, style="green")
        return

    table = Table(title="Missing file references")
    table.add_column("Function")
    table.add_column("Reference")
    table.add_column("Dimension")
    table.add_column("Resolved")
    for item in missing:
        table.add_row(
            str(item["function_id"]),
            str(item["reference"]),
            str(item["dimension"]),
            str(item["resolved"]),
        )
    _status("Missing suite references:", quiet, style="red")
    console.print(table)

    if strict:
        raise typer.Exit(code=1)
