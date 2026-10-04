"""Optional CLI adapter for the shared observed-reference validator."""

import json
from pathlib import Path
from typing import Annotated

import typer

from pyMOFL.reference_validation import validate_reference_manifest


def validate(
    ctx: typer.Context,
    capture_root: Annotated[Path, typer.Option(help="Explicit local approved capture root.")],
    manifest: Annotated[Path, typer.Option(help="Producer-created reference manifest JSON.")],
    report: Annotated[Path, typer.Option(help="Fresh destination for the versioned JSON report.")],
) -> None:
    """Execute required local references and report actual scalar/batch outcomes."""
    result = validate_reference_manifest(manifest, capture_root=capture_root)
    try:
        with report.open("x") as destination:
            destination.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except OSError:
        typer.echo("Could not create the validation report; use a writable fresh path.", err=True)
        raise typer.Exit(code=1) from None
    options = ctx.obj or {}
    if options.get("json"):
        typer.echo(json.dumps(result, allow_nan=False))
    elif not options.get("quiet"):
        typer.echo(
            f"Required cases: {result['required_cases']}; "
            f"scalar: {result['scalar_executed']}; batch: {result['batch_executed']}; "
            f"statuses: {result['status_counts']}"
        )
    raise typer.Exit(code=0 if result["success"] else 1)
