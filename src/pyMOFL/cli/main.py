"""Root typer application for the pyMOFL CLI."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Annotated

import typer

from . import catalog, suite
from . import eval as cli_eval
from . import info as cli_info
from . import reference as cli_reference

app = typer.Typer(
    name="pymofl",
    help="pyMOFL — Python Modular Optimization Function Library CLI.",
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit.",
            is_eager=True,
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress informational tables and progress messages.",
        ),
    ] = False,
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit compact JSON output for commands that support it.",
        ),
    ] = False,
) -> None:
    """Shared CLI behavior and flags."""
    if version:
        try:
            typer.echo(_pkg_version("pyMOFL"))
        except PackageNotFoundError:
            typer.echo("0.0.0")
        raise typer.Exit(code=0)

    ctx.ensure_object(dict)
    ctx.obj["quiet"] = quiet
    ctx.obj["json"] = output_json


# Top-level ergonomics commands
app.command("info", help="Inspect metadata, formulation, bounds, and optima for a function.")(
    cli_info.info
)
app.command(
    "eval", help="Evaluate a benchmark function at specified points or randomized batches."
)(cli_eval.eval_fn)
app.command(
    "list", help="List and search benchmark functions available across suites and registry."
)(catalog.list_functions)
app.command("suites", help="List all benchmark suites supported by pyMOFL.")(catalog.list_suites)
app.command("validate", help="Execute required local reference cases and write a JSON report.")(
    cli_reference.validate
)

# Legacy suite utilities
app.add_typer(suite.app, name="suite", help="Suite utilities for benchmark configurations.")
