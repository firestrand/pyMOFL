"""Root typer application for the pyMOFL CLI."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Annotated

import typer

from . import suite

app = typer.Typer(
    name="pymofl",
    help="pyMOFL — Python Modular Optimization Function Library CLI.",
    no_args_is_help=True,
)


@app.callback()
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


app.add_typer(suite.app, name="suite", help="Suite utilities for benchmark configurations.")
