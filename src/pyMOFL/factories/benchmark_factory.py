"""
BenchmarkFactory (deprecated) — thin adapter to the unified FunctionFactory.

This keeps the legacy import path stable while routing all construction
through the config-driven, purely functional FunctionFactory. It also
supports the historical `dimension` argument by injecting it into the
provided config when missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pyMOFL.core.function import OptimizationFunction
from pyMOFL.utils import inject_dimension

from .function_factory import DataLoader, FunctionFactory, FunctionRegistry


class BenchmarkFactory:
    """Deprecated wrapper that delegates to FunctionFactory.

    Args:
        data_path: Base path for loading vectors/matrices (passed to DataLoader).
    """

    def __init__(self, data_path: str | Path | None = None) -> None:
        base_path = Path(data_path) if data_path is not None else Path.cwd()
        self._loader = DataLoader(base_path=base_path)
        self._registry = FunctionRegistry()
        self._factory = FunctionFactory(data_loader=self._loader, registry=self._registry)

    def create_function(
        self, config: dict[str, Any], dimension: int | None = None
    ) -> OptimizationFunction:
        """Create a function via the unified factory.

        If `dimension` is provided, inject it recursively so nested parsing can pick
        it up from any nested function node.
        """
        if dimension is None:
            return self._factory.create_function(config)
        normalized_config = inject_dimension(config, dimension)
        return self._factory.create_function(normalized_config)
