"""Preserve real built-in registrations and the explicit extension paths."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pyMOFL
from pyMOFL.factories.function_factory import FunctionRegistry
from pyMOFL.functions import benchmark
from pyMOFL.functions.benchmark.rastrigin import RastriginFunction
from pyMOFL.functions.benchmark.sphere import SphereFunction
from pyMOFL.registry import _COMPONENTS, get, register, scan_package


def test_baseline_alias_class_identities_are_preserved() -> None:
    """Keep all captured aliases, including non-benchmark exported components."""
    capture = Path(__file__).parent / "fixtures/registry/baseline-aliases.json"
    expected = json.loads(capture.read_text())["aliases"]
    for alias, qualified_name in expected.items():
        component = get(alias)
        assert f"{component.__module__}.{component.__qualname__}" == qualified_name, alias


def test_all_exported_benchmark_classes_are_registered() -> None:
    registered = set(_COMPONENTS.values())
    for name in benchmark.__all__:
        component = getattr(benchmark, name)
        if inspect.isclass(component) and issubclass(component, pyMOFL.OptimizationFunction):
            assert component in registered, name


def test_explicit_scan_retains_aliases() -> None:
    before = _COMPONENTS.copy()
    scan_package("pyMOFL.functions.benchmark")
    assert before == _COMPONENTS
    assert get("sphere") is get("Sphere") is SphereFunction


def test_registration_after_loading_is_visible() -> None:
    """Update a real alias to another actual source class, then restore it."""
    pyMOFL.load("sphere", dimension=10)
    alias = "sphere"
    original = get(alias)
    try:
        register(alias)(RastriginFunction)
        assert type(pyMOFL.load(alias, dimension=10)) is RastriginFunction
        factory_registry = FunctionRegistry()
        factory_registry.register_base(alias, RastriginFunction)
        assert type(factory_registry.create_base_function(alias, dimension=10)) is RastriginFunction
    finally:
        register(alias)(original)


def test_core_import_does_not_import_optional_packages() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import pyMOFL, sys, json; print(json.dumps(sorted(sys.modules)))"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        cwd=Path(__file__).resolve().parents[1],
    )
    modules = json.loads(result.stdout)
    for prefix in ("pyMOFL.cli", "typer", "rich", "mkdocs"):
        assert not any(name == prefix or name.startswith(prefix + ".") for name in modules)
