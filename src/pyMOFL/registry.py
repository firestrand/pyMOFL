"""
pyMOFL Component Registry

A unified global registry for all optimization functions, decorators, and composites.
Supports self-registration via @register and lookup via get().

Usage:
    from pyMOFL.registry import register, get

    @register()                # uses class.__name__
    class SphereFunction(...):
        ...
    @register("Shift")
    class ShiftedFunction(...):
        ...

    f_cls = get("SphereFunction")
    f = f_cls(...)

"""

import pkgutil
from collections.abc import Callable
from functools import cache
from importlib import import_module

_COMPONENTS: dict[str, type] = {}


def register[T](name: str | None = None) -> Callable[[type[T]], type[T]]:
    """
    Class decorator – registers the class with a global registry.
    Usage:
        @register()                    # uses class.__name__
        class SphereFunction(...):
            ...
        @register("Shift")
        class ShiftedFunction(...):
            ...
    """

    def _inner(cls: type[T]) -> type[T]:
        _COMPONENTS[name or cls.__name__] = cls
        return cls

    return _inner


def get(name: str) -> type:
    try:
        return _COMPONENTS[name]
    except KeyError as e:
        raise ValueError(f"Component '{name}' is not registered") from e


def scan_package(pkg_name: str = "pyMOFL") -> None:
    """
    Import every sub-module once so that their @register decorators fire.
    Explicit scans remain available for extension packages. Automatic built-in
    discovery uses a narrower package and does not import optional integrations.
    """
    pkg = import_module(pkg_name)
    for mod in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg_name}."):
        import_module(mod.name)


@cache
def _discover_builtins() -> None:
    """Discover built-in benchmarks once without scanning optional packages.

    Only automatic discovery is cached. Explicit scans and subsequent register()
    calls still update the live registry used by load() and FunctionRegistry.
    """
    scan_package("pyMOFL.functions.benchmark")
