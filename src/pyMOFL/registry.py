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

    # For migration
    register_function = register
"""

from importlib import import_module
import pkgutil

_COMPONENTS = {}

def register(name=None):
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
    def _inner(cls):
        _COMPONENTS[name or cls.__name__] = cls
        return cls
    return _inner

def get(name):
    try:
        return _COMPONENTS[name]
    except KeyError as e:
        raise ValueError(f"Component '{name}' is not registered") from e

def scan_package(pkg_name="pyMOFL"):
    """
    Import every sub-module once so that their @register decorators fire.
    Call this once in library __init__.py.
    """
    pkg = import_module(pkg_name)
    for mod in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg_name}."):
        import_module(mod.name)

# Alias for migration
register_function = register 