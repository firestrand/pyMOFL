"""
pyMOFL: Python Modular Optimization Function Library

A composable optimization function library for benchmarking optimization algorithms.

This library provides a collection of benchmark functions commonly used in optimization research,
along with tools for transforming and composing these functions to create complex benchmarks.
"""

__version__ = "0.1.0"

# Import the base class from the new location
from pyMOFL.registry import scan_package

# Import function categories (explicit re-exports for public API)
from . import compositions as compositions
from . import functions as functions
from . import utils as utils
from .core.function import OptimizationFunction as OptimizationFunction

scan_package()

# Expose commonly used test utilities in builtins for legacy tests
try:
    import builtins

    from pyMOFL.functions.benchmark.max_absolute import MaxAbsolute
    from pyMOFL.functions.transformations.linear import linear_transform_optimized
    from pyMOFL.functions.transformations.quantized import Quantized

    builtins.MaxAbsolute = MaxAbsolute  # type: ignore[attr-defined]
    builtins.Quantized = Quantized  # type: ignore[attr-defined]
    builtins.linear_transform_optimized = linear_transform_optimized  # type: ignore[attr-defined]
except Exception:
    pass
