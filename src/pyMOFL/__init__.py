"""
pyMOFL: Python Modular Optimization Function Library

A composable optimization function library for benchmarking optimization algorithms.

This library provides a collection of benchmark functions commonly used in optimization research,
along with tools for transforming and composing these functions to create complex benchmarks.
"""

__version__ = "0.4.0"

# Import the base class from the new location
from pyMOFL.registry import _discover_builtins
from pyMOFL.registry import scan_package as scan_package

# Import function categories (explicit re-exports for public API)
from . import compositions as compositions
from . import functions as functions
from . import utils as utils
from .core.function import OptimizationFunction as OptimizationFunction
from .evaluation import evaluate_chunks as evaluate_chunks
from .loader import BenchmarkSuite as BenchmarkSuite
from .loader import get_suite as get_suite
from .loader import load as load

_discover_builtins()

__all__ = [
    "BenchmarkSuite",
    "OptimizationFunction",
    "compositions",
    "evaluate_chunks",
    "functions",
    "get_suite",
    "load",
    "utils",
]
