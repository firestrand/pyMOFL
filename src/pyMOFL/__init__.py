"""
pyMOFL: Python Modular Optimization Function Library

A composable optimization function library for benchmarking optimization algorithms.

This library provides a collection of benchmark functions commonly used in optimization research,
along with tools for transforming and composing these functions to create complex benchmarks.
"""

__version__ = "0.1.0"

# Import the base class
from .base import OptimizationFunction

# Import function categories
from . import functions
from . import decorators
from . import composites
from . import utils 