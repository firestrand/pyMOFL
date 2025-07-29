"""
Expanded functions module.

This module contains implementations of expanded optimization functions,
which are created by composing two or more base functions.
"""

from .griewank_of_rosenbrock import GriewankOfRosenbrock
from .schaffer_f6_expanded import SchafferF6Expanded

__all__ = [
    "GriewankOfRosenbrock",
    "SchafferF6Expanded",
]