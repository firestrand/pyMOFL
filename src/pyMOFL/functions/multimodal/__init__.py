"""
Multimodal benchmark functions for optimization.

These functions have multiple local optima and are typically used to test
the ability of optimization algorithms to escape local optima.
"""

from .rastrigin import RastriginFunction
from .tripod import TripodFunction
from .step import StepFunction
from .lennard_jones import LennardJonesFunction

__all__ = [
    "RastriginFunction",
    "TripodFunction",
    "StepFunction",
    "LennardJonesFunction"
]



 