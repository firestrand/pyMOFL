"""
Multimodal functions module.

This module contains implementations of multimodal optimization functions,
i.e., functions with multiple local minima.
"""

from .rastrigin import RastriginFunction
from .step import StepFunction
from .tripod import TripodFunction
from .lennard_jones import LennardJonesFunction
from .gear_train import GearTrainFunction
from .perm import PermFunction
from .griewank import GriewankFunction
from .ackley import AckleyFunction
from .weierstrass import WeierstrassFunction
from .schwefel_2_13 import SchwefelFunction213
from .scaffer_f6 import SchafferF6Function, ScafferF6Function

__all__ = [
    "RastriginFunction",
    "StepFunction",
    "TripodFunction",
    "LennardJonesFunction",
    "GearTrainFunction",
    "PermFunction",
    "GriewankFunction",
    "AckleyFunction",
    "WeierstrassFunction",
    "SchwefelFunction213",
    "SchafferF6Function",
    "ScafferF6Function"
]



 