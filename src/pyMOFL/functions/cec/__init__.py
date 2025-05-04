"""
CEC benchmark functions package.

This package provides implementations of benchmark functions from various
IEEE Congress on Evolutionary Computation (CEC) competitions.
"""

from .cec2005 import (
    CEC2005Function,
    F01,
    F02,
    create_cec2005_function,
)

__all__ = [
    'CEC2005Function',
    'F01',
    'F02',
    'create_cec2005_function',
] 