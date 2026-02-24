"""
Benchmark optimization functions for the pyMOFL library.

This package contains various benchmark functions commonly used in optimization research,
organized into submodules:

- benchmark: Core mathematical benchmark functions organized by families
- cec: Competition-specific benchmark suites (CEC 2005, etc.)

All benchmark functions use consistent naming conventions:
- Schwefel family: Schwefel_1_2, Schwefel_2_6, Schwefel_2_13
- Schaffer family: Schaffer_F6, Schaffer_F6_Expanded
- Individual functions: SphereFunction, RosenbrockFunction, etc.

Note: Function transformations (shift, rotate, bias, etc.) are now handled
directly by the ComposedFunction class without requiring wrapper objects.
"""

# Import everything from benchmark subfolder for easy access
# Import subpackages
from . import benchmark, cec
from .benchmark import *  # noqa: F403

# Export submodules and all benchmark functions
__all__ = [*benchmark.__all__, "benchmark", "cec"]
