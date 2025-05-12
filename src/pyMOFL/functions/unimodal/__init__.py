"""
Unimodal benchmark functions for optimization.

These functions have a single global optimum and are typically used to test
the convergence speed of optimization algorithms.
"""

from .sphere import SphereFunction
from .rosenbrock import RosenbrockFunction
from .schwefel import (
    SchwefelFunction12, 
    SchwefelFunction26,
    SchwefelProblem12Function,  # For backward compatibility
    SchwefelProblem26Function   # For backward compatibility
)
from .elliptic import EllipticFunction 