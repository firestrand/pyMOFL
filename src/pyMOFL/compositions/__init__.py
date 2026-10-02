"""
Function compositions for the pyMOFL library.

Compositions allow multiple functions to be combined into a single optimization problem.
"""

from .decomposed_function import DecomposedFunction
from .hybrid_function import HybridFunction
from .min_composition import MinComposition
from .weighted_composition import WeightedComposition

__all__ = ["DecomposedFunction", "HybridFunction", "MinComposition", "WeightedComposition"]
