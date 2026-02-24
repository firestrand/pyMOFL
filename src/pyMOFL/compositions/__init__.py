"""
Composition functions for combining multiple optimization functions.
"""

from .hybrid_function import HybridFunction
from .weighted_composition import WeightedComposition

__all__ = [
    "HybridFunction",
    "WeightedComposition",
]
