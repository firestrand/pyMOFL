"""
Hybrid benchmark functions for optimization.

These functions combine multiple characteristics or variable types,
such as continuous and discrete variables.
"""

from .network import NetworkFunction
from .compression_spring import CompressionSpringFunction

__all__ = [
    "NetworkFunction",
    "CompressionSpringFunction"
] 