"""
BoundModeEnum defines the context in which bounds are applied: initialization or operational.
"""
from enum import Enum, auto

class BoundModeEnum(Enum):
    """
    Enum for specifying the mode in which bounds are used.
    - INITIALIZATION: Used for random initialization of solutions.
    - OPERATIONAL: Used for enforcing domain limits during optimization.
    """
    INITIALIZATION = auto()
    OPERATIONAL = auto() 