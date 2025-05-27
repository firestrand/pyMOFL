"""
QuantizationTypeEnum defines the type of quantization applied to variables: continuous, integer, or stepwise.
"""
from enum import Enum, auto

class QuantizationTypeEnum(Enum):
    """
    Enum for specifying the quantization type of variables.
    - CONTINUOUS: Variables can take any real value within bounds.
    - INTEGER: Variables are restricted to integer values within bounds.
    - STEP: Variables are quantized to a fixed step size within bounds.
    """
    CONTINUOUS = auto()
    INTEGER = auto()
    STEP = auto() 