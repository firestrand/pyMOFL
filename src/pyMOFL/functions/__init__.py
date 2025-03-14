"""
Benchmark optimization functions for the pyMOFL library.

This package contains various benchmark functions commonly used in optimization research,
organized into categories such as unimodal, multimodal, hybrid, and CEC functions.
"""

# Import subpackages
from . import unimodal
from . import multimodal
from . import hybrid
from . import cec

# Import common functions for easier access
from .unimodal import *
from .multimodal import * 