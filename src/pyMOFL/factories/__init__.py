"""
Factory modules for creating benchmark functions from configuration.

Exports the unified BenchmarkFactory and the decomposed factory components:
- DataLoader: loads vectors/matrices from disk
- ConfigParser: parses nested JSON configs
- TransformBuilder: constructs transform objects
- CompositionBuilder: builds weighted compositions
- FunctionFactory: orchestrates config → composed function
- FunctionRegistry: maps type strings → benchmark classes
"""

from .benchmark_factory import BenchmarkFactory
from .composition_builder import CompositionBuilder
from .config_parser import ConfigParser
from .data_loader import DataLoader
from .function_factory import FunctionFactory, FunctionRegistry
from .transform_builder import TransformBuilder

__all__ = [
    "BenchmarkFactory",
    "CompositionBuilder",
    "ConfigParser",
    "DataLoader",
    "FunctionFactory",
    "FunctionRegistry",
    "TransformBuilder",
]
