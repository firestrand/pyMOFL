"""Example demonstrating config-driven function construction.

This example shows how CEC-style function definitions in suite JSON are turned
into explicit compositions via the unified `FunctionFactory`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyMOFL.factories import DataLoader, FunctionFactory, FunctionRegistry
from pyMOFL.utils import load_suite_function_config
from pyMOFL.functions.transformations import ComposedFunction


def _factory() -> FunctionFactory:
    project_root = Path(__file__).resolve().parent.parent
    constants_dir = project_root / "src" / "pyMOFL" / "constants" / "cec" / "2005"
    return FunctionFactory(data_loader=DataLoader(base_path=constants_dir), registry=FunctionRegistry())


def show_structure() -> None:
    print("=" * 80)
    print("CEC2005 Config-Driven Factory Example")
    print("=" * 80)
    print("The config for each function is a nested transform graph:")
    print("  bias -> base/inner transform -> shift -> rotate -> ...")
    print("The parser resolves it into explicit input/output transform chains.")


def build_function(function_id: str, dimension: int = 10) -> ComposedFunction:
    """Build one suite function by id from shipped CEC2005 config."""

    project_root = Path(__file__).resolve().parent.parent
    suite_path = project_root / "src" / "pyMOFL" / "constants" / "cec" / "2005" / "cec2005_suite.json"

    function_cfg = load_suite_function_config(suite_path, function_id, dimension=dimension)

    return _factory().create_function(function_cfg)


def describe_chain(fn: ComposedFunction) -> str:
    """Return a compact string summary of base + transform order."""

    chain = [type(fn.base_function).__name__]
    for t in fn.input_transforms:
        chain.append(type(t).__name__)
    for t in fn.output_transforms:
        chain.append(type(t).__name__)
    return " -> ".join(chain)


def demonstrate_factory_building() -> None:
    print("\n" + "=" * 80)
    print("Factory Building")
    print("=" * 80)

    func = build_function("cec05_f01_shifted_sphere", dimension=10)
    print(f"Built function type: {type(func).__name__}")
    print(f"Composition chain: {describe_chain(func)}")

    points = [
        np.zeros(10),
        np.ones(10),
        np.random.default_rng(0).standard_normal(10),
    ]
    print("\nEvaluation sample:")
    for i, point in enumerate(points):
        print(f"  Point {i}: {func.evaluate(point):.12f}")


def demonstrate_transformation_order() -> None:
    print("\n" + "=" * 80)
    print("Transformation Order")
    print("=" * 80)
    print("ComposedFunction applies input transforms, then base function, then output transforms.")

    func = build_function("cec05_f03_shifted_rotated_high_conditioned_elliptic", dimension=10)
    print(f"F03 chain: {describe_chain(func)}")


def demonstrate_composition_building() -> None:
    print("\n" + "=" * 80)
    print("Composition Function Note")
    print("=" * 80)
    print("F15-F25 remain composition nodes in suite config and are assembled through")
    print("the same FunctionFactory path (nested configs + composition builder).")


if __name__ == "__main__":
    show_structure()
    demonstrate_factory_building()
    demonstrate_transformation_order()
    demonstrate_composition_building()

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("- suite JSON defines pipeline structure")
    print("- FunctionFactory resolves into explicit ComposedFunction")
    print("- transforms are explicit and order-specific")
