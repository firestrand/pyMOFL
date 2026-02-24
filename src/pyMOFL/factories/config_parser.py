"""Config parser for nested function configuration trees.

Parses nested JSON-style configs into a structured ParsedConfig,
collecting transforms in application order (innermost first).

Convention: nesting = composition order. Innermost node is applied
first to x. Read inside-out for evaluation order.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedConfig:
    """Result of parsing a nested function configuration.

    Attributes:
        base_type: The base function type string, or None if not found.
        base_params: Parameters dict for the base function.
        transforms: List of (type, params) tuples in application order
            (innermost first — shift before rotate before bias).
        is_composition: True if the base is a composition node.
        raw_composition_config: The full composition node dict if is_composition.
    """

    base_type: str | None
    base_params: dict[str, Any]
    transforms: list[tuple[str, dict[str, Any]]]
    is_composition: bool
    raw_composition_config: dict[str, Any] | None


class ConfigParser:
    """Parses nested function configs into structured ParsedConfig.

    The parser walks the config tree and returns transforms in application
    order (innermost first). This means shift transforms come before
    rotate transforms, which come before bias transforms.
    """

    def __init__(self, known_base_types: frozenset[str]) -> None:
        self._known_base_types = known_base_types

    def set_known_base_types(self, known_base_types: Iterable[str]) -> None:
        """Update known base types after dynamic registry changes."""
        self._known_base_types = frozenset(known_base_types)

    def parse(self, config: dict[str, Any]) -> ParsedConfig:
        """Parse a nested config into a structured ParsedConfig.

        Walks the config tree depth-first, collecting transforms in
        application order (innermost first).
        """
        found_dimension: int | None = None

        def walk(node: dict[str, Any]) -> tuple[str | None, dict[str, Any], list[tuple[str, Any]]]:
            nonlocal found_dimension

            if not node:
                return None, {}, []

            t = node.get("type")
            params = dict(node.get("parameters", {}))

            # Track dimension from any level
            if found_dimension is None and isinstance(params.get("dimension"), int):
                found_dimension = params["dimension"]
            if found_dimension is None and isinstance(params.get("dim"), int):
                found_dimension = params["dim"]

            # Skip transparent wrapper nodes
            if t in (None, "weight"):
                inner = node.get("function")
                if isinstance(inner, dict):
                    return walk(inner)
                return None, {}, []

            # Composition nodes are returned as-is
            if t == "composition":
                return "composition", node, []

            # Base function found
            if t in self._known_base_types:
                base_type = t
                base_params = params

                if "function" in node:
                    # Recurse into inner nodes — inner transforms apply first
                    inner_base, inner_params, inner_transforms = walk(node["function"])
                    if inner_base is not None:
                        base_type = inner_base
                        base_params = inner_params
                    # Inner transforms are already in application order;
                    # they come before any outer transforms
                    return base_type, base_params, inner_transforms

                return base_type, base_params, []

            # Transform node: recurse first (inner), then append this (outer)
            inner_base, inner_params, inner_transforms = walk(node.get("function", {}))

            # Append this transform after inner ones (application order)
            inner_transforms.append((t, params))

            if inner_base is not None:
                return inner_base, inner_params, inner_transforms

            # No base found deeper — return accumulated transforms
            return None, {}, inner_transforms

        base_type, base_params, transforms = walk(config)

        # For compositions, wrap in ParsedConfig with the outer transforms
        if base_type == "composition":
            return ParsedConfig(
                base_type="composition",
                base_params={},
                transforms=transforms,
                is_composition=True,
                raw_composition_config=base_params,  # the full composition node
            )

        # Ensure dimension exists for non-composition base
        if base_type and base_type != "composition" and "dimension" not in base_params:
            if found_dimension is not None:
                base_params["dimension"] = found_dimension
            else:
                base_params["dimension"] = base_params.get("dim") or 2

        return ParsedConfig(
            base_type=base_type,
            base_params=base_params,
            transforms=transforms,
            is_composition=False,
            raw_composition_config=None,
        )

    @staticmethod
    def extract_dimension(config: dict[str, Any]) -> int | None:
        """Walk a config tree for the first explicit integer dimension value."""
        params = config.get("parameters", {})
        if isinstance(params.get("dimension"), int):
            return params["dimension"]

        # Check single inner function
        inner = config.get("function")
        if isinstance(inner, dict):
            result = ConfigParser.extract_dimension(inner)
            if result is not None:
                return result

        # Check functions array
        for func in config.get("functions", []):
            if isinstance(func, dict):
                result = ConfigParser.extract_dimension(func)
                if result is not None:
                    return result

        return None
