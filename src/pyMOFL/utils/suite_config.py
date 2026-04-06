"""Utilities for loading and querying benchmark suite JSON configurations."""

from __future__ import annotations

import contextlib
import copy
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

FILE_REFERENCE_EXTENSIONS = {".txt", ".npy", ".json", ".csv"}
_FUNCTION_CODE_RE = re.compile(r"(?:^|_)f(\d{2})(?:_|$)")


def _extract_function_code(function_id: str) -> str | None:
    """Extract canonical short code like ``f01`` from a function id."""

    match = _FUNCTION_CODE_RE.search(function_id.lower())
    if match is None:
        return None
    return f"f{int(match.group(1)):02d}"


def load_suite_config(path: Path | str) -> dict[str, Any]:
    """Load a suite JSON file and validate it as a mapping."""

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"Suite config '{path}' must be a JSON object")
    return payload


def find_suite_function(suite: Mapping[str, Any], function_id: str) -> dict[str, Any]:
    """Return a suite function entry by id."""

    return find_suite_function_config(suite, function_id)


def load_suite_function_config(
    suite_path: Path | str, function_id: str, *, dimension: int | None = None
) -> dict[str, Any]:
    """Load suite config and return function config by id.

    This helper combines config load + function lookup and optional dimension
    injection so callers stay focused on composition logic.
    """

    suite = load_suite_config(suite_path)
    function_cfg = find_suite_function(suite, function_id)
    if dimension is None:
        return function_cfg
    return inject_dimension(function_cfg, dimension)


def find_suite_function_config(suite: Mapping[str, Any], function_id: str | int) -> dict[str, Any]:
    """Return deep-copied nested function config for a suite function id.

    Supports string IDs (CEC-style ``"cec05_f01_..."``) and integer/numeric
    IDs (BBOB-style ``1``).  A bare numeric query like ``"1"`` or ``1`` will
    match entries whose id is the integer ``1``, the string ``"1"``, or any
    string containing the code ``f01``.
    """
    functions = suite.get("functions", [])
    if not isinstance(functions, list):
        raise TypeError("Suite config must contain a 'functions' list")

    # Normalise query to string
    query = str(function_id)
    target_code = _extract_function_code(query)

    # If the query is a bare integer string (e.g. "1", "10"), also derive
    # a zero-padded function code for matching against integer entry ids.
    bare_numeric_code: str | None = None
    with contextlib.suppress(ValueError):
        bare_numeric_code = f"f{int(query):02d}"

    for function_entry in functions:
        if not isinstance(function_entry, dict):
            continue
        function_entry_id = function_entry.get("id")
        if function_entry_id is None:
            continue

        # Convert entry id to string for uniform matching
        entry_id_str = str(function_entry_id)

        candidate_code = _extract_function_code(entry_id_str)

        # Direct equality (string or stringified integer)
        matches = entry_id_str == query

        # Code-based match (e.g. query "f01" matches "cec05_f01_sphere")
        if target_code is not None and candidate_code == target_code:
            matches = True

        # Bare numeric query match (e.g. query "1" matches entry id 1)
        if bare_numeric_code is not None and candidate_code == bare_numeric_code:
            matches = True

        if not matches:
            continue

        function_cfg = function_entry.get("function")
        if not isinstance(function_cfg, dict):
            raise TypeError(f"Function '{function_id}' has missing or invalid nested config")
        return copy.deepcopy(function_cfg)

    raise ValueError(f"Function id '{function_id}' not found in suite")


def inject_dimension(config: dict[str, Any], dimension: int) -> dict[str, Any]:
    """Recursively inject `parameters.dimension` into nested config nodes."""

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return

        parameters = node.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}
            node["parameters"] = parameters
        parameters.setdefault("dimension", dimension)

        walk(node.get("function"))
        for item in node.get("functions", []) if isinstance(node.get("functions"), list) else []:
            walk(item)

    copied = copy.deepcopy(config)
    walk(copied)
    return copied


def force_inject_dimension(config: dict[str, Any], dimension: int) -> dict[str, Any]:
    """Like inject_dimension but forces dimension override at every level.

    Used by _build_hybrid to ensure component functions get their partition
    dimension, even if an outer inject_dimension already set a different value.
    """

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return

        parameters = node.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}
            node["parameters"] = parameters
        parameters["dimension"] = dimension

        walk(node.get("function"))
        for item in node.get("functions", []) if isinstance(node.get("functions"), list) else []:
            walk(item)

    copied = copy.deepcopy(config)
    walk(copied)
    return copied


def iter_file_references(node: Any, extensions: set[str] | None = None) -> list[str]:
    """Collect string references that look like local resource files."""

    allowed_exts = extensions if extensions is not None else FILE_REFERENCE_EXTENSIONS

    refs: list[str] = []
    if isinstance(node, dict):
        for value in node.values():
            refs.extend(iter_file_references(value, allowed_exts))
    elif isinstance(node, list):
        for item in node:
            refs.extend(iter_file_references(item, allowed_exts))
    elif isinstance(node, str):
        lowered = node.lower()
        if any(lowered.endswith(ext) for ext in allowed_exts) or "/" in node:
            refs.append(node)
    return refs


def supported_dimensions(function_entry: Mapping[str, Any]) -> list[int]:
    """Return sorted, unique supported dimensions declared by a suite function entry."""

    dimensions = function_entry.get("dimensions", {})
    if not isinstance(dimensions, dict):
        return []

    raw = dimensions.get("supported", [])
    if not isinstance(raw, list):
        return []

    dims: list[int] = []
    for value in raw:
        try:
            dim = int(value)
        except (TypeError, ValueError):
            continue
        if dim > 0 and dim not in dims:
            dims.append(dim)
    return sorted(dims)
