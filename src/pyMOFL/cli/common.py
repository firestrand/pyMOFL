"""Shared helpers and formatting utilities for pyMOFL CLI commands."""

from __future__ import annotations

import contextlib
import re
from typing import Any

import numpy as np

from pyMOFL.core.function import OptimizationFunction
from pyMOFL.loader import _PREFIX_TO_SUITE, _discover_suite_configs, _resolve_suite_path
from pyMOFL.utils.suite_config import load_suite_config


def format_vector(vec: np.ndarray, max_elements: int = 5) -> str:
    """Format vector preview cleanly."""
    if len(vec) <= max_elements:
        return f"[{', '.join(f'{v:g}' for v in vec)}]"
    preview = ", ".join(f"{v:g}" for v in vec[:max_elements])
    return f"[{preview}, ...] (dim={len(vec)})"


def format_float(val: float) -> str:
    """Format float with adaptive precision and scientific notation if appropriate."""
    if np.isnan(val):
        return "NaN"
    if np.isinf(val):
        return "Inf" if val > 0 else "-Inf"
    if abs(val) < 1e-4 and val != 0.0:
        return f"{val:.8e}"
    if abs(val) >= 1e7:
        return f"{val:.8e}"
    formatted = f"{val:.8f}".rstrip("0").rstrip(".")
    return formatted if formatted != "-0" else "0"


def format_bounds(low: np.ndarray, high: np.ndarray, dim: int) -> str:
    """Format bounds into a concise readable string."""
    if len(low) == 0 or len(high) == 0:
        return "Unbounded"

    if np.all(low == low[0]) and np.all(high == high[0]):
        l_val = f"{low[0]:.2f}".rstrip("0").rstrip(".") if np.isfinite(low[0]) else str(low[0])
        h_val = f"{high[0]:.2f}".rstrip("0").rstrip(".") if np.isfinite(high[0]) else str(high[0])
        return f"[{l_val}, {h_val}]^{dim}"

    l_preview = ", ".join(f"{v:.1f}" for v in low[: min(dim, 3)])
    h_preview = ", ".join(f"{v:.1f}" for v in high[: min(dim, 3)])
    if dim > 3:
        return f"low=[{l_preview}, ...], high=[{h_preview}, ...]"
    return f"low=[{l_preview}], high=[{h_preview}]"


def find_suite_metadata(
    name_or_id: str, explicit_suite: str | None = None
) -> tuple[str | None, dict[str, Any] | None]:
    """Find suite name and suite entry dictionary for a function ID."""
    suites = _discover_suite_configs()

    # 1. If explicit suite specified
    if explicit_suite:
        clean = explicit_suite.strip().lower()
        if clean in suites:
            with contextlib.suppress(Exception):
                data = load_suite_config(suites[clean])
                for entry in data.get("functions", []):
                    e_id = str(entry.get("id", "")).lower()
                    if name_or_id.lower() in e_id or e_id in name_or_id.lower():
                        return clean, entry

    # 2. Check prefix matching
    for pattern, matched_suite in _PREFIX_TO_SUITE:
        if pattern.search(name_or_id):
            with contextlib.suppress(Exception):
                path = _resolve_suite_path(matched_suite)
                data = load_suite_config(path)
                for entry in data.get("functions", []):
                    e_id = str(entry.get("id", "")).lower()
                    if name_or_id.lower() in e_id or e_id in name_or_id.lower():
                        return matched_suite, entry
            return matched_suite, None

    # 3. Search all suites
    for s_name, s_path in suites.items():
        with contextlib.suppress(Exception):
            data = load_suite_config(s_path)
            for entry in data.get("functions", []):
                e_id = str(entry.get("id", "")).lower()
                if e_id == name_or_id.lower():
                    return s_name, entry

    return None, None


def resolve_optimum(
    func: OptimizationFunction, suite_entry: dict[str, Any] | None = None
) -> tuple[np.ndarray | None, float | None]:
    """Resolve global optimum point x* and value f(x*) through function or transforms."""
    # 1. Direct get_global_minimum()
    with contextlib.suppress(Exception):
        x_min, f_min = func.get_global_minimum()
        return np.asarray(x_min, dtype=np.float64), float(f_min)

    x_opt: np.ndarray | None = None
    f_opt: float | None = None

    # 2. ComposedFunction inspection
    base_f = getattr(func, "base_function", None)
    if base_f is not None:
        with contextlib.suppress(Exception):
            b_x, b_f = base_f.get_global_minimum()
            in_t = getattr(func, "input_transforms", [])
            # If base optimum is at 0 and first transform is shift, then x* is the shift vector
            if len(in_t) >= 1 and type(in_t[0]).__name__ == "ShiftTransform" and np.all(b_x == 0):
                x_opt = in_t[0].shift
            # Accumulate output bias transforms
            bias = 0.0
            for out_t in getattr(func, "output_transforms", []):
                if type(out_t).__name__ == "BiasTransform":
                    bias += getattr(out_t, "bias", 0.0)
            f_opt = b_f + bias

    # 3. Suite entry bias fallback
    if f_opt is None and suite_entry:

        def _find_bias(cfg: Any) -> float | None:
            if not isinstance(cfg, dict):
                return None
            if cfg.get("type") == "bias":
                with contextlib.suppress(ValueError, TypeError):
                    return float(cfg.get("parameters", {}).get("value", 0.0))
            return _find_bias(cfg.get("function"))

        f_opt = _find_bias(suite_entry.get("function"))

    # 4. Global minimum attribute fallback
    if f_opt is None:
        raw_val = getattr(func, "global_minimum", None)
        if raw_val is not None and isinstance(raw_val, (int, float)):
            f_opt = float(raw_val)

    return x_opt, f_opt


def clean_function_name(raw_name: str | None, func_id: str | None, cls_name: str) -> str:
    """Derive a friendly, human-readable function name."""
    if raw_name and raw_name.lower() != "composed":
        return raw_name

    # Try extracting from func_id (e.g. cec17_f01_bent_cigar -> Bent Cigar)
    if func_id:
        m = re.match(r"^[a-zA-Z0-9]+_f\d+_(.+)$", func_id)
        if m:
            return m.group(1).replace("_", " ").title()
        m2 = re.match(r"^[a-zA-Z0-9]+_(.+)$", func_id)
        if m2 and not m2.group(1).startswith("f"):
            return m2.group(1).replace("_", " ").title()

    clean_cls = cls_name.replace("Function", "")
    return clean_cls if clean_cls != "Composed" else (func_id or "Composed")
