"""High-level unified API facade for loading and accessing benchmark functions and suites."""

from __future__ import annotations

import contextlib
import inspect
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.factories.data_loader import DataLoader
from pyMOFL.factories.function_factory import FunctionFactory
from pyMOFL.registry import _COMPONENTS, _discover_builtins
from pyMOFL.registry import scan_package as scan_package
from pyMOFL.utils.suite_config import (
    _extract_function_code,
    find_suite_function_config,
    inject_dimension,
    load_suite_config,
)

_PREFIX_TO_SUITE: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^spso2007(?:_|$)", re.IGNORECASE), "spso2007"),
    (re.compile(r"^spso2011(?:_|$)", re.IGNORECASE), "spso2011"),
    (re.compile(r"^bbob(?:_|$)", re.IGNORECASE), "bbob"),
    (re.compile(r"^gnbg(?:_|$)", re.IGNORECASE), "gnbg"),
    (re.compile(r"^cec(?:05|2005)(?:_|$)", re.IGNORECASE), "cec2005"),
    (re.compile(r"^cec(?:08|2008)(?:_|$)", re.IGNORECASE), "cec2008"),
    (re.compile(r"^cec(?:10|2010)(?:_|$)", re.IGNORECASE), "cec2010"),
    (re.compile(r"^cec(?:13|2013)_lsgo(?:_|$)", re.IGNORECASE), "cec2013_lsgo"),
    (re.compile(r"^cec(?:13|2013)(?:_|$)", re.IGNORECASE), "cec2013"),
    (re.compile(r"^cec(?:14|2014)(?:_|$)", re.IGNORECASE), "cec2014"),
    (re.compile(r"^cec(?:15|2015)_nich(?:ing)?(?:_|$)", re.IGNORECASE), "cec2015_niching"),
    (re.compile(r"^cec(?:15|2015)(?:_|$)", re.IGNORECASE), "cec2015"),
    (re.compile(r"^cec(?:17|2017)(?:_|$)", re.IGNORECASE), "cec2017"),
    (re.compile(r"^cec(?:19|2019)(?:_|$)", re.IGNORECASE), "cec2019"),
    (re.compile(r"^cec(?:20|2020)(?:_|$)", re.IGNORECASE), "cec2020"),
    (re.compile(r"^cec(?:21|2021)(?:_|$)", re.IGNORECASE), "cec2021"),
    (re.compile(r"^cec(?:22|2022)(?:_|$)", re.IGNORECASE), "cec2022"),
    (re.compile(r"^cec(?:24|2024)(?:_|$)", re.IGNORECASE), "cec2024"),
    (re.compile(r"^cec(?:25|2025)(?:_|$)", re.IGNORECASE), "cec2025"),
]


def _constants_root() -> Path:
    """Return the package constants root directory."""
    return Path(__file__).resolve().parent / "constants"


def _discover_suite_configs() -> dict[str, Path]:
    """Discover all suite JSON files and build an alias-to-path mapping."""
    suites: dict[str, Path] = {}
    for p in sorted(_constants_root().rglob("*_suite.json")):
        if not p.is_file():
            continue
        try:
            payload = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        raw_id = str(payload.get("suite_id", "")).strip().lower()
        stem_id = p.stem.replace("_suite", "").strip().lower()

        for key in filter(None, [raw_id, stem_id]):
            suites[key] = p

        # Add common convenience aliases (e.g. cec2017 -> cec17, cec05 -> cec2005)
        m = re.match(r"^cec(?:20)?(\d{2})(.*)$", stem_id)
        if m:
            yy, suffix = m.group(1), m.group(2)
            suites[f"cec{yy}{suffix}"] = p
            suites[f"cec20{yy}{suffix}"] = p

    # Standard well-known aliases
    if "bbob_noiseless" in suites:
        suites["bbob"] = suites["bbob_noiseless"]
    if "gnbg_suite" in suites:
        suites["gnbg"] = suites["gnbg_suite"]
    if "cec2005_suite" in suites:
        suites["cec2005"] = suites["cec2005_suite"]
        suites["cec05"] = suites["cec2005_suite"]

    return suites


def _resolve_suite_path(suite_id: str) -> Path:
    """Resolve a suite ID or alias to a filesystem Path."""
    suites = _discover_suite_configs()
    key = suite_id.strip().lower()
    if key in suites:
        return suites[key]

    # Try matching by prefix or stem
    for candidate_key, path in suites.items():
        if candidate_key == key or candidate_key.replace("_suite", "") == key:
            return path

    available = ", ".join(sorted(set(suites.keys())))
    raise ValueError(f"No suite configuration found for '{suite_id}'. Available: {available}")


def _fixed_entry(payload: dict[str, Any], identifier: str) -> dict[str, Any]:
    """Resolve an entry in a declared fixed heterogeneous suite."""
    query = identifier.strip().lower()
    number = re.fullmatch(r"f?(\d+)", query)
    code = f"f{int(number.group(1)):02d}" if number else None
    for entry in payload["functions"]:
        identity = str(entry["id"]).lower()
        if identity == query or (code is not None and _extract_function_code(identity) == code):
            return entry
    raise ValueError(f"Function '{identifier}' not found in fixed suite")


def _apply_fixed_bounds(function: OptimizationFunction, entry: dict[str, Any]) -> None:
    """Apply only explicit fixed-suite metadata, without enforcing bounds."""
    if function.dimension != entry["dimension"]:
        raise ValueError("Fixed suite entry dimension disagrees with its function")
    metadata = entry["bounds"]
    low = np.asarray(metadata["low"], dtype=np.float64)
    high = np.asarray(metadata["high"], dtype=np.float64)
    steps = np.asarray(metadata["steps"], dtype=np.float64)
    if any(values.shape != (function.dimension,) for values in (low, high, steps)):
        raise ValueError("Fixed suite bounds and steps must match entry dimension")
    qtype = np.array(
        [
            QuantizationTypeEnum.CONTINUOUS
            if step == 0
            else QuantizationTypeEnum.INTEGER
            if step == 1
            else QuantizationTypeEnum.STEP
            for step in steps
        ]
    )
    fractional = np.unique(steps[(steps != 0) & (steps != 1)])
    if len(fractional) > 1:
        raise ValueError("Bounds metadata supports one fractional step per fixed entry")
    step = float(fractional[0]) if len(fractional) else 1.0
    function.initialization_bounds = Bounds(
        low.copy(), high.copy(), BoundModeEnum.INITIALIZATION, qtype.copy(), step
    )
    function.operational_bounds = Bounds(
        low.copy(), high.copy(), BoundModeEnum.OPERATIONAL, qtype.copy(), step
    )


class BenchmarkSuite(list[OptimizationFunction]):
    """Container representing a benchmark suite of OptimizationFunction instances.

    Inherits from list, enabling standard indexing, iteration, and slicing.
    Also provides dictionary-style lookup by function ID, name, or short code.
    """

    def __init__(
        self,
        functions: Iterable[OptimizationFunction],
        suite_id: str,
        dimension: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(functions)
        self.suite_id = suite_id
        self.dimension = dimension
        self.name = name or suite_id

    def __getitem__(self, key: Any) -> Any:
        """Resolve current names/IDs or canonical numbers; preserve list indexing.

        Ambiguous named matches raise ValueError, including repeated entries.
        Numeric strings identify function numbers, never positions in this list.
        """
        if isinstance(key, (int, slice)):
            return super().__getitem__(key)
        if not isinstance(key, str):
            raise TypeError(f"Invalid key type: {type(key)}")
        normalized = key.strip().lower()
        short_match = re.fullmatch(r"f?(\d+)", normalized)
        short_code = f"f{int(short_match.group(1)):02d}" if short_match else None
        qualified_code = re.fullmatch(r".+_f\d+", normalized)
        found = None
        for function in self:
            identifiers = (
                str(value).strip().lower()
                for value in (
                    getattr(function, "function_id", None),
                    getattr(function, "name", None),
                )
                if value
            )
            matches = any(
                identifier == normalized
                or (short_code is not None and _extract_function_code(identifier) == short_code)
                or (qualified_code is not None and identifier.startswith(normalized + "_"))
                for identifier in identifiers
            )
            if matches:
                if found is not None:
                    raise ValueError(f"Ambiguous function '{key}' in suite '{self.suite_id}'")
                found = function
        if found is None:
            raise KeyError(f"Function '{key}' not found in suite '{self.suite_id}'")
        return found

    def get(self, key: int | str, default: Any = None) -> Any:
        """Get function by index or ID, returning default if not found."""
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def __repr__(self) -> str:
        dim_str = f", dimension={self.dimension}" if self.dimension is not None else ""
        return f"<BenchmarkSuite suite_id='{self.suite_id}' functions={len(self)}{dim_str}>"


def load(
    name_or_id: str | int,
    dimension: int | None = None,
    *,
    suite: str | None = None,
    instance: int | None = None,
    **kwargs: Any,
) -> OptimizationFunction:
    """Load and instantiate a benchmark optimization function.

    Parameters
    ----------
    name_or_id : str or int
        Function name, registry alias (e.g. 'sphere', 'rosenbrock'), or suite function ID
        (e.g. 'cec17_f01', 'bbob_f05', 1).
    dimension : int, optional
        Target dimensionality for the function.
    suite : str, optional
        Explicit suite identifier (e.g. 'cec2017', 'bbob', 'gnbg').
    instance : int, optional
        Instance identifier for randomized benchmark suites (e.g. BBOB instance ID).
    **kwargs : Any
        Additional keyword arguments passed to the constructor.

    Returns
    -------
    OptimizationFunction
        Instantiated, callable optimization function.
    """
    _discover_builtins()

    # 1. Suite specified explicitly
    if suite is not None:
        suite_clean = suite.strip().lower()
        if suite_clean in {"bbob", "bbob_noiseless"}:
            from pyMOFL.factories.bbob_suite_factory import BBOBSuiteFactory

            code = _extract_function_code(str(name_or_id))
            raw_str = code if code is not None else str(name_or_id)
            digits = re.search(r"\d+", raw_str)
            if digits is None:
                raise ValueError(f"Could not extract numeric function ID from '{name_or_id}'")
            fid = int(digits.group(0))

            factory = BBOBSuiteFactory()
            func = factory.create_function(fid=fid, iid=instance or 1, dim=dimension or 2, **kwargs)
            func.function_id = f"bbob_f{fid:02d}"
            return func

        if suite_clean in {"gnbg", "gnbg_suite"}:
            from pyMOFL.factories.gnbg_suite_factory import GNBGSuiteFactory

            factory = GNBGSuiteFactory()
            code = _extract_function_code(str(name_or_id))
            raw_str = code if code is not None else str(name_or_id)
            digits = re.search(r"\d+", raw_str)
            fid = int(digits.group(0)) if digits else name_or_id
            func = factory.create_function(fid=fid, dim=dimension or 10, **kwargs)
            return func

        # Config-driven suite
        suite_path = _resolve_suite_path(suite)
        payload = load_suite_config(suite_path)
        fixed_entry = None
        if payload.get("dimension_policy") == "fixed_heterogeneous":
            fixed_entry = _fixed_entry(payload, str(name_or_id))
            if dimension is not None and dimension != fixed_entry["dimension"]:
                raise ValueError("dimension must match the fixed suite entry")
            config = fixed_entry["function"]
        else:
            config = find_suite_function_config(payload, str(name_or_id))
            if dimension is not None:
                config = inject_dimension(config, dimension)
        data_loader = DataLoader(base_path=suite_path.parent)
        func_factory = FunctionFactory(data_loader=data_loader)
        func = func_factory.create_function(
            config, fixed_dimension=fixed_entry["dimension"] if fixed_entry is not None else None
        )
        if fixed_entry is not None:
            _apply_fixed_bounds(func, fixed_entry)
        func.function_id = str(name_or_id)
        return func

    # 2. No suite specified: check classical component registry first
    query_str = str(name_or_id).strip()
    cls = _COMPONENTS.get(query_str.lower()) or _COMPONENTS.get(query_str)
    if cls is None:
        # Check by class __name__
        for comp in _COMPONENTS.values():
            if isinstance(comp, type) and comp.__name__.lower() == query_str.lower():
                cls = comp
                break
    if cls is not None and isinstance(cls, type) and issubclass(cls, OptimizationFunction):
        if dimension is not None:
            return cls(dimension=dimension, **kwargs)
        sig = inspect.signature(cls.__init__)
        dim_param = sig.parameters.get("dimension")
        if dim_param is not None and dim_param.default == inspect.Parameter.empty:
            raise ValueError(f"Function '{name_or_id}' requires a 'dimension' parameter.")
        return cls(**kwargs)

    # 3. Detect suite from name prefix (e.g. cec17_f01, bbob_f01, gnbg_f01)
    for pattern, matched_suite in _PREFIX_TO_SUITE:
        if pattern.search(query_str):
            return load(
                name_or_id, dimension=dimension, suite=matched_suite, instance=instance, **kwargs
            )

    # 4. Search all suite configs for matching function ID
    suites = _discover_suite_configs()
    matching_suites: list[tuple[str, Path]] = []
    for s_name, s_path in suites.items():
        with contextlib.suppress(Exception):
            suite_data = load_suite_config(s_path)
            # Try finding entry
            for entry in suite_data.get("functions", []):
                e_id = str(entry.get("id", ""))
                if e_id.lower() == query_str.lower() or _extract_function_code(
                    e_id
                ) == _extract_function_code(query_str):
                    matching_suites.append((s_name, s_path))
                    break

    # De-duplicate by canonical path
    unique_suites = {p: name for name, p in matching_suites}
    if len(unique_suites) == 1:
        _path, s_name = next(iter(unique_suites.items()))
        return load(name_or_id, dimension=dimension, suite=s_name, instance=instance, **kwargs)
    if len(unique_suites) > 1:
        names = ", ".join(unique_suites.values())
        raise ValueError(
            f"Ambiguous function ID '{name_or_id}' found in multiple suites: {names}. "
            f"Please specify suite='...' explicitly."
        )

    raise ValueError(f"Could not find function '{name_or_id}' in registry or benchmark suites.")


def get_suite(
    suite_id: str,
    dimension: int | None = None,
    *,
    instance: int | None = None,
    **kwargs: Any,
) -> BenchmarkSuite:
    """Instantiate an entire benchmark suite of functions.

    Parameters
    ----------
    suite_id : str
        Suite identifier (e.g. 'cec2017', 'bbob', 'gnbg', 'cec2005').
    dimension : int, optional
        Target dimensionality for all functions in the suite.
    instance : int, optional
        Instance identifier for randomized benchmark suites (e.g. BBOB instance ID).
    **kwargs : Any
        Additional keyword arguments passed to suite construction.

    Returns
    -------
    BenchmarkSuite
        A list-compatible container containing all functions in the suite.
    """
    suite_clean = suite_id.strip().lower()

    # BBOB Noiseless
    if suite_clean in {"bbob", "bbob_noiseless"}:
        from pyMOFL.factories.bbob_suite_factory import BBOBSuiteFactory

        factory = BBOBSuiteFactory()
        funcs: list[OptimizationFunction] = []
        dim = dimension or 2
        iid = instance or 1
        for fid in range(1, 25):
            f = factory.create_function(fid=fid, iid=iid, dim=dim, **kwargs)
            f.function_id = f"bbob_f{fid:02d}"
            funcs.append(f)
        return BenchmarkSuite(
            funcs,
            suite_id="bbob_noiseless",
            dimension=dim,
            name="BBOB Noiseless Benchmark Suite",
        )

    # GNBG Suite
    if suite_clean in {"gnbg", "gnbg_suite"}:
        from pyMOFL.factories.gnbg_suite_factory import GNBGSuiteFactory

        factory = GNBGSuiteFactory()
        funcs = []
        dim = dimension or 10
        for fid in range(1, factory.NUM_FUNCTIONS + 1):
            f = factory.create_function(fid=fid, dim=dim, **kwargs)
            f.function_id = f"gnbg_f{fid:02d}"
            funcs.append(f)
        return BenchmarkSuite(
            funcs,
            suite_id="gnbg_suite",
            dimension=dim,
            name="GNBG-II Benchmark Suite",
        )

    # Config-driven suites (CEC, LSGO, Niching)
    suite_path = _resolve_suite_path(suite_id)
    suite_payload = load_suite_config(suite_path)
    functions_list = suite_payload.get("functions", [])
    fixed = suite_payload.get("dimension_policy") == "fixed_heterogeneous"
    if fixed and dimension is not None:
        raise ValueError("A fixed heterogeneous suite has no single dimension; omit dimension")
    if not isinstance(functions_list, list):
        raise TypeError(f"Suite config '{suite_path}' does not contain a 'functions' list")

    data_loader = DataLoader(base_path=suite_path.parent)
    factory = FunctionFactory(data_loader=data_loader)

    funcs = []
    for entry in functions_list:
        if not isinstance(entry, dict):
            continue
        func_cfg = entry.get("function")
        if not isinstance(func_cfg, dict):
            continue
        if dimension is not None:
            func_cfg = inject_dimension(func_cfg, dimension)
        func = factory.create_function(
            func_cfg, fixed_dimension=entry["dimension"] if fixed else None
        )
        if fixed:
            _apply_fixed_bounds(func, entry)
        func.function_id = str(entry["id"]) if "id" in entry else None
        func.name = str(entry["name"]) if "name" in entry else None
        funcs.append(func)

    return BenchmarkSuite(
        funcs,
        suite_id=str(suite_payload.get("suite_id", suite_id)),
        dimension=dimension,
        name=str(suite_payload.get("name", suite_id)),
    )
