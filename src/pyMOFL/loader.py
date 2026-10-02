"""High-level unified API facade for loading and accessing benchmark functions and suites."""

from __future__ import annotations

import contextlib
import inspect
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pyMOFL.core.function import OptimizationFunction
from pyMOFL.factories.data_loader import DataLoader
from pyMOFL.factories.function_factory import FunctionFactory
from pyMOFL.registry import _COMPONENTS, scan_package
from pyMOFL.utils.suite_config import (
    _extract_function_code,
    inject_dimension,
    load_suite_config,
    load_suite_function_config,
)

_PREFIX_TO_SUITE: list[tuple[re.Pattern[str], str]] = [
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
        self._by_key: dict[str, OptimizationFunction] = {}

        for idx, func in enumerate(self):
            # Indexing (0-based and 1-based string numbers)
            self._by_key[str(idx)] = func
            self._by_key[str(idx + 1)] = func

            # Canonical function ID or name
            func_id = getattr(func, "function_id", getattr(func, "name", None))
            if func_id:
                s_id = str(func_id).strip().lower()
                self._by_key[s_id] = func
                code = _extract_function_code(s_id)
                if code:
                    self._by_key[code] = func
                    self._by_key[code.lstrip("f")] = func

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, (int, slice)):
            return super().__getitem__(key)
        if isinstance(key, str):
            k = key.strip().lower()
            if k in self._by_key:
                return self._by_key[k]
            code = _extract_function_code(k)
            if code and code in self._by_key:
                return self._by_key[code]
            raise KeyError(f"Function '{key}' not found in suite '{self.suite_id}'")
        raise TypeError(f"Invalid key type: {type(key)}")

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
    scan_package()

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
        config = load_suite_function_config(suite_path, str(name_or_id), dimension=dimension)
        data_loader = DataLoader(base_path=suite_path.parent)
        func_factory = FunctionFactory(data_loader=data_loader)
        func = func_factory.create_function(config)
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
        func = factory.create_function(func_cfg)
        func.function_id = str(entry["id"]) if "id" in entry else None
        func.name = str(entry["name"]) if "name" in entry else None
        funcs.append(func)

    return BenchmarkSuite(
        funcs,
        suite_id=str(suite_payload.get("suite_id", suite_id)),
        dimension=dimension,
        name=str(suite_payload.get("name", suite_id)),
    )
