"""GNBG-II Suite Factory.

Programmatic factory for the 24 GNBG-II benchmark functions (Yazdani et al. 2023).
Constructs problem instances from the bundled GNBG suite configuration using
FunctionFactory and MinComposition.

References
----------
.. [1] Yazdani, D., et al. (2023). "GNBG: A Generalized and Configurable Benchmark
       Generator for Continuous Numerical Optimization." arXiv:2312.07083.
.. [2] Salgotra, R., et al. (2025). "Numerical Global Optimization Competition
       on GNBG-II generated Test Suite." In GECCO '25 Companion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pyMOFL.core.function import OptimizationFunction
from pyMOFL.factories.data_loader import DataLoader
from pyMOFL.factories.function_factory import FunctionFactory
from pyMOFL.utils.suite_config import inject_dimension, load_suite_config


def _default_gnbg_path() -> Path:
    """Return default path to GNBG suite constants."""
    return Path(__file__).resolve().parent.parent / "constants" / "gnbg"


def _normalize_fid(fid: int | str) -> str:
    """Normalize function ID to 'gnbg_fXX' format."""
    if isinstance(fid, int):
        return f"gnbg_f{fid:02d}"
    s = str(fid).strip().lower()
    if s.startswith("gnbg_f"):
        return s
    if s.startswith("f") and s[1:].isdigit():
        return f"gnbg_f{int(s[1:]):02d}"
    if s.isdigit():
        return f"gnbg_f{int(s):02d}"
    return s


class GNBGSuiteFactory:
    """Factory for the GNBG-II benchmark suite (24 functions).

    Parameters
    ----------
    data_path : str or Path, optional
        Path to directory containing gnbg_suite.json and rotation/shift data.
        If None, uses the bundled package constants directory.
    """

    SUPPORTED_DIMENSIONS: list[int] = [2, 10, 30, 50, 100]
    NUM_FUNCTIONS: int = 24

    def __init__(self, data_path: str | Path | None = None) -> None:
        self._data_path = Path(data_path) if data_path is not None else _default_gnbg_path()
        if (self._data_path / "cec2025_suite.json").exists():
            self._suite_config_path = self._data_path / "cec2025_suite.json"
        elif (self._data_path / "gnbg_suite.json").exists():
            self._suite_config_path = self._data_path / "gnbg_suite.json"
        else:
            raise FileNotFoundError(f"GNBG suite config not found at: {self._data_path}")

        self._suite_data = load_suite_config(self._suite_config_path)
        self._data_loader = DataLoader(base_path=self._data_path)
        self._factory = FunctionFactory(data_loader=self._data_loader)

        # Index function configs by normalized ID and 1-based integer index
        self._functions_by_id: dict[str, dict[str, Any]] = {}
        for i, entry in enumerate(self._suite_data.get("functions", []), start=1):
            if isinstance(entry, dict) and "id" in entry:
                self._functions_by_id[entry["id"]] = entry
                self._functions_by_id[str(i)] = entry

    @property
    def suite_id(self) -> str:
        """Return suite identifier."""
        return str(self._suite_data.get("suite_id", "gnbg_suite"))

    @property
    def name(self) -> str:
        """Return suite name."""
        return str(self._suite_data.get("name", "GNBG-II Generated Suite"))

    def _resolve_entry(self, fid: int | str) -> dict[str, Any] | None:
        norm_id = _normalize_fid(fid)
        entry = self._functions_by_id.get(norm_id) or self._functions_by_id.get(str(fid))
        if entry is None and norm_id.startswith("gnbg_f"):
            alt_id = norm_id.replace("gnbg_f", "cec25_f")
            entry = self._functions_by_id.get(alt_id)
        if entry is None and norm_id.startswith("cec25_f"):
            alt_id = norm_id.replace("cec25_f", "gnbg_f")
            entry = self._functions_by_id.get(alt_id)
        return entry

    def create_function(self, fid: int | str, dim: int = 10) -> OptimizationFunction:
        """Create a single GNBG function instance.

        Parameters
        ----------
        fid : int or str
            Function index (1-24) or ID string (e.g. 'gnbg_f01', 'cec25_f01', 'f01').
        dim : int, optional
            Dimension of the function (default: 10). Must be one of [2, 10, 30, 50, 100].

        Returns
        -------
        OptimizationFunction
            Instantiated function.
        """
        entry = self._resolve_entry(fid)
        if entry is None:
            raise ValueError(f"Unknown GNBG function '{fid}'. Supported IDs: f01 to f24.")
        supported_dims = entry.get("dimensions", {}).get("supported", self.SUPPORTED_DIMENSIONS)
        if dim not in supported_dims:
            norm_id = entry.get("id", str(fid))
            raise ValueError(
                f"Dimension {dim} not supported for {norm_id}. Supported: {supported_dims}"
            )

        raw_config = entry["function"]
        config_with_dim = inject_dimension(raw_config, dim)
        return self._factory.create_function(config_with_dim)

    def create_suite(self, dim: int = 10) -> list[OptimizationFunction]:
        """Create all 24 functions of the GNBG suite for a given dimension.

        Parameters
        ----------
        dim : int, optional
            Dimension of the functions (default: 10).

        Returns
        -------
        list[OptimizationFunction]
            List of 24 instantiated functions.
        """
        return [self.create_function(fid=i, dim=dim) for i in range(1, self.NUM_FUNCTIONS + 1)]

    def get_function_info(
        self, fid: int | str | None = None
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Get metadata for a specific function or all functions in the suite.

        Parameters
        ----------
        fid : int or str, optional
            Function identifier. If None, returns info for all functions.

        Returns
        -------
        dict or list[dict]
            Metadata dictionary or list of dictionaries.
        """
        if fid is not None:
            entry = self._resolve_entry(fid)
            if entry is None:
                raise ValueError(f"Unknown GNBG function '{fid}'.")
            return dict(entry)

        return [dict(e) for e in self._suite_data.get("functions", [])]
