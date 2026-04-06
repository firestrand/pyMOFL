"""Integration checks for CEC 2014 suite config and construction.

These tests intentionally avoid external golden datasets. They ensure the suite
definition is schema-compliant and all functions can be built in normal CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyMOFL.factories.function_factory import DataLoader, FunctionFactory, FunctionRegistry
from pyMOFL.utils import inject_dimension, load_suite_config

SUITE_PATH = Path("src/pyMOFL/constants/cec/2014/cec2014_suite.json")
DATA_DIR = Path("src/pyMOFL/constants/cec/2014")


def test_cec2014_suite_has_required_top_level_fields():
    suite = load_suite_config(SUITE_PATH)
    assert "suite_id" in suite
    assert "name" in suite
    assert "description" in suite
    assert "functions" in suite


def test_cec2014_suite_has_30_functions():
    suite = load_suite_config(SUITE_PATH)
    assert suite["suite_id"] == "cec2014"
    assert len(suite["functions"]) == 30


def test_all_cec2014_functions_construct_and_evaluate_d10():
    suite = load_suite_config(SUITE_PATH)
    factory = FunctionFactory(
        data_loader=DataLoader(base_path=DATA_DIR),
        registry=FunctionRegistry(),
    )

    for func_cfg in suite["functions"]:
        func = factory.create_function(inject_dimension(func_cfg["function"], 10))
        result = func.evaluate(np.zeros(10, dtype=np.float64))
        assert np.isfinite(result), f"{func_cfg['id']} returned non-finite value"
