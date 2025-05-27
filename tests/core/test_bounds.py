import numpy as np
import pytest
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum

def test_bounds_construction_defaults():
    low = np.array([-5.0, 0.0])
    high = np.array([5.0, 10.0])
    b = Bounds(low=low, high=high)
    assert np.allclose(b.low, low)
    assert np.allclose(b.high, high)
    assert b.mode == BoundModeEnum.OPERATIONAL
    assert b.qtype == QuantizationTypeEnum.CONTINUOUS
    assert b.step == 1.0

def test_bounds_project_continuous():
    b = Bounds(low=np.array([0.0]), high=np.array([1.0]))
    x = np.array([-0.5, 0.5, 1.5])
    projected = b.project(x)
    assert np.allclose(projected, [0.0, 0.5, 1.0])

def test_bounds_project_integer():
    b = Bounds(low=np.array([0.0]), high=np.array([10.0]), qtype=QuantizationTypeEnum.INTEGER)
    x = np.array([-1.2, 3.7, 11.9])
    projected = b.project(x)
    assert np.allclose(projected, [0.0, 4.0, 10.0])
    assert np.all(np.equal(projected, np.round(projected)))

def test_bounds_project_step():
    b = Bounds(low=np.array([0.0]), high=np.array([2.0]), qtype=QuantizationTypeEnum.STEP, step=0.5)
    x = np.array([-0.3, 0.2, 0.7, 1.3, 2.1])
    projected = b.project(x)
    assert np.allclose(projected, [0.0, 0.0, 0.5, 1.5, 2.0])

def test_bounds_docstring():
    assert Bounds.__doc__ is not None
    assert "bounds, quantization, and enforcement mode" in Bounds.__doc__ 