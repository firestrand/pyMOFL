import numpy as np
import pytest
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum

def test_bounds_metadata_only():
    low = np.array([-5.0, 0.0])
    high = np.array([5.0, 10.0])
    b = Bounds(low=low, high=high, mode=BoundModeEnum.OPERATIONAL, qtype=QuantizationTypeEnum.CONTINUOUS)
    assert np.all(b.low == low)
    assert np.all(b.high == high)
    assert b.mode == BoundModeEnum.OPERATIONAL
    assert b.qtype == QuantizationTypeEnum.CONTINUOUS
    # Bounds does not have project or enforcement methods
    assert not hasattr(b, 'project')

def test_bounds_construction_defaults():
    low = np.array([-5.0, 0.0])
    high = np.array([5.0, 10.0])
    b = Bounds(low=low, high=high)
    assert np.allclose(b.low, low)
    assert np.allclose(b.high, high)
    assert b.mode == BoundModeEnum.OPERATIONAL
    assert b.qtype == QuantizationTypeEnum.CONTINUOUS
    assert b.step == 1.0