import numpy as np
import pytest
from pyMOFL.core.quantized_function import QuantizedFunction
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum

class DummyFunction(OptimizationFunction):
    def __init__(self, bounds):
        self.initialization_bounds = bounds
        self.operational_bounds = bounds
    def evaluate(self, z):
        return np.sum(z)
    def violations(self, x):
        return 0.0

def test_quantized_function_integer():
    bounds = Bounds(low=np.array([0.0]), high=np.array([10.0]))
    base = DummyFunction(bounds)
    qf = QuantizedFunction(base, QuantizationTypeEnum.INTEGER)
    # Should round to nearest integer and clip
    assert qf(np.array([2.7])) == 3.0
    assert qf(np.array([10.9])) == 10.0
    assert qf(np.array([-1.2])) == 0.0

def test_quantized_function_step():
    bounds = Bounds(low=np.array([0.0]), high=np.array([2.0]))
    base = DummyFunction(bounds)
    qf = QuantizedFunction(base, QuantizationTypeEnum.STEP, step=0.5)
    # Should snap to nearest 0.5 and clip
    assert qf(np.array([0.7])) == 0.5
    assert qf(np.array([1.3])) == 1.5
    assert qf(np.array([-0.2])) == 0.0
    assert qf(np.array([2.2])) == 2.0

def test_quantized_function_preserves_bounds():
    bounds = Bounds(low=np.array([1.0]), high=np.array([5.0]))
    base = DummyFunction(bounds)
    qf = QuantizedFunction(base, QuantizationTypeEnum.INTEGER)
    assert np.allclose(qf.operational_bounds.low, [1.0])
    assert np.allclose(qf.operational_bounds.high, [5.0])
    assert qf.operational_bounds.qtype == QuantizationTypeEnum.INTEGER

def test_quantized_function_delegates_violations():
    class ViolatingDummy(DummyFunction):
        def violations(self, x):
            return 1.0
    bounds = Bounds(low=np.array([0.0]), high=np.array([1.0]))
    base = ViolatingDummy(bounds)
    qf = QuantizedFunction(base, QuantizationTypeEnum.INTEGER)
    # Should return np.nan due to violation
    assert np.isnan(qf(np.array([0.5]))) 