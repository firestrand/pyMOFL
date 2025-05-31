import numpy as np
import pytest
from pyMOFL.decorators import Quantized, Biased, Shifted
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.functions.unimodal import SphereFunction

class DummyFunction(OptimizationFunction):
    def __init__(self, bounds):
        self.initialization_bounds = bounds
        self.operational_bounds = bounds
    def evaluate(self, z):
        return np.sum(z)
    def violations(self, x):
        return 0.0

def test_quantized_base_integer():
    # Quantized as a base: should return quantized value
    qf = Quantized(dimension=1, qtype=QuantizationTypeEnum.INTEGER,
                   initialization_bounds=Bounds(low=np.array([0.0]), high=np.array([10.0]), mode=BoundModeEnum.INITIALIZATION, qtype=QuantizationTypeEnum.INTEGER),
                   operational_bounds=Bounds(low=np.array([0.0]), high=np.array([10.0]), mode=BoundModeEnum.OPERATIONAL, qtype=QuantizationTypeEnum.INTEGER))
    assert qf(np.array([2.7])) == 3.0
    assert qf(np.array([10.9])) == 10.0
    assert qf(np.array([-1.2])) == 0.0

def test_quantized_base_step():
    # Quantized as a base: should return quantized value (step)
    qf = Quantized(dimension=1, qtype=QuantizationTypeEnum.STEP, step=0.5,
                   initialization_bounds=Bounds(low=np.array([0.0]), high=np.array([2.0]), mode=BoundModeEnum.INITIALIZATION, qtype=QuantizationTypeEnum.STEP, step=0.5),
                   operational_bounds=Bounds(low=np.array([0.0]), high=np.array([2.0]), mode=BoundModeEnum.OPERATIONAL, qtype=QuantizationTypeEnum.STEP, step=0.5))
    assert qf(np.array([0.7])) == 0.5
    assert qf(np.array([1.3])) == 1.5
    assert qf(np.array([-0.2])) == 0.0
    assert qf(np.array([2.2])) == 2.0

def test_quantized_decorator_integer():
    # Quantized as a decorator: should pass quantized input to base function
    bounds = Bounds(low=np.array([0.0]), high=np.array([10.0]), mode=BoundModeEnum.OPERATIONAL, qtype=QuantizationTypeEnum.CONTINUOUS)
    base = DummyFunction(bounds)
    qf = Quantized(base_function=base, qtype=QuantizationTypeEnum.INTEGER)
    # The DummyFunction returns the sum of the quantized input
    assert qf(np.array([2.7])) == 3.0
    assert qf(np.array([10.9])) == 10.0
    assert qf(np.array([-1.2])) == 0.0

def test_quantized_decorator_step():
    # Quantized as a decorator: should pass quantized input to base function (step)
    bounds = Bounds(low=np.array([0.0]), high=np.array([2.0]), mode=BoundModeEnum.OPERATIONAL, qtype=QuantizationTypeEnum.CONTINUOUS)
    base = DummyFunction(bounds)
    qf = Quantized(base_function=base, qtype=QuantizationTypeEnum.STEP, step=0.5)
    assert qf(np.array([0.7])) == 0.5
    assert qf(np.array([1.3])) == 1.5
    assert qf(np.array([-0.2])) == 0.0
    assert qf(np.array([2.2])) == 2.0

def test_quantized_function_preserves_bounds():
    # Quantized as a decorator: operational_bounds should have correct qtype
    bounds = Bounds(low=np.array([1.0]), high=np.array([5.0]), mode=BoundModeEnum.OPERATIONAL, qtype=QuantizationTypeEnum.CONTINUOUS)
    base = DummyFunction(bounds)
    qf = Quantized(base_function=base, qtype=QuantizationTypeEnum.INTEGER)
    assert np.allclose(qf.operational_bounds.low, [1.0])
    assert np.allclose(qf.operational_bounds.high, [5.0])
    assert qf.operational_bounds.qtype == QuantizationTypeEnum.INTEGER

def test_quantized_function_delegates_violations():
    # Quantized should delegate violations to base function
    class ViolatingDummy(DummyFunction):
        def violations(self, x):
            return 1.0
    bounds = Bounds(low=np.array([0.0]), high=np.array([1.0]), mode=BoundModeEnum.OPERATIONAL, qtype=QuantizationTypeEnum.CONTINUOUS)
    base = ViolatingDummy(bounds)
    qf = Quantized(base_function=base, qtype=QuantizationTypeEnum.INTEGER)
    # Should return np.nan due to violation
    assert np.isnan(qf(np.array([0.5])))

def test_quantized_chain_with_biased():
    # Chaining: Quantized -> Biased -> DummyFunction
    bounds = Bounds(low=np.array([0.0]), high=np.array([10.0]), mode=BoundModeEnum.OPERATIONAL, qtype=QuantizationTypeEnum.CONTINUOUS)
    base = DummyFunction(bounds)
    biased = Biased(base_function=base, bias=5.0)
    qf = Quantized(base_function=biased, qtype=QuantizationTypeEnum.INTEGER)
    # Should quantize, then sum, then add bias
    assert qf(np.array([2.7])) == 8.0  # 3 + 5
    assert qf(np.array([10.9])) == 15.0  # 10 + 5
    assert qf(np.array([-1.2])) == 5.0  # 0 + 5

def test_quantized_chain_with_shifted():
    # Chaining: Quantized -> Shifted -> DummyFunction
    bounds = Bounds(low=np.array([0.0]), high=np.array([10.0]), mode=BoundModeEnum.OPERATIONAL, qtype=QuantizationTypeEnum.CONTINUOUS)
    base = DummyFunction(bounds)
    shifted = Shifted(base_function=base, shift=np.array([1.0]))
    qf = Quantized(base_function=shifted, qtype=QuantizationTypeEnum.INTEGER)
    # Should quantize, then shift, then sum
    assert qf(np.array([2.7])) == 2.0  # (3 - 1)
    assert qf(np.array([10.9])) == 9.0  # (10 - 1)
    assert qf(np.array([-1.2])) == -1.0  # (0 - 1)

def test_quantized_with_real_base():
    # Quantized as a decorator on a real function (SphereFunction)
    sphere = SphereFunction(dimension=2)
    qf = Quantized(base_function=sphere, qtype=QuantizationTypeEnum.INTEGER)
    # Should quantize input, then compute sum of squares
    x = np.array([1.7, 2.2])
    # Quantized: [2, 2], Sphere: 2^2 + 2^2 = 8
    assert qf(x) == 8.0
    x = np.array([-1.8, 0.2])
    # Quantized: [-2, 0], Sphere: 4 + 0 = 4
    assert qf(x) == 4.0

def test_quantized_base_integer_batch() -> None:
    """
    Test batch evaluation for Quantized as a base function (integer quantization).
    """
    qf = Quantized(dimension=1, qtype=QuantizationTypeEnum.INTEGER,
                   initialization_bounds=Bounds(low=np.array([0.0]), high=np.array([10.0]), mode=BoundModeEnum.INITIALIZATION, qtype=QuantizationTypeEnum.INTEGER),
                   operational_bounds=Bounds(low=np.array([0.0]), high=np.array([10.0]), mode=BoundModeEnum.OPERATIONAL, qtype=QuantizationTypeEnum.INTEGER))
    X = np.array([[2.7], [10.9], [-1.2]])
    expected = np.array([[3.0], [10.0], [0.0]])
    result = qf.evaluate_batch(X)
    assert np.allclose(result, expected)

def test_quantized_decorator_integer_batch() -> None:
    """
    Test batch evaluation for Quantized as a decorator (integer quantization).
    """
    bounds = Bounds(low=np.array([0.0]), high=np.array([10.0]), mode=BoundModeEnum.OPERATIONAL, qtype=QuantizationTypeEnum.CONTINUOUS)
    base = DummyFunction(bounds)
    qf = Quantized(base_function=base, qtype=QuantizationTypeEnum.INTEGER)
    X = np.array([[2.7], [10.9], [-1.2]])
    expected = np.array([3.0, 10.0, 0.0])
    result = qf.evaluate_batch(X)
    assert np.allclose(result, expected)

def test_quantized_per_variable_step() -> None:
    """
    Test per-variable step quantization in Quantized as a base function.
    """
    steps = np.array([0.5, 2.0])
    qf = Quantized(dimension=2, qtype=QuantizationTypeEnum.STEP, step=steps,
                   initialization_bounds=Bounds(low=np.array([0.0, 0.0]), high=np.array([2.0, 4.0]), mode=BoundModeEnum.INITIALIZATION, qtype=QuantizationTypeEnum.STEP, step=steps),
                   operational_bounds=Bounds(low=np.array([0.0, 0.0]), high=np.array([2.0, 4.0]), mode=BoundModeEnum.OPERATIONAL, qtype=QuantizationTypeEnum.STEP, step=steps))
    x = np.array([0.7, 3.1])
    # Quantized: [0.5, 4.0]
    assert np.allclose(qf(x), np.array([0.5, 4.0]))
    X = np.array([[0.7, 3.1], [1.3, 1.9]])
    expected = np.array([[0.5, 4.0], [1.5, 2.0]])
    result = qf.evaluate_batch(X)
    assert np.allclose(result, expected)

def test_quantized_deep_composability() -> None:
    """
    Test deep composability: Quantized -> Shifted -> Biased -> DummyFunction.
    """
    bounds = Bounds(low=np.array([0.0]), high=np.array([10.0]), mode=BoundModeEnum.OPERATIONAL, qtype=QuantizationTypeEnum.CONTINUOUS)
    base = DummyFunction(bounds)
    biased = Biased(base_function=base, bias=2.0)
    shifted = Shifted(base_function=biased, shift=np.array([1.0]))
    qf = Quantized(base_function=shifted, qtype=QuantizationTypeEnum.INTEGER)
    # Should quantize, then shift, then sum, then add bias
    # x = 2.7 -> quantize 3, shift 2, sum 2, bias 4
    assert qf(np.array([2.7])) == 4.0  # (3 - 1) + 2
    assert qf(np.array([10.9])) == 11.0  # (10 - 1) + 2
    assert qf(np.array([-1.2])) == 1.0  # (0 - 1) + 2

def test_quantized_bounds_delegation() -> None:
    """
    Test that bounds are correctly delegated through multiple decorator layers.
    """
    bounds = Bounds(low=np.array([1.0]), high=np.array([5.0]), mode=BoundModeEnum.OPERATIONAL, qtype=QuantizationTypeEnum.CONTINUOUS)
    base = DummyFunction(bounds)
    shifted = Shifted(base_function=base, shift=np.array([1.0]))
    qf = Quantized(base_function=shifted, qtype=QuantizationTypeEnum.INTEGER)
    # The operational_bounds should reflect the quantized decorator
    assert np.allclose(qf.operational_bounds.low, [1.0])
    assert np.allclose(qf.operational_bounds.high, [5.0])
    assert qf.operational_bounds.qtype == QuantizationTypeEnum.INTEGER
    # The initialization_bounds should be delegated from the base
    assert np.allclose(qf.initialization_bounds.low, [1.0])
    assert np.allclose(qf.initialization_bounds.high, [5.0]) 