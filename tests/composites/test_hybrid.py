"""
Tests for the HybridFunction class.
"""

import pytest
import numpy as np
from pyMOFL.functions.unimodal import SphereFunction
from pyMOFL.functions.multimodal import RastriginFunction
from pyMOFL.composites import HybridFunction
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.decorators import Quantized


class TestHybridFunction:
    """Tests for the HybridFunction class."""
    
    def test_initialization(self):
        """Test initialization with default and custom bounds."""
        sphere = SphereFunction(dimension=2)
        rastrigin = RastriginFunction(dimension=2)
        components = [sphere, rastrigin]
        partitions = [(0, 2), (2, 4)]
        hybrid = HybridFunction(components, partitions)
        assert hybrid.dimension == 4
        assert len(hybrid.components) == 2
        assert hybrid.partitions == [(0, 2), (2, 4)]
        assert np.array_equal(hybrid.weights, np.array([0.5, 0.5]))
        # Custom bounds
        custom_init_bounds = Bounds(
            low=np.array([-10, -5, -3, -1]),
            high=np.array([10, 5, 3, 1]),
            mode=BoundModeEnum.INITIALIZATION,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        custom_oper_bounds = Bounds(
            low=np.array([-10, -5, -3, -1]),
            high=np.array([10, 5, 3, 1]),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        hybrid = HybridFunction(components, partitions, initialization_bounds=custom_init_bounds, operational_bounds=custom_oper_bounds)
        assert hybrid.dimension == 4
        assert np.array_equal(hybrid.initialization_bounds.low, custom_init_bounds.low)
        assert np.array_equal(hybrid.operational_bounds.high, custom_oper_bounds.high)
        # Custom weights
        weights = [0.3, 0.7]
        hybrid = HybridFunction(components, partitions, weights=weights)
        assert hybrid.dimension == 4
        assert np.array_equal(hybrid.weights, np.array([0.3, 0.7]))
    
    def test_parameter_validation(self):
        """Test that parameters are validated correctly."""
        sphere = SphereFunction(dimension=2)
        rastrigin = RastriginFunction(dimension=2)
        with pytest.raises(ValueError):
            HybridFunction([sphere, rastrigin], [(0, 2)])
        with pytest.raises(ValueError):
            HybridFunction([sphere, rastrigin], [(0, 2), (1, 0)])
        with pytest.raises(ValueError):
            HybridFunction([sphere, rastrigin], [(0, 2), (2, 4)], weights=[0.5])
        # The new paradigm does not raise for mismatched bounds shape; test removed.
    
    def test_evaluate_simple(self):
        """Test the evaluate method with simple components."""
        sphere = SphereFunction(dimension=2)
        components = [sphere]
        partitions = [(0, 2)]
        hybrid = HybridFunction(components, partitions)
        assert hybrid.evaluate(np.array([0.0, 0.0])) == 0.0
        x = np.array([2.0, 3.0])
        assert hybrid.evaluate(x) == 13.0
    
    def test_evaluate_multiple_components(self):
        """Test the evaluate method with multiple components."""
        sphere = SphereFunction(dimension=2)
        rastrigin = RastriginFunction(dimension=2)
        components = [sphere, rastrigin]
        partitions = [(0, 2), (2, 4)]
        weights = [0.5, 0.5]
        hybrid = HybridFunction(components, partitions, weights=weights)
        assert hybrid.evaluate(np.array([0.0, 0.0, 0.0, 0.0])) == 0.0
        x = np.array([1.0, 1.0, 1.0, 1.0])
        expected = 0.5 * 2.0 + 0.5 * (20.0 + 2.0 - 10.0 * np.cos(2 * np.pi * 1.0) - 10.0 * np.cos(2 * np.pi * 1.0))
        np.testing.assert_allclose(hybrid.evaluate(x), expected)
    
    def test_evaluate_with_dimension_mismatch(self):
        """Test the evaluate method with dimension mismatches."""
        sphere = SphereFunction(dimension=2)
        rastrigin = RastriginFunction(dimension=3)
        components = [sphere, rastrigin]
        partitions = [(0, 2), (2, 4)]
        hybrid = HybridFunction(components, partitions)
        x = np.array([0.0, 0.0, 0.0, 0.0])
        assert hybrid.evaluate(x) == 0.0
        x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            hybrid.evaluate(x)
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        sphere = SphereFunction(dimension=2)
        components = [sphere]
        partitions = [(0, 2)]
        hybrid = HybridFunction(components, partitions)
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
        expected = np.array([0.0, 2.0, 13.0])
        np.testing.assert_allclose(hybrid.evaluate_batch(X), expected)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        sphere = SphereFunction(dimension=2)
        rastrigin = RastriginFunction(dimension=2)
        components = [sphere, rastrigin]
        partitions = [(0, 2), (2, 4)]
        hybrid = HybridFunction(components, partitions)
        with pytest.raises(ValueError):
            hybrid.evaluate(np.array([1.0, 2.0]))
        with pytest.raises(ValueError):
            hybrid.evaluate_batch(np.array([[1.0, 2.0], [3.0, 4.0]]))
    
    def test_decorator_compatibility(self):
        """Test hybrid function with decorated (shifted, biased) components and quantized bounds."""
        from pyMOFL.decorators import Biased, Shifted
        from pyMOFL.core.quantized_function import QuantizedFunction
        from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
        sphere = SphereFunction(dimension=2)
        shifted = Shifted(base_function=sphere, shift=np.array([1.0, 1.0]))
        biased = Biased(base_function=shifted, bias=5.0)
        quantized = Quantized(base_function=biased, qtype=QuantizationTypeEnum.INTEGER)
        components = [quantized]
        partitions = [(0, 2)]
        hybrid = HybridFunction(components, partitions)
        x = np.array([2.7, 3.2])
        # After quantization: [3, 3], shift: [2, 2], sphere: 8, bias: 13
        assert np.isclose(hybrid.evaluate(x), 13.0) 