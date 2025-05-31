"""
Tests for the Sphere function (refactored for new bounds logic).
"""

import pytest
import numpy as np
from pyMOFL.functions.unimodal import SphereFunction
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.decorators import Biased


class TestSphereFunction:
    """Tests for the Sphere function."""
    
    def test_initialization_defaults(self):
        """Test initialization with default bounds."""
        func = SphereFunction(dimension=2)
        assert func.dimension == 2
        assert isinstance(func.initialization_bounds, Bounds)
        assert isinstance(func.operational_bounds, Bounds)
        np.testing.assert_allclose(func.initialization_bounds.low, [-100, -100])
        np.testing.assert_allclose(func.initialization_bounds.high, [100, 100])
        np.testing.assert_allclose(func.operational_bounds.low, [-100, -100])
        np.testing.assert_allclose(func.operational_bounds.high, [100, 100])
    
    def test_initialization_custom_bounds(self):
        """Test initialization with custom bounds."""
        init_bounds = Bounds(low=np.array([-10, -5]), high=np.array([10, 5]), mode=BoundModeEnum.INITIALIZATION)
        op_bounds = Bounds(low=np.array([-1, -2]), high=np.array([1, 2]), mode=BoundModeEnum.OPERATIONAL)
        func = SphereFunction(dimension=2, initialization_bounds=init_bounds, operational_bounds=op_bounds)
        np.testing.assert_allclose(func.initialization_bounds.low, [-10, -5])
        np.testing.assert_allclose(func.initialization_bounds.high, [10, 5])
        np.testing.assert_allclose(func.operational_bounds.low, [-1, -2])
        np.testing.assert_allclose(func.operational_bounds.high, [1, 2])
    
    def test_evaluate_and_enforcement(self):
        """Test the evaluate method and enforcement."""
        op_bounds = Bounds(low=np.array([0, 0]), high=np.array([1, 1]), mode=BoundModeEnum.OPERATIONAL)
        func = SphereFunction(dimension=2, operational_bounds=op_bounds)
        
        # Test in bounds
        assert func(np.array([0.5, 0.5])) == 0.5
        
        # Test out of bounds (should be clipped)
        assert func(np.array([2.0, -1.0])) == 1.0  # [1, 0] -> 1^2 + 0^2 = 1
    
    def test_quantization_integer(self):
        """Test quantization with integer type."""
        op_bounds = Bounds(low=np.array([0]), high=np.array([10]), qtype=QuantizationTypeEnum.INTEGER)
        func = SphereFunction(dimension=1, operational_bounds=op_bounds)
        
        # Test rounding and clipping
        assert func(np.array([2.7])) == 9.0  # 3^2
        assert func(np.array([10.9])) == 100.0  # 10^2
        assert func(np.array([-1.2])) == 0.0  # 0^2
    
    def test_quantization_step(self):
        """Test quantization with step type."""
        op_bounds = Bounds(low=np.array([0]), high=np.array([2]), qtype=QuantizationTypeEnum.STEP, step=0.5)
        func = SphereFunction(dimension=1, operational_bounds=op_bounds)
        
        # Test snapping and clipping
        assert func(np.array([0.7])) == 0.25  # 0.5^2
        assert func(np.array([1.3])) == 2.25  # 1.5^2
        assert func(np.array([-0.2])) == 0.0  # 0^2
        assert func(np.array([2.2])) == 4.0  # 2^2
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        func = SphereFunction(dimension=2)
        
        # Test with batch of vectors
        X = np.array([[0, 0], [1, 1], [2, 3]])
        expected = np.array([0, 2, 13])
        np.testing.assert_allclose(func.evaluate_batch(X), expected)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        func = SphereFunction(dimension=2)
        
        # Test with incorrect dimension
        with pytest.raises(ValueError):
            func.evaluate(np.array([1, 2, 3]))
        
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1, 2, 3], [4, 5, 6]]))
    
    def test_global_minimum(self):
        """Test the global minimum function."""
        for dim in [2, 5, 10]:
            # Get the global minimum
            point, value = SphereFunction.get_global_minimum(dim)
            
            # Check the point shape and value
            assert point.shape == (dim,)
            assert np.all(point == 0)
            assert value == 0.0
            
            # Verify that evaluating at the global minimum gives the expected value
            func = SphereFunction(dimension=dim)
            assert np.isclose(func.evaluate(point), value)
            
            # Check with bias
            bias_value = 3.0
            biased_func = Biased(func, bias=bias_value)
            assert np.isclose(biased_func.evaluate(point), value + bias_value) 