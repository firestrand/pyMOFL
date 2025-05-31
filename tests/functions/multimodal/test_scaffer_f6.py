"""
Tests for the Schaffer's F6 function implementation.
"""

import pytest
import numpy as np
from pyMOFL.functions.multimodal.scaffer_f6 import SchafferF6Function, ScafferF6Function
from pyMOFL.decorators import Biased
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum


class TestSchafferF6Function:
    """Test suite for the Schaffer's F6 function."""
    
    def test_initialization(self):
        """Test the initialization of the function with various parameters."""
        # Default initialization (2D)
        f = SchafferF6Function()
        assert f.dimension == 2
        np.testing.assert_allclose(f.initialization_bounds.low, [-100, -100])
        np.testing.assert_allclose(f.initialization_bounds.high, [100, 100])
        np.testing.assert_allclose(f.operational_bounds.low, [-100, -100])
        np.testing.assert_allclose(f.operational_bounds.high, [100, 100])
        
        # Custom dimension
        f = SchafferF6Function(dimension=3)
        assert f.dimension == 3
        np.testing.assert_allclose(f.initialization_bounds.low, [-100, -100, -100])
        np.testing.assert_allclose(f.initialization_bounds.high, [100, 100, 100])
        np.testing.assert_allclose(f.operational_bounds.low, [-100, -100, -100])
        np.testing.assert_allclose(f.operational_bounds.high, [100, 100, 100])
        
        # Custom bounds
        custom_init_bounds = Bounds(low=np.array([-50, -50]), high=np.array([50, 50]), mode=BoundModeEnum.INITIALIZATION)
        custom_oper_bounds = Bounds(low=np.array([-50, -50]), high=np.array([50, 50]), mode=BoundModeEnum.OPERATIONAL)
        f = SchafferF6Function(initialization_bounds=custom_init_bounds, operational_bounds=custom_oper_bounds)
        np.testing.assert_allclose(f.initialization_bounds.low, [-50, -50])
        np.testing.assert_allclose(f.initialization_bounds.high, [50, 50])
        np.testing.assert_allclose(f.operational_bounds.low, [-50, -50])
        np.testing.assert_allclose(f.operational_bounds.high, [50, 50])
    
    def test_alias(self):
        """Test that the ScafferF6Function alias works."""
        f1 = SchafferF6Function()
        f2 = ScafferF6Function()
        
        x = np.array([1.0, 2.0])
        assert f1.evaluate(x) == f2.evaluate(x)
    
    def test_global_minimum(self):
        """Test the function evaluation at the global minimum."""
        f = SchafferF6Function()
        
        # The global minimum is at (0, 0)
        x_opt = np.zeros(2)
        assert np.isclose(f.evaluate(x_opt), 0.0)
        
        # Add bias using decorator and check
        bias = 5.0
        f_biased = Biased(f, bias=bias)
        assert np.isclose(f_biased.evaluate(x_opt), bias)
    
    def test_evaluate(self):
        """Test the evaluate method with various inputs."""
        f = SchafferF6Function()
        
        # Test with a random point
        x = np.array([1.0, 2.0])
        expected = 0.5 + (np.sin(np.sqrt(5.0))**2 - 0.5) / (1 + 0.001 * 5.0)**2
        assert np.isclose(f.evaluate(x), expected)
        
        # Test with a different point
        x = np.array([-2.0, 3.0])
        expected = 0.5 + (np.sin(np.sqrt(13.0))**2 - 0.5) / (1 + 0.001 * 13.0)**2
        assert np.isclose(f.evaluate(x), expected)
        
        # Test with bias using decorator
        bias = 3.0
        f_biased = Biased(f, bias=bias)
        assert np.isclose(f_biased.evaluate(x), expected + bias)
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        f = SchafferF6Function()
        
        # Test with a batch of points
        X = np.array([
            [0.0, 0.0],  # Global minimum
            [1.0, 2.0],
            [-2.0, 3.0]
        ])
        
        expected = np.array([
            0.0,  # At global minimum
            0.5 + (np.sin(np.sqrt(5.0))**2 - 0.5) / (1 + 0.001 * 5.0)**2,
            0.5 + (np.sin(np.sqrt(13.0))**2 - 0.5) / (1 + 0.001 * 13.0)**2
        ])
        
        results = f.evaluate_batch(X)
        assert np.allclose(results, expected)
        
        # Test with bias using decorator
        bias = 2.5
        f_biased = Biased(f, bias=bias)
        biased_results = f_biased.evaluate_batch(X)
        assert np.allclose(biased_results, expected + bias)
    
    def test_input_validation(self):
        """Test input validation for the evaluate method."""
        f = SchafferF6Function()
        
        # Test with wrong dimension
        with pytest.raises(ValueError):
            f.evaluate(np.array([1.0]))
        
        with pytest.raises(ValueError):
            f.evaluate(np.array([1.0, 2.0, 3.0]))
        
        # Test with non-array input that cannot be converted
        with pytest.raises(Exception):
            f.evaluate("not an array")
    
    def test_batch_input_validation(self):
        """Test input validation for the evaluate_batch method."""
        f = SchafferF6Function()
        
        # Test with wrong batch shape
        with pytest.raises(ValueError):
            f.evaluate_batch(np.array([[1.0], [2.0]]))
        
        with pytest.raises(ValueError):
            f.evaluate_batch(np.array([[1.0, 2.0, 3.0]]))
        
        # Test with non-array input that cannot be properly converted
        with pytest.raises(Exception):
            f.evaluate_batch("not an array")
            
    def test_get_global_minimum(self):
        """Test the get_global_minimum method."""
        for dim in [2, 3, 5]:
            # Get the global minimum
            point, value = SchafferF6Function.get_global_minimum(dim)
            
            # Check the point shape and value
            assert point.shape == (dim,)
            assert np.all(point == 0)
            assert value == 0.0
            
            # Verify that evaluating at the global minimum gives the expected value
            func = SchafferF6Function(dimension=dim)
            assert np.isclose(func.evaluate(point), value)
            
            # Check with bias using decorator
            bias_value = 7.5
            biased_func = Biased(func, bias=bias_value)
            assert np.isclose(biased_func.evaluate(point), value + bias_value) 