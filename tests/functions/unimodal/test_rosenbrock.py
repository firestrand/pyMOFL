"""
Tests for the Rosenbrock function.
"""

import pytest
import numpy as np
from pyMOFL.functions.unimodal import RosenbrockFunction
from pyMOFL.decorators import Biased
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum


class TestRosenbrockFunction:
    """Tests for the Rosenbrock function."""
    
    def test_initialization(self):
        """Test initialization with default and custom bounds."""
        # Test with default bounds
        func = RosenbrockFunction(dimension=2)
        assert func.dimension == 2
        assert func.initialization_bounds.low.shape == (2,)
        assert func.initialization_bounds.high.shape == (2,)
        np.testing.assert_allclose(func.initialization_bounds.low, [-30, -30])
        np.testing.assert_allclose(func.initialization_bounds.high, [30, 30])
        np.testing.assert_allclose(func.operational_bounds.low, [-30, -30])
        np.testing.assert_allclose(func.operational_bounds.high, [30, 30])
        
        # Test with custom bounds
        custom_init_bounds = Bounds(low=np.array([-10, -5]), high=np.array([10, 5]), mode=BoundModeEnum.INITIALIZATION)
        custom_oper_bounds = Bounds(low=np.array([-10, -5]), high=np.array([10, 5]), mode=BoundModeEnum.OPERATIONAL)
        func = RosenbrockFunction(dimension=2, initialization_bounds=custom_init_bounds, operational_bounds=custom_oper_bounds)
        assert func.dimension == 2
        np.testing.assert_allclose(func.initialization_bounds.low, [-10, -5])
        np.testing.assert_allclose(func.initialization_bounds.high, [10, 5])
        np.testing.assert_allclose(func.operational_bounds.low, [-10, -5])
        np.testing.assert_allclose(func.operational_bounds.high, [10, 5])
    
    def test_global_minimum(self):
        """Test the function at its global minimum."""
        for dim in [2, 3, 5]:
            # Get the global minimum
            min_point, min_value = RosenbrockFunction.get_global_minimum(dim)
            
            # Create function
            func = RosenbrockFunction(dimension=dim)
            
            # Verify global minimum point and value
            np.testing.assert_allclose(min_point, np.ones(dim))
            assert min_value == 0.0
            
            # Test at global minimum
            np.testing.assert_allclose(func.evaluate(min_point), min_value)
            
            # Test with bias
            bias_value = 10.0
            biased_func = Biased(func, bias=bias_value)
            np.testing.assert_allclose(biased_func.evaluate(min_point), min_value + bias_value)
    
    def test_function_values(self):
        """Test function values at specific points."""
        func = RosenbrockFunction(dimension=2)
        
        # Test points and expected values
        test_cases = [
            # Global minimum
            (np.array([1, 1]), 0.0),
            
            # Origin point
            (np.array([0, 0]), 1.0),
            
            # Arbitrary point
            (np.array([2, 3]), 101.0)
        ]
        
        # Test each point
        for point, expected in test_cases:
            np.testing.assert_allclose(func.evaluate(point), expected)
        
        # Test with bias decorator
        bias_value = 50.0
        biased_func = Biased(func, bias=bias_value)
        for point, expected in test_cases:
            np.testing.assert_allclose(biased_func.evaluate(point), expected + bias_value)
    
    def test_higher_dimensions(self):
        """Test the function in higher dimensions."""
        func = RosenbrockFunction(dimension=3)
        
        # Test points and expected values
        test_cases = [
            # Global minimum
            (np.array([1, 1, 1]), 0.0),
            
            # Arbitrary point
            (np.array([2, 2, 2]), 802.0)
        ]
        
        # Test each point
        for point, expected in test_cases:
            np.testing.assert_allclose(func.evaluate(point), expected)
        
        # Test with bias decorator
        bias_value = 30.0
        biased_func = Biased(func, bias=bias_value)
        for point, expected in test_cases:
            np.testing.assert_allclose(biased_func.evaluate(point), expected + bias_value)
    
    def test_evaluate_batch(self):
        """Test the batch evaluation method."""
        func = RosenbrockFunction(dimension=2)
        
        # Create a batch of test points
        points = np.array([
            [1, 1],  # global minimum
            [0, 0],  # origin
            [2, 3]   # arbitrary point
        ])
        
        # Get expected values by individual evaluation
        expected = np.array([
            func.evaluate(points[0]),
            func.evaluate(points[1]),
            func.evaluate(points[2])
        ])
        
        # Test batch evaluation
        np.testing.assert_allclose(func.evaluate_batch(points), expected)
        
        # Test with bias decorator
        bias_value = 25.0
        biased_func = Biased(func, bias=bias_value)
        biased_results = biased_func.evaluate_batch(points)
        np.testing.assert_allclose(biased_results, expected + bias_value)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        func = RosenbrockFunction(dimension=2)
        
        # Test with incorrect dimension
        with pytest.raises(ValueError):
            func.evaluate(np.array([1, 2, 3]))
        
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1, 2, 3], [4, 5, 6]]))
    
    def test_bounds_respect(self):
        """Test that bounds are properly respected."""
        # Create function with specific bounds
        custom_init_bounds = Bounds(low=np.array([-10, -10]), high=np.array([10, 10]), mode=BoundModeEnum.INITIALIZATION)
        custom_oper_bounds = Bounds(low=np.array([-10, -10]), high=np.array([10, 10]), mode=BoundModeEnum.OPERATIONAL)
        func = RosenbrockFunction(dimension=2, initialization_bounds=custom_init_bounds, operational_bounds=custom_oper_bounds)
        
        # Points at the bounds
        edge_points = [
            np.array([-10, 0]),
            np.array([10, 0]),
            np.array([0, -10]),
            np.array([0, 10])
        ]
        
        # All points should be valid and evaluate without error
        for point in edge_points:
            # This should execute without error
            value = func.evaluate(point)
            assert isinstance(value, (int, float))
            assert value >= 0 