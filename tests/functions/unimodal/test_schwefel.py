"""
Tests for the Schwefel functions.
"""

import pytest
import numpy as np
from pyMOFL.functions.unimodal import SchwefelFunction12, SchwefelFunction26
from pyMOFL.decorators import BiasedFunction


class TestSchwefelFunction12:
    """Tests for the Schwefel's Problem 1.2 function."""
    
    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Default initialization
        func = SchwefelFunction12(dimension=3)
        assert func.dimension == 3
        assert np.array_equal(func.bounds, np.array([[-100, 100]] * 3))
        
        # Custom bounds
        custom_bounds = np.array([[-10, 10]] * 3)
        func = SchwefelFunction12(dimension=3, bounds=custom_bounds)
        assert np.array_equal(func.bounds, custom_bounds)
    
    def test_global_minimum(self):
        """Test the function at its global minimum."""
        for dim in [2, 3, 5]:
            # Get the global minimum
            min_point, min_value = SchwefelFunction12.get_global_minimum(dim)
            
            # Create function
            func = SchwefelFunction12(dimension=dim)
            
            # Verify global minimum point and value
            np.testing.assert_allclose(min_point, np.zeros(dim))
            assert min_value == 0.0
            
            # Test at global minimum
            np.testing.assert_allclose(func.evaluate(min_point), min_value)
            
            # Test with bias
            bias_value = 10.0
            biased_func = BiasedFunction(func, bias=bias_value)
            np.testing.assert_allclose(biased_func.evaluate(min_point), min_value + bias_value)
    
    def test_function_values(self):
        """Test function values at specific points."""
        func = SchwefelFunction12(dimension=3)
        
        # Test points and expected values
        test_cases = [
            # Global minimum
            (np.array([0.0, 0.0, 0.0]), 0.0),
            
            # Point with all coordinates equal to 1
            (np.array([1.0, 1.0, 1.0]), 14.0),
            
            # Point with different coordinates
            (np.array([2.0, -1.0, 3.0]), 21.0)
        ]
        
        # Test each point
        for point, expected in test_cases:
            np.testing.assert_allclose(func.evaluate(point), expected)
        
        # Test with bias decorator
        bias_value = 5.0
        biased_func = BiasedFunction(func, bias=bias_value)
        for point, expected in test_cases:
            np.testing.assert_allclose(biased_func.evaluate(point), expected + bias_value)
    
    def test_evaluate_batch(self):
        """Test the batch evaluation method."""
        func = SchwefelFunction12(dimension=3)
        
        # Create a batch of test points
        points = np.array([
            [0.0, 0.0, 0.0],  # global minimum
            [1.0, 1.0, 1.0],  # simple point
            [2.0, -1.0, 3.0]  # different coordinates
        ])
        
        # Get expected values by individual evaluation
        expected = np.array([
            func.evaluate(points[0]),
            func.evaluate(points[1]),
            func.evaluate(points[2])
        ])
        
        # Test batch evaluation
        results = func.evaluate_batch(points)
        np.testing.assert_allclose(results, expected)
        
        # Test with bias decorator
        bias_value = 15.0
        biased_func = BiasedFunction(func, bias=bias_value)
        biased_results = biased_func.evaluate_batch(points)
        np.testing.assert_allclose(biased_results, expected + bias_value)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        func = SchwefelFunction12(dimension=3)
        
        # Test with incorrect dimensions
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0]))
        
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0, 4.0]))
        
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0], [3.0, 4.0]]))
        
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))

    def test_bounds_respect(self):
        """Test that bounds are properly respected."""
        # Create function with specific bounds
        bounds = np.array([[-10, 10], [-10, 10], [-10, 10]])
        func = SchwefelFunction12(dimension=3, bounds=bounds)
        
        # Points at the bounds
        edge_points = [
            np.array([-10, 0, 0]),
            np.array([10, 0, 0]),
            np.array([0, -10, 0]),
            np.array([0, 10, 0]),
            np.array([0, 0, -10]),
            np.array([0, 0, 10])
        ]
        
        # All points should be valid and evaluate without error
        for point in edge_points:
            # This should execute without error
            value = func.evaluate(point)
            assert isinstance(value, (int, float))
            assert value >= 0


class TestSchwefelFunction26:
    """Tests for the Schwefel's Problem 2.6 function."""
    
    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Default initialization
        func = SchwefelFunction26(dimension=3)
        assert func.dimension == 3
        assert np.array_equal(func.bounds, np.array([[-100, 100]] * 3))
        assert func.A.shape == (3, 3)
        assert func.B.shape == (3,)
        assert np.array_equal(func.optimum_point, np.zeros(3))
        
        # Custom bounds
        custom_bounds = np.array([[-10, 10]] * 3)
        func = SchwefelFunction26(dimension=3, bounds=custom_bounds)
        assert np.array_equal(func.bounds, custom_bounds)
        
        # Custom A matrix and optimum point
        A = np.eye(3)  # Identity matrix
        optimum_point = np.ones(3)
        func = SchwefelFunction26(dimension=3, A=A, optimum_point=optimum_point)
        assert np.array_equal(func.A, A)
        assert np.array_equal(func.optimum_point, optimum_point)
        assert np.array_equal(func.B, np.ones(3))  # B = A·optimum_point = optimum_point for identity A
    
    def test_global_minimum(self):
        """Test the function at its global minimum."""
        # Use a fixed A and optimum point for reproducibility
        dim = 3
        A = np.eye(dim)
        optimum_point = np.array([1.0, 2.0, 3.0])
        func = SchwefelFunction26(dimension=dim, A=A, optimum_point=optimum_point)
        
        # Get global minimum
        min_point, min_value = func.get_global_minimum()
        
        # Verify global minimum point and value
        np.testing.assert_allclose(min_point, optimum_point)
        assert min_value == 0.0
        
        # Test evaluation at global minimum
        np.testing.assert_allclose(func.evaluate(min_point), min_value)
        
        # Test with bias
        bias_value = 10.0
        biased_func = BiasedFunction(func, bias=bias_value)
        np.testing.assert_allclose(biased_func.evaluate(min_point), min_value + bias_value)
    
    def test_function_values(self):
        """Test function values at specific points."""
        # Use a fixed A and optimum point for reproducibility
        dim = 3
        A = np.eye(dim)
        optimum_point = np.array([1.0, 2.0, 3.0])
        func = SchwefelFunction26(dimension=dim, A=A, optimum_point=optimum_point)
        
        # Test points and expected values
        test_cases = [
            # Global minimum
            (np.array([1.0, 2.0, 3.0]), 0.0),
            
            # Point with first coordinate different
            (np.array([2.0, 2.0, 3.0]), 1.0),
            
            # Point with all coordinates different
            (np.array([0.0, 0.0, 0.0]), 3.0)
        ]
        
        # Test each point
        for point, expected in test_cases:
            np.testing.assert_allclose(func.evaluate(point), expected)
        
        # Test with bias decorator
        bias_value = 5.0
        biased_func = BiasedFunction(func, bias=bias_value)
        for point, expected in test_cases:
            np.testing.assert_allclose(biased_func.evaluate(point), expected + bias_value)
    
    def test_evaluate_batch(self):
        """Test the batch evaluation method."""
        # Use a fixed A and optimum point for reproducibility
        dim = 3
        A = np.eye(dim)
        optimum_point = np.array([1.0, 2.0, 3.0])
        func = SchwefelFunction26(dimension=dim, A=A, optimum_point=optimum_point)
        
        # Create a batch of test points
        points = np.array([
            [1.0, 2.0, 3.0],  # optimum point
            [2.0, 2.0, 3.0],  # first coordinate off by 1
            [0.0, 0.0, 0.0]   # all coordinates different
        ])
        
        # Get expected values by individual evaluation
        expected = np.array([
            func.evaluate(points[0]),
            func.evaluate(points[1]),
            func.evaluate(points[2])
        ])
        
        # Test batch evaluation
        results = func.evaluate_batch(points)
        np.testing.assert_allclose(results, expected)
        
        # Test with bias decorator
        bias_value = 7.5
        biased_func = BiasedFunction(func, bias=bias_value)
        biased_results = biased_func.evaluate_batch(points)
        np.testing.assert_allclose(biased_results, expected + bias_value)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        func = SchwefelFunction26(dimension=3)
        
        # Test with incorrect dimensions
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0]))
        
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0, 4.0]))
        
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0], [3.0, 4.0]]))
        
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
            
    def test_bounds_respect(self):
        """Test that bounds are properly respected."""
        # Create function with specific bounds
        bounds = np.array([[-10, 10], [-10, 10], [-10, 10]])
        func = SchwefelFunction26(dimension=3, bounds=bounds)
        
        # Points at the bounds
        edge_points = [
            np.array([-10, 0, 0]),
            np.array([10, 0, 0]),
            np.array([0, -10, 0]),
            np.array([0, 10, 0]),
            np.array([0, 0, -10]),
            np.array([0, 0, 10])
        ]
        
        # All points should be valid and evaluate without error
        for point in edge_points:
            # This should execute without error
            value = func.evaluate(point)
            assert isinstance(value, (int, float))
            assert value >= 0 