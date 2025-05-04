"""
Tests for the Network function.
"""

import pytest
import numpy as np
from pyMOFL.functions.hybrid import NetworkFunction


class TestNetworkFunction:
    """Tests for the Network function."""
    
    def test_initialization(self):
        """Test initialization with default and custom bounds."""
        # Test with default bounds
        func = NetworkFunction()
        assert func.dimension == 42  # 19*2 + 2*2 = 42
        assert func.bounds.shape == (42, 2)
        
        # Check that binary and continuous bounds are set correctly
        binary_count = func.bts_count * func.bsc_count
        
        # Check binary bounds
        for i in range(binary_count):
            assert np.array_equal(func.bounds[i], np.array([0, 1]))
        
        # Check continuous bounds
        for i in range(binary_count, func.dimension):
            assert np.array_equal(func.bounds[i], np.array([0, 20]))
        
        # Test with custom bounds
        custom_bounds = np.array([[0, 0.5]] * 42)
        func = NetworkFunction(bounds=custom_bounds)
        assert func.dimension == 42
        assert func.bounds.shape == (42, 2)
        assert np.array_equal(func.bounds, custom_bounds)
    
    def test_internal_properties(self):
        """Test that internal properties are set correctly."""
        func = NetworkFunction()
        
        # Check BTS count
        assert func.bts_count == 19
        
        # Check BSC count
        assert func.bsc_count == 2
        
        # Check BTS positions
        assert func.bts_positions.shape == (19, 2)
        
        # Check penalty factor
        assert func.penalty == 100.0
    
    def test_evaluate_valid_solution(self):
        """Test evaluation of a valid solution where each BTS connects to exactly one BSC."""
        func = NetworkFunction()
        
        # Create a valid solution:
        # - First 19 binary vars: BTS 0-18 connect to BSC 0
        # - Next 19 binary vars: BTS 0-18 don't connect to BSC 1
        # - BSC 0 at position (5, 5)
        # - BSC 1 at position (15, 15)
        x = np.zeros(42)
        
        # Set connections to BSC 0 for all BTS
        x[:19] = 1.0
        
        # Set BSC positions
        x[38:40] = [5, 5]    # BSC 0
        x[40:42] = [15, 15]  # BSC 1
        
        result = func.evaluate(x)
        
        # Calculate expected value: sum of distances from all BTS to BSC 0
        expected = 0.0
        for i in range(19):
            dx = func.bts_positions[i, 0] - 5
            dy = func.bts_positions[i, 1] - 5
            expected += np.sqrt(dx*dx + dy*dy)
        
        np.testing.assert_allclose(result, expected)
    
    def test_evaluate_invalid_connection(self):
        """Test evaluation with an invalid connection (BTS connected to more than one BSC)."""
        func = NetworkFunction()
        
        # Create an invalid solution where BTS 0 connects to both BSCs
        x = np.zeros(42)
        
        # All BTS connect to BSC 0 except BTS 0 which connects to both
        x[:19] = 1.0
        x[19] = 1.0  # BTS 0 also connects to BSC 1
        
        # Set BSC positions
        x[38:40] = [5, 5]    # BSC 0
        x[40:42] = [15, 15]  # BSC 1
        
        result = func.evaluate(x)
        
        # Calculate expected distances
        distance_sum = 0.0
        for i in range(19):
            dx = func.bts_positions[i, 0] - 5
            dy = func.bts_positions[i, 1] - 5
            distance_sum += np.sqrt(dx*dx + dy*dy)
        
        # Add distance for BTS 0 to BSC 1
        dx0 = func.bts_positions[0, 0] - 15
        dy0 = func.bts_positions[0, 1] - 15
        distance_sum += np.sqrt(dx0*dx0 + dy0*dy0)
        
        # Add penalty for BTS 0 (connected to both BSCs)
        expected = distance_sum + func.penalty
        
        np.testing.assert_allclose(result, expected)
    
    def test_evaluate_missing_connection(self):
        """Test evaluation with a missing connection (BTS not connected to any BSC)."""
        func = NetworkFunction()
        
        # Create an invalid solution where BTS 0 connects to no BSC
        x = np.zeros(42)
        
        # All BTS connect to BSC 0 except BTS 0
        x[1:19] = 1.0
        
        # Set BSC positions
        x[38:40] = [5, 5]    # BSC 0
        x[40:42] = [15, 15]  # BSC 1
        
        result = func.evaluate(x)
        
        # Calculate expected distances
        distance_sum = 0.0
        for i in range(1, 19):  # Skip BTS 0
            dx = func.bts_positions[i, 0] - 5
            dy = func.bts_positions[i, 1] - 5
            distance_sum += np.sqrt(dx*dx + dy*dy)
        
        # Add penalty for BTS 0 (not connected to any BSC)
        expected = distance_sum + func.penalty
        
        np.testing.assert_allclose(result, expected)
    
    def test_evaluate_batch(self):
        """Test batch evaluation."""
        func = NetworkFunction()
        
        # Create a batch of test points
        # 1. Valid solution: all BTS connect to BSC 0
        # 2. Valid solution: all BTS connect to BSC 1
        # 3. Invalid: no connections
        
        # Solution 1
        x1 = np.zeros(42)
        x1[:19] = 1.0
        x1[38:42] = [5, 5, 15, 15]  # BSC positions
        
        # Solution 2
        x2 = np.zeros(42)
        x2[19:38] = 1.0
        x2[38:42] = [5, 5, 15, 15]  # BSC positions
        
        # Solution 3
        x3 = np.zeros(42)
        x3[38:42] = [5, 5, 15, 15]  # BSC positions
        
        batch = np.vstack([x1, x2, x3])
        
        # Evaluate batch
        results = func.evaluate_batch(batch)
        
        # Verify each result individually
        for i in range(batch.shape[0]):
            expected = func.evaluate(batch[i])
            np.testing.assert_allclose(results[i], expected)
    
    def test_non_negativity(self):
        """Test that function values are non-negative for points in the domain."""
        func = NetworkFunction()
        
        # Generate random points in the domain
        rng = np.random.default_rng(42)
        
        # For binary part
        binary_part = rng.random(size=(100, func.bts_count * func.bsc_count))
        
        # For continuous part (BSC positions)
        continuous_part = rng.uniform(0, 20, size=(100, 2 * func.bsc_count))
        
        # Combine
        points = np.hstack([binary_part, continuous_part])
        
        # Evaluate function at these points
        values = func.evaluate_batch(points)
        
        # Check that all values are non-negative
        assert (values >= 0).all()
    
    def test_with_bias(self):
        """Test function with a bias value."""
        bias = 10.0
        func = NetworkFunction(bias=bias)
        
        # Create a valid solution
        x = np.zeros(42)
        x[:19] = 1.0  # All BTS connect to BSC 0
        x[38:42] = [5, 5, 15, 15]  # BSC positions
        
        result_with_bias = func.evaluate(x)
        result_without_bias = NetworkFunction().evaluate(x)
        
        np.testing.assert_allclose(result_with_bias, result_without_bias + bias) 