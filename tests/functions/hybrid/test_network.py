"""
Tests for the Network function.
"""

import pytest
import numpy as np
from pyMOFL.functions.hybrid.network import NetworkFunction
from pyMOFL.decorators import Biased
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum


class TestNetworkFunction:
    """Tests for the Network function."""
    
    def test_initialization(self):
        """Test initialization with default and custom bounds."""
        func = NetworkFunction()
        assert func.dimension == 42  # 19*2 + 2*2 = 42
        # Check binary and continuous bounds in initialization_bounds and operational_bounds
        binary_count = func.bts_count * func.bsc_count
        cont_count = 2 * func.bsc_count
        np.testing.assert_allclose(func.initialization_bounds.low[:binary_count], 0)
        np.testing.assert_allclose(func.initialization_bounds.high[:binary_count], 1)
        np.testing.assert_allclose(func.initialization_bounds.low[binary_count:], 0)
        np.testing.assert_allclose(func.initialization_bounds.high[binary_count:], 20)
        np.testing.assert_allclose(func.operational_bounds.low[:binary_count], 0)
        np.testing.assert_allclose(func.operational_bounds.high[:binary_count], 1)
        np.testing.assert_allclose(func.operational_bounds.low[binary_count:], 0)
        np.testing.assert_allclose(func.operational_bounds.high[binary_count:], 20)

        # Test with custom bounds
        custom_init_bounds = Bounds(
            low=np.zeros(42),
            high=np.full(42, 0.5),
            mode=BoundModeEnum.INITIALIZATION,
            qtype=np.array([QuantizationTypeEnum.CONTINUOUS]*42)
        )
        custom_oper_bounds = Bounds(
            low=np.zeros(42),
            high=np.full(42, 0.5),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=np.array([QuantizationTypeEnum.CONTINUOUS]*42)
        )
        func = NetworkFunction(initialization_bounds=custom_init_bounds, operational_bounds=custom_oper_bounds)
        np.testing.assert_allclose(func.initialization_bounds.low, np.zeros(42))
        np.testing.assert_allclose(func.initialization_bounds.high, np.full(42, 0.5))
        np.testing.assert_allclose(func.operational_bounds.low, np.zeros(42))
        np.testing.assert_allclose(func.operational_bounds.high, np.full(42, 0.5))
    
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
        
        # Test with Biased decorator
        bias_value = 10.0
        biased_func = Biased(func, bias=bias_value)
        biased_result = biased_func.evaluate(x)
        
        # Check that bias is correctly applied
        assert np.isclose(biased_result, result + bias_value)
    
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
        
        # Evaluate
        result = func.evaluate(x)
        
        # The result should include a penalty
        assert result > 0.0
        
        # Create a similar solution without the invalid connection
        x_valid = x.copy()
        x_valid[19] = 0.0  # Remove the connection of BTS 0 to BSC 1
        
        # The valid solution should have a lower value
        assert func.evaluate(x) > func.evaluate(x_valid)
    
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
        
        # Evaluate
        result = func.evaluate(x)
        
        # The result should include a penalty
        assert result > 0.0
        
        # Create a similar solution without the missing connection
        x_valid = x.copy()
        x_valid[0] = 1.0  # Add connection of BTS 0 to BSC 0
        
        # The valid solution should have a lower value
        assert func.evaluate(x) > func.evaluate(x_valid)
    
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
        
        # Calculate expected results individually
        expected = np.array([
            func.evaluate(batch[0]),
            func.evaluate(batch[1]),
            func.evaluate(batch[2])
        ])
        
        # Verify results match individual evaluations
        np.testing.assert_allclose(results, expected)
        
        # Test with Biased decorator
        bias_value = 5.0
        biased_func = Biased(func, bias=bias_value)
        biased_results = biased_func.evaluate_batch(batch)
        
        # Check that bias is correctly applied to all results
        np.testing.assert_allclose(biased_results, results + bias_value)
    
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