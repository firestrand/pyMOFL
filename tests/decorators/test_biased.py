"""
Tests for the BiasedFunction decorator.
"""

import pytest
import numpy as np
from pyMOFL.functions.unimodal import SphereFunction
from pyMOFL.functions.multimodal import AckleyFunction
from pyMOFL.decorators import BiasedFunction


class TestBiasedFunction:
    """Tests for the BiasedFunction decorator."""
    
    def test_initialization(self):
        """Test initialization with different bias values."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Test with default bias (0.0)
        biased_func = BiasedFunction(base_func)
        assert biased_func.dimension == 2
        assert biased_func.bias == 0.0
        assert np.array_equal(biased_func.bounds, base_func.bounds)
        
        # Test with positive bias
        bias = 10.0
        biased_func = BiasedFunction(base_func, bias)
        assert biased_func.dimension == 2
        assert biased_func.bias == 10.0
        assert np.array_equal(biased_func.bounds, base_func.bounds)
        
        # Test with negative bias
        bias = -5.0
        biased_func = BiasedFunction(base_func, bias)
        assert biased_func.dimension == 2
        assert biased_func.bias == -5.0
        assert np.array_equal(biased_func.bounds, base_func.bounds)
    
    def test_evaluate(self):
        """Test the evaluate method."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        bias = 5.0
        biased_func = BiasedFunction(base_func, bias)
        
        # Test at origin (0, 0)
        # SphereFunction(0, 0) = 0, so BiasedFunction(0, 0) = 0 + 5 = 5
        assert biased_func.evaluate(np.array([0.0, 0.0])) == 5.0
        
        # Test at point (1, 2)
        # SphereFunction(1, 2) = 1² + 2² = 5, so BiasedFunction(1, 2) = 5 + 5 = 10
        assert biased_func.evaluate(np.array([1.0, 2.0])) == 10.0
        
        # Test with a different function (Ackley)
        base_func_ackley = AckleyFunction(dimension=2)
        biased_func_ackley = BiasedFunction(base_func_ackley, bias)
        
        # Test at global minimum (0, 0)
        # AckleyFunction(0, 0) = 0, so BiasedFunction(0, 0) = 0 + 5 = 5
        assert np.isclose(biased_func_ackley.evaluate(np.array([0.0, 0.0])), 5.0, atol=1e-15)
        
        # Test at arbitrary point
        x = np.array([1.0, 1.0])
        expected = base_func_ackley.evaluate(x) + bias
        assert np.isclose(biased_func_ackley.evaluate(x), expected)
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        bias = 3.0
        biased_func = BiasedFunction(base_func, bias)
        
        # Test with batch of vectors
        X = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
        
        # Calculate expected values manually
        # SphereFunction(0, 0) = 0, so BiasedFunction(0, 0) = 0 + 3 = 3
        # SphereFunction(1, 2) = 1² + 2² = 5, so BiasedFunction(1, 2) = 5 + 3 = 8
        # SphereFunction(3, 4) = 3² + 4² = 25, so BiasedFunction(3, 4) = 25 + 3 = 28
        expected = np.array([3.0, 8.0, 28.0])
        
        np.testing.assert_allclose(biased_func.evaluate_batch(X), expected)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        biased_func = BiasedFunction(base_func, 1.0)
        
        # Test with incorrect dimension
        with pytest.raises(ValueError):
            biased_func.evaluate(np.array([1, 2, 3]))
        
        with pytest.raises(ValueError):
            biased_func.evaluate_batch(np.array([[1, 2, 3], [4, 5, 6]]))
    
    def test_nested_bias(self):
        """Test that biasing can be nested."""
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Create a biased function
        bias1 = 2.0
        biased_func1 = BiasedFunction(base_func, bias1)
        
        # Create a nested biased function
        bias2 = 3.0
        biased_func2 = BiasedFunction(biased_func1, bias2)
        
        # Test at point (0, 0)
        # SphereFunction(0, 0) = 0
        # biased_func1(0, 0) = 0 + 2 = 2
        # biased_func2(0, 0) = 2 + 3 = 5
        assert biased_func2.evaluate(np.array([0.0, 0.0])) == 5.0
        
        # Test at point (1, 2)
        # SphereFunction(1, 2) = 1² + 2² = 5
        # biased_func1(1, 2) = 5 + 2 = 7
        # biased_func2(1, 2) = 7 + 3 = 10
        assert biased_func2.evaluate(np.array([1.0, 2.0])) == 10.0
        
        # Test with batch
        X = np.array([[0.0, 0.0], [1.0, 2.0]])
        expected = np.array([5.0, 10.0])
        np.testing.assert_allclose(biased_func2.evaluate_batch(X), expected)
    
    def test_with_other_decorators(self):
        """Test biasing combined with other decorators like shift."""
        # Import ShiftedFunction
        from pyMOFL.decorators import ShiftedFunction
        
        # Create a base function
        base_func = SphereFunction(dimension=2)
        
        # Create a shifted function
        shift = np.array([1.0, 2.0])
        shifted_func = ShiftedFunction(base_func, shift)
        
        # Create a biased shifted function
        bias = 3.0
        biased_shifted_func = BiasedFunction(shifted_func, bias)
        
        # Test at the shifted global minimum (1, 2)
        # ShiftedSphereFunction(1, 2) = 0
        # BiasedShiftedSphereFunction(1, 2) = 0 + 3 = 3
        assert np.isclose(biased_shifted_func.evaluate(np.array([1.0, 2.0])), 3.0)
        
        # Test at origin (0, 0)
        # ShiftedSphereFunction(0, 0) = SphereFunction(0-1, 0-2) = SphereFunction(-1, -2) = 1² + 2² = 5
        # BiasedShiftedSphereFunction(0, 0) = 5 + 3 = 8
        assert np.isclose(biased_shifted_func.evaluate(np.array([0.0, 0.0])), 8.0)
        
        # Create a biased function first, then shifted
        biased_func = BiasedFunction(base_func, bias)
        shifted_biased_func = ShiftedFunction(biased_func, shift)
        
        # Test at the shifted global minimum (1, 2)
        # BiasedSphereFunction(1-1, 2-2) = BiasedSphereFunction(0, 0) = 0 + 3 = 3
        assert np.isclose(shifted_biased_func.evaluate(np.array([1.0, 2.0])), 3.0)
        
        # Both decorators should produce equivalent results for the same points
        X = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(
            biased_shifted_func.evaluate_batch(X), 
            shifted_biased_func.evaluate_batch(X)
        ) 