"""
Tests for the Gear Train function.
"""

import pytest
import numpy as np
from pyMOFL.functions.multimodal.gear_train import GearTrainFunction
from pyMOFL.decorators import Biased
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum


class TestGearTrainFunction:
    """Tests for the Gear Train function."""
    
    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        func = GearTrainFunction()
        assert func.dimension == 4
        np.testing.assert_allclose(func.initialization_bounds.low, [12, 12, 12, 12])
        np.testing.assert_allclose(func.initialization_bounds.high, [60, 60, 60, 60])
        np.testing.assert_allclose(func.operational_bounds.low, [12, 12, 12, 12])
        np.testing.assert_allclose(func.operational_bounds.high, [60, 60, 60, 60])
        
        custom_init_bounds = Bounds(low=np.array([10, 10, 10, 10]), high=np.array([70, 70, 70, 70]), mode=BoundModeEnum.INITIALIZATION)
        custom_oper_bounds = Bounds(low=np.array([10, 10, 10, 10]), high=np.array([70, 70, 70, 70]), mode=BoundModeEnum.OPERATIONAL)
        func = GearTrainFunction(initialization_bounds=custom_init_bounds, operational_bounds=custom_oper_bounds)
        np.testing.assert_allclose(func.initialization_bounds.low, [10, 10, 10, 10])
        np.testing.assert_allclose(func.initialization_bounds.high, [70, 70, 70, 70])
        np.testing.assert_allclose(func.operational_bounds.low, [10, 10, 10, 10])
        np.testing.assert_allclose(func.operational_bounds.high, [70, 70, 70, 70])
    
    def test_evaluate_global_minimum(self):
        """Test the evaluate method at the global minimum."""
        func = GearTrainFunction()
        
        # Test at the known global minimum
        # (z₁, z₂, z₃, z₄) = (16, 19, 43, 49)
        best_design = np.array([16, 19, 43, 49])
        result = func.evaluate(best_design)
        
        # The expected error at the global minimum is approximately 1.643428e-6
        assert np.isclose(result, 1.643428e-6, atol=1e-8)
        
        # Test with bias decorator
        bias_value = 0.1
        biased_func = Biased(func, bias=bias_value)
        result_with_bias = biased_func.evaluate(best_design)
        assert np.isclose(result_with_bias, result + bias_value, atol=1e-8)
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        func = GearTrainFunction()
        
        # Create a batch with the global minimum and another point
        batch = np.array([
            [16, 19, 43, 49],  # Global minimum
            [20, 30, 40, 50]   # Some other point
        ])
        
        # Calculate expected results individually
        expected = np.array([
            func.evaluate(batch[0]),
            func.evaluate(batch[1])
        ])
        
        # Test batch evaluation
        results = func.evaluate_batch(batch)
        assert results.shape == (2,)
        np.testing.assert_allclose(results, expected)
        
        # Test with bias decorator
        bias_value = 0.5
        biased_func = Biased(func, bias=bias_value)
        biased_results = biased_func.evaluate_batch(batch)
        np.testing.assert_allclose(biased_results, results + bias_value)
    
    def test_ratio_calculation(self):
        """Test that the gear ratio is calculated correctly."""
        func = GearTrainFunction()
        
        # Check the target ratio
        assert np.isclose(func._R_TARGET, 0.144279, rtol=1e-5)
        
        # Test with a simple design where we can easily verify
        design = np.array([20, 20, 20, 20])
        result = func.evaluate(design)
        
        # Simple ratio is 1.0, which is 0.855721 away from the target
        expected_error = np.abs(1.0 - func._R_TARGET)
        assert np.isclose(result, expected_error)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        func = GearTrainFunction()
        
        # Test with incorrect dimension
        with pytest.raises(ValueError):
            func.evaluate(np.array([1, 2, 3]))
        
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1, 2, 3], [4, 5, 6]]))
    
    def test_non_negativity(self):
        """Test that function values are non-negative."""
        func = GearTrainFunction()
        
        # Generate random points in the domain
        rng = np.random.default_rng(42)
        points = rng.uniform(12, 60, size=(100, 4))
        
        # Evaluate function at these points
        values = func.evaluate_batch(points)
        
        # Check that all values are >= 0
        assert (values >= 0.0).all()
    
    def test_get_global_minimum(self):
        """Test the static get_global_minimum method."""
        min_point, min_value = GearTrainFunction.get_global_minimum()
        np.testing.assert_allclose(min_point, np.array([16, 19, 43, 49]))
        assert np.isclose(min_value, 1.643428e-6, atol=1e-8) 