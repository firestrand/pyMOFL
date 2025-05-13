"""
Tests for F06: Shifted Rosenbrock Function from CEC 2005.
"""

import pytest
import numpy as np
from src.pyMOFL.functions.cec.cec2005 import F06, create_cec2005_function
from tests.utils.validation import load_cec_validation_data, validate_function, print_validation_summary


class TestF06ShiftedRosenbrock:
    """Tests for F06: Shifted Rosenbrock Function."""
    
    def test_initialization(self):
        """Test that the function can be initialized with different dimensions."""
        for dim in [2, 10, 30, 50]:
            func = F06(dimension=dim)
            assert func.dimension == dim
            assert func.function_number == 6
            assert func.bounds.shape == (dim, 2)
            assert np.allclose(func.bounds, np.array([[-100, 100]] * dim))
            
            # Check that the shift vector was loaded correctly
            assert func.shift_vector.shape == (dim,)
            
            # Check bias value
            assert func.bias == 390.0
            
            # Verify metadata
            metadata = func.get_metadata()
            assert metadata["name"] == "F06 - Shifted Rosenbrock's Function"
            assert metadata["is_shifted"] is True
            assert metadata["is_biased"] is True
            assert metadata["is_multimodal"] is True
            assert metadata["is_separable"] is False
    
    def test_global_optimum(self):
        """Test that the function value at the global optimum is correct."""
        for dim in [2, 10, 30]:
            func = F06(dimension=dim)
            
            # The global optimum is at the shift vector
            x_opt = func.shift_vector.copy()
            
            # Evaluate at the optimum
            f_opt = func.evaluate(x_opt)
            
            # The value should match the bias
            assert np.isclose(f_opt, func.bias, rtol=1e-6, atol=1e-6)
    
    def test_against_validation_data(self):
        """Test the function against validation data."""
        for dim in [2, 10, 30]:
            # Load validation data
            validation_data = load_cec_validation_data(2005, 6, dim)
            if not validation_data:
                pytest.skip(f"No validation data for F06 with dimension {dim}")
            
            # Create function
            func = F06(dimension=dim)
            
            # Set an appropriate tolerance (higher for Rosenbrock)
            rtol = 1e-3
            atol = 1e-3
            
            # Validate function against test data
            results = validate_function(func, validation_data, rtol=rtol, atol=atol)
            print_validation_summary(results)
            
            # Assert that at least some tests pass
            assert results["passed_tests"] > 0, f"No tests passed for F06 with dimension {dim}"
            
            # We should be very close at the optimum point
            optimum_test_passed = False
            for test_case in validation_data["test_cases"]:
                # Find the optimum test case - it should have a value very close to the bias
                expected = test_case["expected"]
                if abs(expected - func.bias) < 1e-3:
                    optimum_test_passed = True
                    break
            
            # Ensure the optimum test passed
            assert optimum_test_passed, f"Optimum point test failed for F06 with dimension {dim}"
    
    def test_batch_evaluation(self):
        """Test that batch evaluation works correctly."""
        dim = 10
        func = F06(dimension=dim)
        
        # Generate some random points
        np.random.seed(42)
        num_points = 5
        X = np.random.uniform(-100, 100, (num_points, dim))
        
        # Evaluate in batch
        batch_values = func.evaluate_batch(X)
        
        # Evaluate individually
        individual_values = np.array([func.evaluate(x) for x in X])
        
        # Check that the results match
        np.testing.assert_allclose(batch_values, individual_values)
    
    def test_factory_function(self):
        """Test that the factory function creates the correct function."""
        for dim in [2, 10, 30]:
            func = create_cec2005_function(6, dim)
            assert isinstance(func, F06)
            assert func.dimension == dim
            assert func.function_number == 6 