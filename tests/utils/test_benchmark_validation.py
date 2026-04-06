"""
Smoke tests for the BenchmarkValidator utility.

Exercises the validator against existing benchmark functions to verify
the utility itself works correctly.
"""

from pyMOFL.functions.benchmark.ackley import AckleyFunction
from pyMOFL.functions.benchmark.sphere import SphereFunction
from tests.utils.benchmark_validation import BenchmarkValidator


class TestBenchmarkValidatorWithSphere:
    """Verify BenchmarkValidator works against SphereFunction (simplest case)."""

    def test_full_contract_dim2(self):
        func = SphereFunction(dimension=2)
        BenchmarkValidator.assert_contract(func)

    def test_full_contract_dim10(self):
        func = SphereFunction(dimension=10)
        BenchmarkValidator.assert_contract(func)

    def test_multiple_dimensions(self):
        BenchmarkValidator.assert_contract_multiple_dimensions(
            SphereFunction, dimensions=[2, 5, 10, 30]
        )


class TestBenchmarkValidatorWithAckley:
    """Verify BenchmarkValidator works against AckleyFunction (multimodal)."""

    def test_full_contract_dim2(self):
        func = AckleyFunction(dimension=2)
        BenchmarkValidator.assert_contract(func)

    def test_multiple_dimensions(self):
        BenchmarkValidator.assert_contract_multiple_dimensions(
            AckleyFunction, dimensions=[2, 10, 30]
        )


class TestBenchmarkValidatorIndividualAssertions:
    """Test individual assertion methods for targeted usage."""

    def test_evaluate_returns_float(self):
        func = SphereFunction(dimension=3)
        BenchmarkValidator.assert_evaluate_returns_float(func)

    def test_evaluate_batch_shape(self):
        func = SphereFunction(dimension=3)
        BenchmarkValidator.assert_evaluate_batch_shape(func, batch_size=7)

    def test_evaluate_batch_consistent(self):
        func = SphereFunction(dimension=3)
        BenchmarkValidator.assert_evaluate_batch_consistent(func, batch_size=7)

    def test_callable_consistent(self):
        func = SphereFunction(dimension=3)
        BenchmarkValidator.assert_callable_consistent(func)

    def test_global_minimum(self):
        func = SphereFunction(dimension=5)
        BenchmarkValidator.assert_global_minimum(func)

    def test_bounds_set(self):
        func = SphereFunction(dimension=3)
        BenchmarkValidator.assert_bounds_set(func)

    def test_dimension_validation(self):
        func = SphereFunction(dimension=3)
        BenchmarkValidator.assert_dimension_validation(func)
