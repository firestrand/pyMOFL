"""
Performance tests for linear transformation optimizations.

This module provides TDD-driven benchmarks to ensure optimization
doesn't break correctness while improving performance.

Following TDD principles:
- Comprehensive tests before optimization
- Baseline performance measurement
- Regression testing during optimization
"""

import time

import numpy as np
import pytest

from pyMOFL.functions.transformations.linear import (
    LinearTransform,
    linear_transform,
    linear_transform_batch,
    linear_transform_optimized,
)
from tests.utils.exact_transforms import linear_transform_exact


class TestLinearTransformPerformance:
    """Test suite for linear transformation performance optimization."""

    @pytest.fixture(params=[2, 10, 30, 50])
    def dimension(self, request):
        """Test dimensions matching CEC 2005 benchmarks."""
        return request.param

    @pytest.fixture
    def random_matrix(self, dimension):
        """Generate random transformation matrix."""
        np.random.seed(42)  # Reproducible tests
        return np.random.randn(dimension, dimension).astype(np.float64)

    @pytest.fixture
    def cec_matrix(self, dimension):
        """Load actual CEC matrix if available, otherwise random."""
        try:
            # Try to load actual CEC matrix
            matrix_file = f"src/pyMOFL/constants/cec/2005/f03/matrix_rotation_D{dimension}.txt"
            matrix = np.loadtxt(matrix_file, dtype=np.float64)
            if matrix.size == dimension * dimension:
                return matrix.reshape((dimension, dimension))
            return matrix
        except (FileNotFoundError, OSError):
            # Fall back to random matrix for performance testing
            np.random.seed(42)
            return np.random.randn(dimension, dimension).astype(np.float64)

    @pytest.fixture
    def random_vector(self, dimension):
        """Generate random input vector."""
        np.random.seed(123)  # Different seed for vector
        return np.random.randn(dimension).astype(np.float64)

    def test_exact_vs_optimized_correctness(self, random_matrix, random_vector):
        """Test that optimized version matches exact version numerically."""
        exact_result = linear_transform_exact(random_vector, random_matrix)
        optimized_result = linear_transform(random_vector, random_matrix)

        # Should be very close (within floating point precision)
        np.testing.assert_allclose(
            exact_result,
            optimized_result,
            rtol=1e-14,
            atol=1e-14,
            err_msg="Optimized version must match exact version numerically",
        )

    def test_cec_matrix_correctness(self, cec_matrix, random_vector):
        """Test correctness with actual CEC matrices."""
        exact_result = linear_transform_exact(random_vector, cec_matrix)
        optimized_result = linear_transform(random_vector, cec_matrix)

        # Should be very close for CEC matrices
        np.testing.assert_allclose(
            exact_result,
            optimized_result,
            rtol=1e-12,
            atol=1e-12,
            err_msg="CEC matrix optimization must preserve exactness",
        )

    def test_class_interface_correctness(self, random_matrix, random_vector):
        """Test LinearTransform class matches exact implementation."""
        transform = LinearTransform(random_matrix)

        exact_result = linear_transform_exact(random_vector, random_matrix)
        transform_result = transform(random_vector)

        np.testing.assert_allclose(
            exact_result,
            transform_result,
            rtol=1e-14,
            atol=1e-14,
            err_msg="LinearTransform class must match exact implementation",
        )

    def measure_performance(self, func, *args, iterations=1000):
        """Measure function performance over multiple iterations."""
        # Warm up
        for _ in range(10):
            func(*args)

        # Measure
        start_time = time.perf_counter()
        for _ in range(iterations):
            result = func(*args)
        end_time = time.perf_counter()

        return (end_time - start_time) / iterations, result

    @pytest.mark.parametrize("dimension", [2, 10, 30, 50])
    def test_performance_comparison(self, dimension):
        """Compare performance between exact and optimized versions."""
        # Generate test data
        np.random.seed(42)
        matrix = np.random.randn(dimension, dimension).astype(np.float64)
        vector = np.random.randn(dimension).astype(np.float64)

        # Measure exact version
        exact_time, exact_result = self.measure_performance(linear_transform_exact, vector, matrix)

        # Measure optimized version
        optimized_time, optimized_result = self.measure_performance(
            linear_transform, vector, matrix
        )

        # Verify correctness first
        np.testing.assert_allclose(exact_result, optimized_result, rtol=1e-14, atol=1e-14)

        # Report performance
        speedup = exact_time / optimized_time if optimized_time > 0 else float("inf")

        print(f"\nDimension {dimension}:")
        print(f"  Exact:     {exact_time * 1e6:.2f} μs")
        print(f"  Optimized: {optimized_time * 1e6:.2f} μs")
        print(f"  Speedup:   {speedup:.2f}x")

        # For higher dimensions, optimized should be faster
        if dimension >= 10:
            assert optimized_time <= exact_time, (
                f"Optimization should improve performance for D={dimension}"
            )

    @pytest.mark.parametrize("dimension", [10, 30, 50])
    def test_batch_performance_comparison(self, dimension):
        """Compare batch processing performance."""
        # Generate test data
        np.random.seed(42)
        matrix = np.random.randn(dimension, dimension).astype(np.float64)
        batch_size = 1000
        X = np.random.randn(batch_size, dimension).astype(np.float64)

        # Measure individual transformations
        start_time = time.perf_counter()
        individual_results = []
        for i in range(batch_size):
            individual_results.append(linear_transform(X[i], matrix))
        individual_time = time.perf_counter() - start_time
        individual_results = np.array(individual_results)

        # Measure batch transformation
        start_time = time.perf_counter()
        batch_results = linear_transform_batch(X, matrix)
        batch_time = time.perf_counter() - start_time

        # Verify correctness
        np.testing.assert_allclose(individual_results, batch_results, rtol=1e-14, atol=1e-14)

        # Report performance
        speedup = individual_time / batch_time if batch_time > 0 else float("inf")

        print(f"\nBatch Performance D={dimension} (N={batch_size}):")
        print(f"  Individual: {individual_time * 1000:.2f} ms")
        print(f"  Batch:      {batch_time * 1000:.2f} ms")
        print(f"  Speedup:    {speedup:.2f}x")

        # Batch should be faster for reasonable sizes
        assert batch_time < individual_time, f"Batch processing should be faster for D={dimension}"

    def test_batch_transformation_correctness(self, random_matrix, dimension):
        """Test batch transformation correctness."""
        # Generate batch of vectors
        np.random.seed(456)
        batch_size = 100
        X = np.random.randn(batch_size, dimension).astype(np.float64)

        # Transform each vector individually with exact method
        individual_results = []
        for i in range(batch_size):
            individual_results.append(linear_transform_exact(X[i], random_matrix))
        individual_results = np.array(individual_results)

        # Transform batch with optimized method (loop version)
        batch_results_loop = np.empty_like(X)
        for i in range(batch_size):
            batch_results_loop[i] = linear_transform(X[i], random_matrix)

        # Transform batch with vectorized method
        batch_results_vectorized = linear_transform_batch(X, random_matrix)

        # All should match
        np.testing.assert_allclose(
            individual_results,
            batch_results_loop,
            rtol=1e-14,
            atol=1e-14,
            err_msg="Batch loop transformation must match individual transformations",
        )

        np.testing.assert_allclose(
            individual_results,
            batch_results_vectorized,
            rtol=1e-14,
            atol=1e-14,
            err_msg="Batch vectorized transformation must match individual transformations",
        )

    def test_linear_transform_class_batch(self, random_matrix, dimension):
        """Test LinearTransform class batch functionality."""
        # Generate batch of vectors
        np.random.seed(789)
        batch_size = 50
        X = np.random.randn(batch_size, dimension).astype(np.float64)

        # Test LinearTransform class
        transform = LinearTransform(random_matrix)

        # Transform individually
        individual = np.array([transform(X[i]) for i in range(batch_size)])

        # Transform as batch
        batch = transform.transform_batch(X)

        # Should match
        np.testing.assert_allclose(
            individual,
            batch,
            rtol=1e-14,
            atol=1e-14,
            err_msg="LinearTransform batch must match individual",
        )

        # Also verify against exact implementation
        from tests.utils.exact_transforms import linear_transform_exact_batch

        exact_batch = linear_transform_exact_batch(X, random_matrix)

        np.testing.assert_allclose(
            batch,
            exact_batch,
            rtol=1e-14,
            atol=1e-14,
            err_msg="LinearTransform batch must match exact implementation",
        )

    def test_edge_cases(self):
        """Test edge cases that optimization must handle."""
        # Identity matrix
        I = np.eye(3, dtype=np.float64)
        v = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        exact_result = linear_transform_exact(v, I)
        optimized_result = linear_transform_optimized(v, I)

        np.testing.assert_allclose(exact_result, v, rtol=1e-15)
        np.testing.assert_allclose(optimized_result, v, rtol=1e-15)
        np.testing.assert_allclose(exact_result, optimized_result, rtol=1e-15)

        # Zero matrix
        Z = np.zeros((3, 3), dtype=np.float64)
        exact_result = linear_transform_exact(v, Z)
        optimized_result = linear_transform_optimized(v, Z)

        expected = np.zeros(3, dtype=np.float64)
        np.testing.assert_allclose(exact_result, expected, atol=1e-15)
        np.testing.assert_allclose(optimized_result, expected, atol=1e-15)

        # Diagonal matrix (scaling)
        D = np.diag([1.0, 2.0, 0.5]).astype(np.float64)
        exact_result = linear_transform_exact(v, D)
        optimized_result = linear_transform_optimized(v, D)

        expected = np.array([1.0, 4.0, 1.5], dtype=np.float64)
        np.testing.assert_allclose(exact_result, expected, rtol=1e-15)
        np.testing.assert_allclose(optimized_result, expected, rtol=1e-15)


if __name__ == "__main__":
    # Quick performance test
    test_suite = TestLinearTransformPerformance()

    print("Linear Transformation Performance Analysis")
    print("=" * 50)

    for dim in [2, 10, 30, 50]:
        test_suite.test_performance_comparison(dim)

    print("\nAll correctness tests should pass before optimization!")
