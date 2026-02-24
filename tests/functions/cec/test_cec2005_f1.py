"""
Test CEC 2005 F1 - Shifted Sphere Function.

Following TDD principles, we write tests first to drive the implementation.
"""

from pathlib import Path

import numpy as np


class TestCEC2005F1:
    """Test CEC 2005 F1 implementation."""

    def test_f1_at_optimum(self):
        """Test F1 at its optimum point."""
        # Load the shift vector for dimension 10
        shift_path = Path("src/pyMOFL/constants/cec/2005/f01/vector_shift_D50.txt")
        shift_data = np.loadtxt(shift_path)
        optimum = shift_data[:10]  # First 10 dimensions

        # Create F1 function
        from pyMOFL.factories import BenchmarkFactory

        factory = BenchmarkFactory(data_path="src/pyMOFL/constants/cec/2005")

        # F1 configuration (corrected order based on C code)
        config = {
            "type": "bias",
            "parameters": {"value": -450},
            "function": {
                "type": "sphere",
                "parameters": {},
                "function": {"type": "shift", "parameters": {"vector": "f01/vector_shift_D50.txt"}},
            },
        }

        f1 = factory.create_function(config, dimension=10)

        # Evaluate at optimum - should be -450 (bias value)
        result = f1.evaluate(optimum)
        assert abs(result - (-450.0)) < 1e-10, f"Expected -450 at optimum, got {result}"

    def test_f1_at_origin(self):
        """Test F1 at origin."""
        from pyMOFL.factories import BenchmarkFactory

        factory = BenchmarkFactory(data_path="src/pyMOFL/constants/cec/2005")

        config = {
            "type": "bias",
            "parameters": {"value": -450},
            "function": {
                "type": "sphere",
                "parameters": {},
                "function": {"type": "shift", "parameters": {"vector": "f01/vector_shift_D50.txt"}},
            },
        }

        f1 = factory.create_function(config, dimension=10)

        # At origin, we should get sphere(0 - shift) + bias = ||shift||^2 - 450
        origin = np.zeros(10)
        result = f1.evaluate(origin)

        # Load shift and compute expected value
        shift_path = Path("src/pyMOFL/constants/cec/2005/f01/vector_shift_D50.txt")
        shift_data = np.loadtxt(shift_path)
        shift = shift_data[:10]
        expected = np.sum(shift**2) - 450.0

        assert abs(result - expected) < 1e-10, f"Expected {expected} at origin, got {result}"

    def test_f1_composition_order(self):
        """Test that the composition order is correct."""
        from pyMOFL.functions.benchmark.sphere import SphereFunction
        from pyMOFL.functions.transformations import BiasTransform, ComposedFunction, ShiftTransform

        # Create components
        shift = np.array([1.0, 2.0, 3.0])
        sphere = SphereFunction(dimension=3)

        # Create composed function: bias(sphere(shift(x)))
        composed = ComposedFunction(
            base_function=sphere,
            input_transforms=[ShiftTransform(shift)],
            output_transforms=[BiasTransform(-450)],
        )

        # Test at origin
        x = np.zeros(3)
        result = composed.evaluate(x)

        # Expected: sphere([0,0,0] - [1,2,3]) + (-450) = sphere([-1,-2,-3]) - 450 = 14 - 450 = -436
        expected = (1**2 + 2**2 + 3**2) - 450
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"

    def test_f1_batch_evaluation(self):
        """Test batch evaluation of F1."""
        from pyMOFL.factories import BenchmarkFactory

        factory = BenchmarkFactory(data_path="src/pyMOFL/constants/cec/2005")

        config = {
            "type": "bias",
            "parameters": {"value": -450},
            "function": {
                "type": "sphere",
                "parameters": {},
                "function": {
                    "type": "shift",
                    "parameters": {
                        "vector": np.ones(5) * 2.0  # Simple shift for testing
                    },
                },
            },
        }

        f1 = factory.create_function(config, dimension=5)

        # Create batch of test points
        X = np.array(
            [
                np.zeros(5),  # Origin
                np.ones(5) * 2.0,  # Optimum (at shift)
                np.ones(5),  # Another point
            ]
        )

        results = f1.evaluate_batch(X)

        # Check each result
        assert abs(results[0] - (5 * 4.0 - 450)) < 1e-10  # sphere([0,0,0,0,0] - [2,2,2,2,2]) - 450
        assert abs(results[1] - (-450)) < 1e-10  # At optimum
        assert abs(results[2] - (5 * 1.0 - 450)) < 1e-10  # sphere([1,1,1,1,1] - [2,2,2,2,2]) - 450
