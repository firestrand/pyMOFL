"""
Tests for the Lennard-Jones cluster function.
"""

from pathlib import Path

import numpy as np
import pytest

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.functions import LennardJonesCECFunction, LennardJonesFunction
from tests.utils.benchmark_validation import BenchmarkValidator


class TestLennardJonesFunction:
    """Tests for the Lennard-Jones cluster function."""

    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Test with defaults (6 atoms)
        func = LennardJonesFunction()
        assert func.dimension == 18
        np.testing.assert_allclose(func.initialization_bounds.low, [-2] * 18)
        np.testing.assert_allclose(func.initialization_bounds.high, [2] * 18)
        np.testing.assert_allclose(func.operational_bounds.low, [-2] * 18)
        np.testing.assert_allclose(func.operational_bounds.high, [2] * 18)
        assert func.n_atoms == 6
        assert func.global_minimum == -12.7121

        # Test with custom atom count
        func = LennardJonesFunction(n_atoms=4)
        assert func.dimension == 12
        np.testing.assert_allclose(func.initialization_bounds.low, [-2] * 12)
        np.testing.assert_allclose(func.initialization_bounds.high, [2] * 12)
        np.testing.assert_allclose(func.operational_bounds.low, [-2] * 12)
        np.testing.assert_allclose(func.operational_bounds.high, [2] * 12)
        assert func.n_atoms == 4
        assert func.global_minimum == -6.0

        # Test with custom bounds
        custom_init_bounds = Bounds(
            low=np.array([-1] * 18), high=np.array([1] * 18), mode=BoundModeEnum.INITIALIZATION
        )
        custom_oper_bounds = Bounds(
            low=np.array([-1] * 18), high=np.array([1] * 18), mode=BoundModeEnum.OPERATIONAL
        )
        func = LennardJonesFunction(
            initialization_bounds=custom_init_bounds, operational_bounds=custom_oper_bounds
        )
        np.testing.assert_allclose(func.initialization_bounds.low, [-1] * 18)
        np.testing.assert_allclose(func.initialization_bounds.high, [1] * 18)
        np.testing.assert_allclose(func.operational_bounds.low, [-1] * 18)
        np.testing.assert_allclose(func.operational_bounds.high, [1] * 18)

    def test_constructor_dimension_validation(self):
        """Test strict dimension and n_atoms validation."""
        # Non-multiples of 3 should be rejected (not silently shortened)
        with pytest.raises(ValueError, match="multiple of 3"):
            LennardJonesFunction(dimension=10)
        with pytest.raises(ValueError, match="multiple of 3"):
            LennardJonesFunction(dimension=5)

        # Mismatch between dimension and n_atoms should be rejected
        with pytest.raises(ValueError, match="dimension must equal 3 \\* n_atoms"):
            LennardJonesFunction(dimension=12, n_atoms=5)
        with pytest.raises(ValueError, match="dimension must equal 3 \\* n_atoms"):
            LennardJonesFunction(dimension=9, n_atoms=4)

        # n_atoms < 2 should be rejected
        with pytest.raises(ValueError, match="n_atoms must be >= 2"):
            LennardJonesFunction(n_atoms=1)

        # Consistent dimension and n_atoms should succeed
        func = LennardJonesFunction(dimension=12, n_atoms=4)
        assert func.dimension == 12
        assert func.n_atoms == 4

    def test_evaluate_octahedral(self):
        """Test the energy of an octahedral configuration."""
        func = LennardJonesFunction()

        # Octahedral structure at equilibrium distance
        r_eq = 2.0 ** (1.0 / 6.0)  # Equilibrium distance in LJ units
        coords = (
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,  # Central atom
                    1.0,
                    0.0,
                    0.0,  # Surrounding atoms in octahedral arrangement
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            )
            * r_eq
        )

        # Calculate the energy with our implementation
        energy = func.evaluate(coords)

        # For this configuration, the expected energy is approximately -6.9370
        expected_energy = -6.9370
        assert np.isclose(energy, expected_energy, rtol=1e-4)

        # Test with bias decorator

    # #         biased_func = BiasWrapper(inner_function=func, bias=bias_value)
    # energy_with_bias = biased_func.evaluate(coords)
    # assert np.isclose(energy_with_bias, energy + bias_value, rtol=1e-4)

    def test_evaluate_two_atoms(self):
        """Test the simplest case: two atoms."""
        func = LennardJonesFunction(n_atoms=2)

        # Two atoms at equilibrium distance along x-axis
        r_eq = 2.0 ** (1.0 / 6.0)
        coords = np.array([0.0, 0.0, 0.0, r_eq, 0.0, 0.0])

        # At equilibrium distance, energy should be -1.0
        energy = func.evaluate(coords)
        assert np.isclose(energy, -1.0, atol=1e-5)

    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        func = LennardJonesFunction()

        # Create the octahedral structure with proper scaling
        r_eq = 2.0 ** (1.0 / 6.0)  # Equilibrium distance in LJ units
        coords = (
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,  # Central atom
                    1.0,
                    0.0,
                    0.0,  # Surrounding atoms in octahedral arrangement
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            )
            * r_eq
        )

        # Create a batch with the octahedral configuration and a perturbed version
        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        perturbed = coords + 0.1 * rng.standard_normal(18)
        batch = np.vstack([coords, perturbed])

        # Test batch evaluation
        energies = func.evaluate_batch(batch)
        assert energies.shape == (2,)

        # Verify individual evaluations
        expected = np.array([func.evaluate(batch[0]), func.evaluate(batch[1])])
        np.testing.assert_allclose(energies, expected)

        # The original configuration should have lower energy than the perturbed one
        assert energies[0] < energies[1]

        # Test with bias decorator

    # #         biased_func = BiasWrapper(inner_function=func, bias=bias_value)
    # biased_energies = biased_func.evaluate_batch(batch)
    # np.testing.assert_allclose(biased_energies, energies + bias_value)

    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        func = LennardJonesFunction()

        # Test with incorrect dimension
        with pytest.raises(ValueError):
            func.evaluate(np.array([1, 2, 3]))

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_physical_minimum(self):
        """Test that random points never have energy below the physical minimum."""
        func = LennardJonesFunction()

        # Generate random points in the domain
        rng = np.random.default_rng(42)
        points = rng.uniform(-2, 2, size=(100, 18))

        # Evaluate function at these points
        values = func.evaluate_batch(points)

        # Check that no values are below the global minimum
        # We use a slightly lower bound to account for potential numerical errors
        assert (values > -13.0).all()

    def test_multiple_atom_counts(self):
        """Test the function with different numbers of atoms."""
        # Test with a few different atom counts
        for n_atoms in [2, 3, 4, 5, 6]:
            func = LennardJonesFunction(n_atoms=n_atoms)
            expected_minimum = LennardJonesFunction.LJ_GLOBAL_MINIMA[n_atoms]

            # Generate a random configuration (not at the minimum)
            rng = np.random.default_rng(42)
            coords = rng.uniform(-1, 1, size=3 * n_atoms)

            # Energy of a random configuration should be higher than the global minimum
            energy = func.evaluate(coords)
            assert energy > expected_minimum

    def _load_coordinates(self, xyz_file):
        """Helper to load coordinates from an XYZ file."""
        with Path(xyz_file).open() as f:
            lines = f.readlines()

        coords = []
        # Skip the first two lines (atom count and comment) and parse the rest
        for line in lines[2:]:
            parts = line.split()
            coords.extend([float(parts[1]), float(parts[2]), float(parts[3])])

        return np.array(coords)


class TestLennardJonesCECFunction:
    """Tests for the CEC 2019 Lennard-Jones cluster function variant."""

    def test_initialization(self):
        """Test default initialization (18D, [-4, 4])."""
        func = LennardJonesCECFunction()
        assert func.dimension == 18
        np.testing.assert_allclose(func.initialization_bounds.low, [-4.0] * 18)
        np.testing.assert_allclose(func.initialization_bounds.high, [4.0] * 18)
        np.testing.assert_allclose(func.operational_bounds.low, [-4.0] * 18)
        np.testing.assert_allclose(func.operational_bounds.high, [4.0] * 18)

    def test_initialization_invalid_dimension(self):
        """Test invalid dimension validation and k outside CEC_MINIMA."""
        with pytest.raises(ValueError):
            LennardJonesCECFunction(dimension=5)
        with pytest.raises(ValueError):
            LennardJonesCECFunction(dimension=16)
        # k outside CEC_MINIMA range [2, 25] (dim 6 to 75)
        with pytest.raises(ValueError, match="multiple of 3 >= 6"):
            LennardJonesCECFunction(dimension=3)  # k = 1
        with pytest.raises(ValueError, match="only supports k in"):
            LennardJonesCECFunction(dimension=78)  # k = 26
        # Mismatch between dimension and n_atoms
        with pytest.raises(ValueError, match="dimension must equal 3 \\* n_atoms"):
            LennardJonesCECFunction(dimension=12, n_atoms=5)

    def test_benchmark_contract(self):
        """Test compliance with BenchmarkValidator contract and global minimum behavior."""
        # For k=2 (6D), exact stored minimizer exists and evaluates to 0.0
        func2 = LennardJonesCECFunction(dimension=6)
        pt2, val2 = func2.get_global_minimum()
        assert val2 == 0.0
        assert abs(func2.evaluate(pt2)) < 1e-12
        BenchmarkValidator.assert_contract(func2, check_global_minimum=True)

        # For k=6 (18D), analytical coordinates are not stored: raises NotImplementedError
        func18 = LennardJonesCECFunction(dimension=18)
        with pytest.raises(NotImplementedError):
            func18.get_global_minimum()
        # BenchmarkValidator.assert_contract cleanly skips NotImplementedError
        BenchmarkValidator.assert_contract(func18, check_global_minimum=True)

    def test_evaluate_batch_matches_single(self):
        """Test batch evaluation matches individual evaluations."""
        func = LennardJonesCECFunction(dimension=18)
        rng = np.random.default_rng(42)
        X = rng.uniform(-2.0, 2.0, size=(10, 18))
        batch_results = func.evaluate_batch(X)
        single_results = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch_results, single_results, rtol=1e-12, atol=1e-12)

    def test_registry(self):
        """Test registry retrieval for LennardJonesCEC."""
        from pyMOFL.registry import get

        func1 = get("LennardJonesCEC")(dimension=18)
        func2 = get("lennard_jones_cec")(dimension=18)
        assert isinstance(func1, LennardJonesCECFunction)
        assert isinstance(func2, LennardJonesCECFunction)
        x = np.ones(18)
        assert func1.evaluate(x) == func2.evaluate(x)
