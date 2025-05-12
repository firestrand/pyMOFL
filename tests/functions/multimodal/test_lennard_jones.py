"""
Tests for the Lennard-Jones cluster function.
"""

import pytest
import numpy as np
from pyMOFL.functions.multimodal import LennardJonesFunction
from pyMOFL.decorators import BiasedFunction


class TestLennardJonesFunction:
    """Tests for the Lennard-Jones cluster function."""
    
    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Test with defaults (6 atoms)
        func = LennardJonesFunction()
        assert func.dimension == 18
        assert func.bounds.shape == (18, 2)
        assert np.array_equal(func.bounds, np.array([[-2, 2]] * 18))
        assert func.n_atoms == 6
        assert func.global_minimum == -12.7121
        
        # Test with custom atom count
        func = LennardJonesFunction(n_atoms=4)
        assert func.dimension == 12
        assert func.bounds.shape == (12, 2)
        assert func.n_atoms == 4
        assert func.global_minimum == -6.0
        
        # Test with custom bounds
        custom_bounds = np.array([[-1, 1]] * 18)
        func = LennardJonesFunction(bounds=custom_bounds)
        assert func.bounds.shape == (18, 2)
        assert np.array_equal(func.bounds, custom_bounds)
    
    def test_evaluate_octahedral(self):
        """Test the energy of an octahedral configuration."""
        func = LennardJonesFunction()
        
        # Octahedral structure at equilibrium distance
        r_eq = 2.0**(1.0/6.0)  # Equilibrium distance in LJ units
        coords = np.array([
            0.0, 0.0, 0.0,      # Central atom 
            1.0, 0.0, 0.0,      # Surrounding atoms in octahedral arrangement
            -1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, 0.0, 1.0
        ]) * r_eq
        
        # Calculate the energy with our implementation
        energy = func.evaluate(coords)
        
        # For this configuration, the expected energy is approximately -6.9370
        expected_energy = -6.9370
        assert np.isclose(energy, expected_energy, rtol=1e-4)
        
        # Test with bias decorator
        bias_value = 10.0
        biased_func = BiasedFunction(func, bias=bias_value)
        energy_with_bias = biased_func.evaluate(coords)
        assert np.isclose(energy_with_bias, energy + bias_value, rtol=1e-4)
    
    def test_evaluate_two_atoms(self):
        """Test the simplest case: two atoms."""
        func = LennardJonesFunction(n_atoms=2)
        
        # Two atoms at equilibrium distance along x-axis
        r_eq = 2.0**(1.0/6.0)
        coords = np.array([0.0, 0.0, 0.0, r_eq, 0.0, 0.0])
        
        # At equilibrium distance, energy should be -1.0
        energy = func.evaluate(coords)
        assert np.isclose(energy, -1.0, atol=1e-5)
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        func = LennardJonesFunction()
        
        # Create the octahedral structure with proper scaling
        r_eq = 2.0**(1.0/6.0)  # Equilibrium distance in LJ units
        coords = np.array([
            0.0, 0.0, 0.0,      # Central atom 
            1.0, 0.0, 0.0,      # Surrounding atoms in octahedral arrangement
            -1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, 0.0, 1.0
        ]) * r_eq
        
        # Create a batch with the octahedral configuration and a perturbed version
        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)
        perturbed = coords + 0.1 * rng.standard_normal(18)
        batch = np.vstack([coords, perturbed])
        
        # Test batch evaluation
        energies = func.evaluate_batch(batch)
        assert energies.shape == (2,)
        
        # Verify individual evaluations
        expected = np.array([
            func.evaluate(batch[0]),
            func.evaluate(batch[1])
        ])
        np.testing.assert_allclose(energies, expected)
        
        # The original configuration should have lower energy than the perturbed one
        assert energies[0] < energies[1]
        
        # Test with bias decorator
        bias_value = 3.0
        biased_func = BiasedFunction(func, bias=bias_value)
        biased_energies = biased_func.evaluate_batch(batch)
        np.testing.assert_allclose(biased_energies, energies + bias_value)
    
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
            coords = rng.uniform(-1, 1, size=3*n_atoms)
            
            # Energy of a random configuration should be higher than the global minimum
            energy = func.evaluate(coords)
            assert energy > expected_minimum
    
    def _load_coordinates(self, xyz_file):
        """Helper to load coordinates from an XYZ file."""
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
        
        coords = []
        # Skip the first two lines (atom count and comment) and parse the rest
        for line in lines[2:]:
            parts = line.split()
            coords.extend([float(parts[1]), float(parts[2]), float(parts[3])])
        
        return np.array(coords) 