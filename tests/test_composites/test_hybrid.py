"""
Tests for the HybridFunction class.
"""

import pytest
import numpy as np
from pyMOFL.functions.unimodal import SphereFunction
from pyMOFL.functions.multimodal import RastriginFunction
from pyMOFL.composites import HybridFunction


class TestHybridFunction:
    """Tests for the HybridFunction class."""
    
    def test_initialization(self):
        """Test initialization with default and custom bounds."""
        # Create component functions
        sphere = SphereFunction(dimension=2)
        rastrigin = RastriginFunction(dimension=2)
        
        # Create a hybrid function with default bounds
        components = [sphere, rastrigin]
        partitions = [(0, 2), (2, 4)]
        hybrid = HybridFunction(components, partitions)
        
        assert hybrid.dimension == 4
        assert len(hybrid.components) == 2
        assert hybrid.partitions == [(0, 2), (2, 4)]
        assert np.array_equal(hybrid.weights, np.array([0.5, 0.5]))
        
        # Create a hybrid function with custom bounds
        custom_bounds = np.array([[-10, 10], [-5, 5], [-3, 3], [-1, 1]])
        hybrid = HybridFunction(components, partitions, bounds=custom_bounds)
        
        assert hybrid.dimension == 4
        assert np.array_equal(hybrid.bounds, custom_bounds)
        
        # Create a hybrid function with custom weights
        weights = [0.3, 0.7]
        hybrid = HybridFunction(components, partitions, weights=weights)
        
        assert hybrid.dimension == 4
        assert np.array_equal(hybrid.weights, np.array([0.3, 0.7]))
    
    def test_parameter_validation(self):
        """Test that parameters are validated correctly."""
        # Create component functions
        sphere = SphereFunction(dimension=2)
        rastrigin = RastriginFunction(dimension=2)
        
        # Test with mismatched component and partition counts
        with pytest.raises(ValueError):
            HybridFunction([sphere, rastrigin], [(0, 2)])
        
        # Test with invalid partition
        with pytest.raises(ValueError):
            HybridFunction([sphere, rastrigin], [(0, 2), (1, 0)])
        
        # Test with mismatched weight count
        with pytest.raises(ValueError):
            HybridFunction([sphere, rastrigin], [(0, 2), (2, 4)], weights=[0.5])
        
        # Test with invalid bounds shape
        custom_bounds = np.array([[-10, 10], [-5, 5]])
        with pytest.raises(ValueError):
            HybridFunction([sphere, rastrigin], [(0, 2), (2, 4)], bounds=custom_bounds)
    
    def test_evaluate_simple(self):
        """Test the evaluate method with simple components."""
        # Create component functions
        sphere = SphereFunction(dimension=2)
        
        # Create a hybrid function with a single component
        components = [sphere]
        partitions = [(0, 2)]
        hybrid = HybridFunction(components, partitions)
        
        # Test at global minimum
        assert hybrid.evaluate(np.array([0.0, 0.0])) == 0.0
        
        # Test with arbitrary vector
        x = np.array([2.0, 3.0])
        assert hybrid.evaluate(x) == 13.0  # 2^2 + 3^2 = 13
    
    def test_evaluate_multiple_components(self):
        """Test the evaluate method with multiple components."""
        # Create component functions
        sphere = SphereFunction(dimension=2)
        rastrigin = RastriginFunction(dimension=2)
        
        # Create a hybrid function with multiple components
        components = [sphere, rastrigin]
        partitions = [(0, 2), (2, 4)]
        weights = [0.5, 0.5]
        hybrid = HybridFunction(components, partitions, weights=weights)
        
        # Test at global minimum
        # Both components have their minimum at [0, 0], so the hybrid should have its minimum at [0, 0, 0, 0]
        assert hybrid.evaluate(np.array([0.0, 0.0, 0.0, 0.0])) == 0.0
        
        # Test with arbitrary vector
        x = np.array([1.0, 1.0, 1.0, 1.0])
        # sphere(1,1) = 1^2 + 1^2 = 2
        # rastrigin(1,1) = 10*2 + (1^2 - 10*cos(2*pi*1) + 1^2 - 10*cos(2*pi*1))
        # The result should be a weighted sum: 0.5*2 + 0.5*rastrigin(1,1)
        expected = 0.5 * 2.0 + 0.5 * (20.0 + 2.0 - 10.0 * np.cos(2 * np.pi * 1.0) - 10.0 * np.cos(2 * np.pi * 1.0))
        np.testing.assert_allclose(hybrid.evaluate(x), expected)
    
    def test_evaluate_with_dimension_mismatch(self):
        """Test the evaluate method with dimension mismatches."""
        # Create component functions
        sphere = SphereFunction(dimension=2)
        rastrigin = RastriginFunction(dimension=3)
        
        # Create a hybrid function with components of different dimensions
        components = [sphere, rastrigin]
        partitions = [(0, 2), (2, 4)]  # Note: rastrigin expects 3 dimensions but we only give it 2
        hybrid = HybridFunction(components, partitions)
        
        # Test with a vector that has the correct total dimension
        x = np.array([0.0, 0.0, 0.0, 0.0])
        # The second component will be padded with zeros to match its expected dimension
        assert hybrid.evaluate(x) == 0.0
        
        # Test with a vector that has extra dimensions for the second component
        x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        # The second component will use only the first 3 dimensions it needs
        with pytest.raises(ValueError):
            hybrid.evaluate(x)
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        # Create component functions
        sphere = SphereFunction(dimension=2)
        
        # Create a hybrid function with a single component
        components = [sphere]
        partitions = [(0, 2)]
        hybrid = HybridFunction(components, partitions)
        
        # Test with batch of vectors
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
        expected = np.array([0.0, 2.0, 13.0])
        np.testing.assert_allclose(hybrid.evaluate_batch(X), expected)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        # Create component functions
        sphere = SphereFunction(dimension=2)
        rastrigin = RastriginFunction(dimension=2)
        
        # Create a hybrid function
        components = [sphere, rastrigin]
        partitions = [(0, 2), (2, 4)]
        hybrid = HybridFunction(components, partitions)
        
        # Test with incorrect dimension
        with pytest.raises(ValueError):
            hybrid.evaluate(np.array([1.0, 2.0]))
        
        with pytest.raises(ValueError):
            hybrid.evaluate_batch(np.array([[1.0, 2.0], [3.0, 4.0]])) 