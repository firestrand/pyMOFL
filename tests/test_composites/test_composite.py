"""
Tests for the CompositeFunction class.
"""

import pytest
import numpy as np
from pyMOFL.functions.unimodal import SphereFunction
from pyMOFL.functions.multimodal import RastriginFunction
from pyMOFL.decorators import ShiftedFunction
from pyMOFL.composites import CompositeFunction


class TestCompositeFunction:
    """Tests for the CompositeFunction class."""
    
    def test_initialization(self):
        """Test initialization with default and custom bounds."""
        # Create component functions
        sphere = SphereFunction(dimension=2)
        rastrigin = RastriginFunction(dimension=2)
        
        # Create a composite function with default bounds
        components = [sphere, rastrigin]
        sigmas = [1.0, 2.0]
        lambdas = [1.0, 1.0]
        biases = [0.0, 100.0]
        composite = CompositeFunction(components, sigmas, lambdas, biases)
        
        assert composite.dimension == 2
        assert np.array_equal(composite.bounds, sphere.bounds)
        assert len(composite.components) == 2
        assert np.array_equal(composite.sigmas, np.array([1.0, 2.0]))
        assert np.array_equal(composite.lambdas, np.array([1.0, 1.0]))
        assert np.array_equal(composite.biases, np.array([0.0, 100.0]))
        
        # Create a composite function with custom bounds
        custom_bounds = np.array([[-10, 10], [-5, 5]])
        composite = CompositeFunction(components, sigmas, lambdas, biases, bounds=custom_bounds)
        
        assert composite.dimension == 2
        assert np.array_equal(composite.bounds, custom_bounds)
    
    def test_parameter_validation(self):
        """Test that parameters are validated correctly."""
        # Create component functions
        sphere = SphereFunction(dimension=2)
        rastrigin = RastriginFunction(dimension=2)
        
        # Test with mismatched component dimensions
        sphere3d = SphereFunction(dimension=3)
        with pytest.raises(ValueError):
            CompositeFunction([sphere, sphere3d], [1.0, 2.0], [1.0, 1.0], [0.0, 100.0])
        
        # Test with mismatched parameter lengths
        with pytest.raises(ValueError):
            CompositeFunction([sphere, rastrigin], [1.0], [1.0, 1.0], [0.0, 100.0])
        
        with pytest.raises(ValueError):
            CompositeFunction([sphere, rastrigin], [1.0, 2.0], [1.0], [0.0, 100.0])
        
        with pytest.raises(ValueError):
            CompositeFunction([sphere, rastrigin], [1.0, 2.0], [1.0, 1.0], [0.0])
    
    def test_evaluate_simple(self):
        """Test the evaluate method with simple components."""
        # Create component functions
        sphere = SphereFunction(dimension=2)
        
        # Create a composite function with a single component
        components = [sphere]
        sigmas = [1.0]
        lambdas = [1.0]
        biases = [0.0]
        composite = CompositeFunction(components, sigmas, lambdas, biases)
        
        # Test at global minimum
        assert composite.evaluate(np.array([0.0, 0.0])) == 0.0
        
        # Test with arbitrary vector
        x = np.array([2.0, 3.0])
        assert composite.evaluate(x) == 13.0  # 2^2 + 3^2 = 13
    
    def test_evaluate_multiple_components(self):
        """Test the evaluate method with multiple components."""
        # Create component functions
        sphere = SphereFunction(dimension=2)
        rastrigin = RastriginFunction(dimension=2)
        
        # Create a composite function with multiple components
        components = [sphere, rastrigin]
        sigmas = [1.0, 2.0]
        lambdas = [1.0, 1.0]
        biases = [0.0, 100.0]
        composite = CompositeFunction(components, sigmas, lambdas, biases)
        
        # Test at global minimum
        # Both components have their minimum at [0, 0], so the composite should too
        # The value should be a weighted sum of the biases: w1*0 + w2*100
        # Since both components have their minimum at the same point, the weights
        # should be normalized to [0.5, 0.5]
        assert composite.evaluate(np.array([0.0, 0.0])) == 50.0
        
        # Test with arbitrary vector
        x = np.array([1.0, 1.0])
        # sphere(1,1) = 1^2 + 1^2 = 2
        # rastrigin(1,1) = 10*2 + (1^2 - 10*cos(2*pi*1) + 1^2 - 10*cos(2*pi*1))
        # The weights will depend on the distance to the optimum
        # We'll just check that the result is reasonable
        result = composite.evaluate(x)
        assert result > 0.0
    
    def test_evaluate_with_shifted_components(self):
        """Test the evaluate method with shifted components."""
        # Create component functions
        sphere = SphereFunction(dimension=2)
        shifted_sphere = ShiftedFunction(sphere, np.array([1.0, 2.0]))
        
        # Create a composite function with shifted components
        components = [sphere, shifted_sphere]
        sigmas = [1.0, 2.0]
        lambdas = [1.0, 1.0]
        biases = [0.0, 100.0]
        composite = CompositeFunction(components, sigmas, lambdas, biases)
        
        # Test at the global minimum of the first component
        # The value should be a weighted sum: w1*0 + w2*(shifted_sphere([0,0]) + 100)
        # The weights will depend on the distance to each component's optimum
        result = composite.evaluate(np.array([0.0, 0.0]))
        assert result > 0.0
        
        # Test at the global minimum of the second component
        # The value should be a weighted sum: w1*(sphere([1,2])) + w2*100
        # The weights will depend on the distance to each component's optimum
        result = composite.evaluate(np.array([1.0, 2.0]))
        assert result > 0.0
    
    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        # Create component functions
        sphere = SphereFunction(dimension=2)
        
        # Create a composite function with a single component
        components = [sphere]
        sigmas = [1.0]
        lambdas = [1.0]
        biases = [0.0]
        composite = CompositeFunction(components, sigmas, lambdas, biases)
        
        # Test with batch of vectors
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
        expected = np.array([0.0, 2.0, 13.0])
        np.testing.assert_allclose(composite.evaluate_batch(X), expected)
    
    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        # Create component functions
        sphere = SphereFunction(dimension=2)
        
        # Create a composite function
        components = [sphere]
        sigmas = [1.0]
        lambdas = [1.0]
        biases = [0.0]
        composite = CompositeFunction(components, sigmas, lambdas, biases)
        
        # Test with incorrect dimension
        with pytest.raises(ValueError):
            composite.evaluate(np.array([1.0, 2.0, 3.0]))
        
        with pytest.raises(ValueError):
            composite.evaluate_batch(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])) 