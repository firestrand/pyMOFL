"""
Tests for the CompositeFunction class.
"""

import pytest
import numpy as np
from pyMOFL.functions.unimodal import SphereFunction
from pyMOFL.functions.multimodal import RastriginFunction
from pyMOFL.decorators import Shifted
from pyMOFL.composites import CompositeFunction
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.decorators import Biased
from pyMOFL.decorators import Quantized


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
        assert np.array_equal(composite.initialization_bounds.low, sphere.initialization_bounds.low)
        assert np.array_equal(composite.operational_bounds.high, sphere.operational_bounds.high)
        assert len(composite.components) == 2
        assert np.array_equal(composite.sigmas, np.array([1.0, 2.0]))
        assert np.array_equal(composite.lambdas, np.array([1.0, 1.0]))
        assert np.array_equal(composite.biases, np.array([0.0, 100.0]))
        
        # Create a composite function with custom bounds
        custom_init_bounds = Bounds(
            low=np.array([-10, -5]),
            high=np.array([10, 5]),
            mode=BoundModeEnum.INITIALIZATION,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        custom_oper_bounds = Bounds(
            low=np.array([-10, -5]),
            high=np.array([10, 5]),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        composite = CompositeFunction(
            components, sigmas, lambdas, biases,
            initialization_bounds=custom_init_bounds,
            operational_bounds=custom_oper_bounds
        )
        
        assert composite.dimension == 2
        assert np.array_equal(composite.initialization_bounds.low, custom_init_bounds.low)
        assert np.array_equal(composite.operational_bounds.high, custom_oper_bounds.high)
    
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
        shifted_sphere = Shifted(base_function=sphere, shift=np.array([1.0, 2.0]))
        
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
    
    def test_decorator_compatibility(self):
        """Test composite function with decorated (shifted, biased) components and quantized bounds."""
        sphere = SphereFunction(dimension=2)
        shifted = Shifted(base_function=sphere, shift=np.array([1.0, 1.0]))
        biased = Biased(base_function=shifted, bias=5.0)
        quantized = Quantized(base_function=biased, qtype=QuantizationTypeEnum.INTEGER)
        components = [quantized]
        sigmas = [1.0]
        lambdas = [1.0]
        biases = [0.0]
        composite = CompositeFunction(components, sigmas, lambdas, biases)
        # Should round input to integer, shift, then bias
        x = np.array([2.7, 3.2])
        # After quantization: [3, 3], shift: [2, 2], sphere: 8, bias: 13
        assert np.isclose(composite.evaluate(x), 13.0) 