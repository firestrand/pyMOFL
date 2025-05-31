import numpy as np
import pytest
from pyMOFL.functions.unimodal.sphere import SphereFunction
from pyMOFL.decorators.noise import Noise
from pyMOFL.decorators.biased import Biased

class TestNoise:
    def test_gaussian_noise_single(self):
        base = SphereFunction(dimension=1)
        noise = Noise(base_function=base, noise_type='gaussian', noise_level=0.5, noise_seed=123)
        val = noise(np.array([2.0]))
        expected = base(np.array([2.0]))
        assert val != expected

    def test_uniform_noise_single(self):
        base = SphereFunction(dimension=1)
        noise = Noise(base_function=base, noise_type='uniform', noise_level=0.5, noise_seed=123)
        val = noise(np.array([2.0]))
        expected = base(np.array([2.0]))
        assert val != expected

    def test_gaussian_noise_batch(self):
        base = SphereFunction(dimension=2)
        noise = Noise(base_function=base, noise_type='gaussian', noise_level=0.1, noise_seed=42)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        vals = noise.evaluate_batch(X)
        expected = base.evaluate_batch(X)
        assert vals.shape == expected.shape
        assert not np.allclose(vals, expected)

    def test_uniform_noise_batch(self):
        base = SphereFunction(dimension=2)
        noise = Noise(base_function=base, noise_type='uniform', noise_level=0.1, noise_seed=42)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        vals = noise.evaluate_batch(X)
        expected = base.evaluate_batch(X)
        assert vals.shape == expected.shape
        assert not np.allclose(vals, expected)

    def test_property_delegation(self):
        base = SphereFunction(dimension=2)
        noise = Noise(base_function=base, noise_type='gaussian', noise_level=0.1)
        assert noise.dimension == base.dimension
        assert noise.initialization_bounds == base.initialization_bounds
        assert noise.operational_bounds == base.operational_bounds

    def test_composability(self):
        base = SphereFunction(dimension=2)
        biased = Biased(base_function=base, bias=5.0)
        noise = Noise(base_function=biased, noise_type='gaussian', noise_level=0.1, noise_seed=42)
        x = np.array([1.0, 2.0])
        val = noise(x)
        expected = biased(x)
        assert not np.allclose(val, expected)

    def test_error_on_unsupported_noise_type(self):
        base = SphereFunction(dimension=1)
        with pytest.raises(ValueError):
            Noise(base_function=base, noise_type='unsupported') 