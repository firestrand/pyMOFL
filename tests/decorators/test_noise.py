"""
Tests for the NoiseDecorator (refactored for new bounds logic).
"""

import pytest
import numpy as np
from pyMOFL.functions.unimodal.sphere import SphereFunction
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.decorators.noise import NoiseDecorator
from pyMOFL.decorators.biased import BiasedFunction

class TestNoiseDecorator:
    def test_initialization_and_delegation(self):
        func = SphereFunction(dimension=2)
        noise_func = NoiseDecorator(func, noise_type='gaussian', noise_level=0.2, noise_seed=42)
        assert noise_func.dimension == 2
        assert noise_func.initialization_bounds == func.initialization_bounds
        assert noise_func.operational_bounds == func.operational_bounds

    def test_gaussian_noise_single(self):
        func = SphereFunction(dimension=1)
        noise_func = NoiseDecorator(func, noise_type='gaussian', noise_level=0.5, noise_seed=123)
        # With fixed seed, noise is deterministic
        val = noise_func(np.array([2.0]))
        expected = func(np.array([2.0]))
        assert val != expected
        assert np.isclose(val / expected, 1.0 + noise_func.noise_level * np.abs(np.random.normal()))

    def test_uniform_noise_single(self):
        func = SphereFunction(dimension=1)
        noise_func = NoiseDecorator(func, noise_type='uniform', noise_level=0.5, noise_seed=123)
        val = noise_func(np.array([2.0]))
        expected = func(np.array([2.0]))
        assert val != expected
        # Uniform noise is in [1, 1.5]
        assert 1.0 <= val / expected <= 1.5

    def test_gaussian_noise_batch(self):
        func = SphereFunction(dimension=2)
        noise_func = NoiseDecorator(func, noise_type='gaussian', noise_level=0.1, noise_seed=42)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        vals = noise_func.evaluate_batch(X)
        expected = func.evaluate_batch(X)
        assert vals.shape == expected.shape
        assert not np.allclose(vals, expected)

    def test_uniform_noise_batch(self):
        func = SphereFunction(dimension=2)
        noise_func = NoiseDecorator(func, noise_type='uniform', noise_level=0.1, noise_seed=42)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        vals = noise_func.evaluate_batch(X)
        expected = func.evaluate_batch(X)
        assert vals.shape == expected.shape
        assert not np.allclose(vals, expected)

    def test_composability(self):
        func = SphereFunction(dimension=2)
        biased = BiasedFunction(func, bias=5.0)
        noise_func = NoiseDecorator(biased, noise_type='gaussian', noise_level=0.1, noise_seed=42)
        x = np.array([1.0, 2.0])
        val = noise_func(x)
        expected = biased(x)
        assert val != expected

    def test_error_on_unsupported_noise_type(self):
        func = SphereFunction(dimension=1)
        with pytest.raises(ValueError):
            NoiseDecorator(func, noise_type='unsupported') 