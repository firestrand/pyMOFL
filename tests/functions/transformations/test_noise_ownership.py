"""V4.1: owner-authorized streams (V0.2.2), actual DATA-01 objective values.

NumPy PCG64/MT19937 seed 20261003; installed NumPy version is recorded at phase
verification. These are stream ownership checks, not external reference captures.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pyMOFL.factories.data_loader import DataLoader
from pyMOFL.factories.transform_builder import TransformBuilder
from pyMOFL.functions.transformations import (
    CauchyNoiseTransform,
    GaussianNoiseTransform,
    NoiseTransform,
    UniformNoiseTransform,
)

SEED = 20261003
CASES = [
    ("noise", NoiseTransform, {}),
    ("gaussian_noise", GaussianNoiseTransform, {"beta": 1.0}),
    ("uniform_noise", UniformNoiseTransform, {"alpha": 0.01, "beta": 0.01}),
    ("cauchy_noise", CauchyNoiseTransform, {"alpha": 0.01, "p": 0.05}),
]


@pytest.fixture(autouse=True)
def restore_global_random_state():
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


@pytest.fixture
def captured_values():
    path = Path(__file__).parents[2] / "validation_data/cec/2005/f01.json"
    case = next(c for c in json.loads(path.read_text())["cases"] if c["dimension"] == 10)
    return np.asarray([case["outputs"][key] for key in ("optimum", "random", "lower", "upper")])


def assert_global_state_equal(before, after):
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("kind,cls,params", CASES)
def test_explicit_generators_replay_without_global_state_changes(
    kind, cls, params, captured_values
):
    first_rng = np.random.Generator(np.random.PCG64(SEED))
    second_rng = np.random.Generator(np.random.PCG64(SEED))
    before = np.random.get_state()
    first = cls(**params, rng=first_rng)
    second = cls(**params, rng=second_rng)
    initial_stream_state = first_rng.bit_generator.state
    original = captured_values.copy()
    for _ in range(2):
        np.testing.assert_array_equal(
            first.transform_batch(captured_values), second.transform_batch(captured_values)
        )
        assert first(float(captured_values[0])) == second(float(captured_values[0]))
    assert first_rng.bit_generator.state != initial_stream_state
    assert_global_state_equal(before, np.random.get_state())
    np.testing.assert_array_equal(captured_values, original)


@pytest.mark.parametrize("kind,cls,params", CASES)
def test_seed_and_generator_are_rejected_before_reseeding(kind, cls, params):
    before = np.random.get_state()
    with pytest.raises(ValueError, match="seed"):
        cls(**params, seed=SEED, rng=np.random.default_rng(SEED))
    with pytest.raises(TypeError, match="Generator"):
        cls(**params, rng=np.random.RandomState(SEED))
    assert_global_state_equal(before, np.random.get_state())


@pytest.mark.parametrize("kind,cls,params", CASES)
def test_programmatic_builder_forwards_generator(kind, cls, params, captured_values):
    rng = np.random.default_rng(SEED)
    before = np.random.get_state()
    loader = DataLoader(base_path=Path(__file__).parents[3] / "src/pyMOFL/constants")
    built = TransformBuilder(loader).build(kind, {**params, "rng": rng}, dimension=10)
    expected = cls(**params, rng=np.random.default_rng(SEED))
    np.testing.assert_array_equal(
        built.transform_batch(captured_values), expected.transform_batch(captured_values)
    )
    assert_global_state_equal(before, np.random.get_state())


def test_cec_noise_preserves_legacy_global_sequence(captured_values):
    reference = np.random.RandomState(SEED)
    transform = NoiseTransform(seed=SEED)
    expected = captured_values * (1 + 0.4 * np.abs(reference.randn(*captured_values.shape)))
    out = np.empty_like(captured_values)
    assert transform.transform_batch(captured_values, out=out) is out
    np.testing.assert_array_equal(out, expected)
    value = float(captured_values[0])
    assert transform(value) == value * (1 + 0.4 * abs(reference.randn()))
    expected_array = captured_values * (1 + 0.4 * np.abs(reference.randn(*captured_values.shape)))
    np.testing.assert_array_equal(transform(captured_values), expected_array)
    assert_global_state_equal(reference.get_state(), np.random.get_state())


def test_cec_noise_injected_stream_matches_formula_and_buffer(captured_values):
    rng = np.random.default_rng(SEED)
    reference = np.random.default_rng(SEED)
    transform = NoiseTransform(rng=rng)
    value = float(captured_values[0])
    assert transform(value) == value * (1 + 0.4 * abs(reference.standard_normal()))
    expected_array = captured_values * (
        1 + 0.4 * np.abs(reference.standard_normal(captured_values.shape))
    )
    np.testing.assert_array_equal(transform(captured_values), expected_array)
    out = np.empty_like(captured_values)
    expected_batch = captured_values * (
        1 + 0.4 * np.abs(reference.standard_normal(captured_values.shape))
    )
    assert transform.transform_batch(captured_values, out=out) is out
    np.testing.assert_array_equal(out, expected_batch)


@pytest.mark.parametrize("kind,cls,params", CASES[1:])
def test_existing_local_seed_defaults_preserve_global_state(kind, cls, params, captured_values):
    before = np.random.get_state()
    first = cls(**params, seed=SEED)
    second = cls(**params, seed=SEED)
    np.testing.assert_array_equal(
        first.transform_batch(captured_values), second.transform_batch(captured_values)
    )
    assert_global_state_equal(before, np.random.get_state())
