"""
Tests for multi-basin composition and specialized transforms.
"""

import numpy as np

from pyMOFL.compositions.min_composition import MinComposition
from pyMOFL.functions.benchmark.multi_basin import MultiBasinFunction
from pyMOFL.functions.benchmark.sphere import SphereFunction
from pyMOFL.functions.transformations.log_sin_transform import LogSinTransform
from pyMOFL.functions.transformations.power import PowerTransform


def test_power_transform():
    """Test the PowerTransform scalar transformation."""
    pt = PowerTransform(exponent=0.5)
    assert pt(4.0) == 2.0
    assert pt(0.0) == 0.0
    assert pt(-1.0) == 0.0  # Clipping to 0 for non-negative inputs

    Y = np.array([1.0, 4.0, 9.0])
    np.testing.assert_allclose(pt.transform_batch(Y), [1.0, 2.0, 3.0])


def test_log_sin_transform():
    """Test the LogSinTransform vector transformation."""
    # Identity-like near 0 but with sinusoids
    gt = LogSinTransform(mu=(0.0, 0.0), omega=(0.0, 0.0, 0.0, 0.0))
    x = np.array([1.0, -1.0, 0.0])
    np.testing.assert_allclose(gt(x), x)

    # Test with non-zero mu/omega
    gt2 = LogSinTransform(mu=(1.0, 1.0), omega=(1.0, 1.0, 1.0, 1.0))
    val = np.exp(1.0 + 2.0 * np.sin(1.0))
    np.testing.assert_allclose(gt2(np.array([np.e])), [val])


def test_min_composition():
    """Test the MinComposition."""
    f1 = SphereFunction(dimension=2)
    # Sphere centered at (2, 2)
    from pyMOFL.functions.transformations import ComposedFunction, ShiftTransform

    f2 = ComposedFunction(
        base_function=SphereFunction(dimension=2),
        input_transforms=[ShiftTransform(np.array([2.0, 2.0]))],
    )

    mc = MinComposition(dimension=2, components=[f1, f2])

    # At (0,0), f1 is 0, f2 is 8. Min is 0.
    assert mc.evaluate(np.array([0.0, 0.0])) == 0.0
    # At (2,2), f1 is 8, f2 is 0. Min is 0.
    assert mc.evaluate(np.array([2.0, 2.0])) == 0.0
    # At (1,1), f1 is 2, f2 is 2. Min is 2.
    assert mc.evaluate(np.array([1.0, 1.0])) == 2.0


def test_multi_basin_function():
    """Test the MultiBasinFunction (GNBG generator)."""
    # 2 basins, quadratic
    centers = np.array([[0.0, 0.0], [5.0, 5.0]])
    biases = np.array([0.0, 10.0])

    func = MultiBasinFunction(
        dimension=2, n_components=2, centers=centers, biases=biases, rotate=False
    )

    assert len(func.components) == 2
    # Global min at centers[0]
    assert func.evaluate(centers[0]) == 0.0
    # Local min at centers[1]
    assert func.evaluate(centers[1]) == 10.0


def test_factory_gnbg():
    """Test creating a MultiBasinFunction via the 'gnbg' alias."""
    from pyMOFL.factories.function_factory import FunctionFactory

    config = {
        "type": "gnbg",
        "parameters": {
            "dimension": 2,
            "n_components": 2,
            "centers": [[0.0, 0.0], [10.0, 10.0]],
            "biases": [0.0, 5.0],
            "rotate": False,
        },
    }

    factory = FunctionFactory()
    func = factory.create_function(config)

    assert isinstance(func.base_function, MultiBasinFunction)
    assert func.evaluate(np.array([0.0, 0.0])) == 0.0
    assert func.evaluate(np.array([10.0, 10.0])) == 5.0
