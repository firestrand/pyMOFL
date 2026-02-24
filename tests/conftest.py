"""
Pytest configuration file for pyMOFL tests.

This file contains common fixtures and settings for the test suite.
"""

import numpy as np
import pytest

# NOTE: Tests require the pyMOFL package to be installed in editable mode.
# Run 'pip install -e .' in the project root if you see import errors.


@pytest.fixture
def random_vector():
    """
    Generate a random vector for testing.

    Returns:
        np.ndarray: A random vector of length 2.
    """
    return np.random.rand(2)


@pytest.fixture
def random_matrix():
    """
    Generate a random matrix for testing.

    Returns:
        np.ndarray: A random matrix of shape (2, 2).
    """
    return np.random.rand(2, 2)


@pytest.fixture
def random_batch():
    """
    Generate a random batch of vectors for testing.

    Returns:
        np.ndarray: A random batch of 5 vectors of length 2.
    """
    return np.random.rand(5, 2)
