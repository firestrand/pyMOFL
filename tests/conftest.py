"""
Pytest configuration file for pyMOFL tests.

This file contains common fixtures and settings for the test suite.
"""

import pytest
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


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