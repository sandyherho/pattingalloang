"""Pytest configuration and fixtures for pattingalloang tests."""

import pytest
import numpy as np


@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def default_params():
    """Default Aizawa parameters."""
    return {
        'a': 0.95,
        'b': 0.7,
        'c': 0.6,
        'd': 3.5,
        'e': 0.25,
        'f': 0.1,
    }


@pytest.fixture
def default_integration_params():
    """Default integration parameters."""
    return {
        'dt': 0.01,
        'n_steps': 10000,
        'transient_steps': 1000,
    }
