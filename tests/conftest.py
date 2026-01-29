"""Pytest configuration and fixtures for RapidPhase tests."""

import numpy as np
import pytest
import torch


@pytest.fixture
def simple_phase():
    """Create a simple smooth phase ramp for testing."""
    H, W = 64, 64
    y, x = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
    # Create a smooth phase that wraps a few times
    phase = 6 * np.pi * (x + y)
    return phase


@pytest.fixture
def wrapped_phase(simple_phase):
    """Create wrapped version of simple phase."""
    return np.arctan2(np.sin(simple_phase), np.cos(simple_phase))


@pytest.fixture
def coherence_map():
    """Create a coherence map with high values."""
    H, W = 64, 64
    # High coherence everywhere
    coherence = 0.9 * np.ones((H, W))
    # Add some variation
    y, x = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
    coherence -= 0.2 * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)
    coherence = np.clip(coherence, 0.3, 1.0)
    return coherence


@pytest.fixture
def complex_interferogram(wrapped_phase):
    """Create a complex interferogram from wrapped phase."""
    return np.exp(1j * wrapped_phase)


@pytest.fixture
def device_manager():
    """Create a DeviceManager for testing."""
    from rapidphase.device.manager import DeviceManager
    return DeviceManager(device="cpu")  # Use CPU for consistent tests


@pytest.fixture
def available_device():
    """Return the best available device for testing."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"
