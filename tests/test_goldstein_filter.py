"""
Tests for Goldstein adaptive filter.
"""

import numpy as np
import pytest
import torch

from rapidphase.device.manager import DeviceManager
from rapidphase.filtering.goldstein import GoldsteinFilter
from rapidphase import goldstein_filter


@pytest.fixture
def device_manager():
    """Create a device manager for testing."""
    return DeviceManager("cpu")


@pytest.fixture
def sample_interferogram():
    """Create a sample complex interferogram for testing."""
    np.random.seed(42)
    size = 128
    x, y = np.meshgrid(np.linspace(-3, 3, size), np.linspace(-3, 3, size))

    # Create phase pattern
    phase = 2 * np.pi * (np.sin(x) + np.sin(y))

    # Add noise
    noise = 0.5 * np.random.normal(0, 1, phase.shape)

    # Create complex interferogram
    interferogram = np.exp(1j * (phase + noise))

    return interferogram


class TestGoldsteinFilter:
    """Tests for GoldsteinFilter class."""

    def test_filter_preserves_shape(self, device_manager, sample_interferogram):
        """Filter output should have same shape as input."""
        filt = GoldsteinFilter(device_manager, alpha=0.6, window_size=32)
        igram_t = torch.from_numpy(sample_interferogram).to(device_manager.device)

        filtered = filt.filter(igram_t)

        assert filtered.shape == igram_t.shape

    def test_filter_reduces_noise(self, device_manager, sample_interferogram):
        """Filter should reduce high-frequency noise."""
        filt = GoldsteinFilter(device_manager, alpha=0.6, window_size=32)

        filtered = filt(sample_interferogram)

        # Filtered phase should have lower variance in the noise component
        orig_phase = np.angle(sample_interferogram)
        filt_phase = np.angle(filtered)

        # Check that the filtered result is smoother (lower gradient magnitude)
        orig_grad = np.gradient(orig_phase)
        filt_grad = np.gradient(filt_phase)

        orig_grad_mag = np.sqrt(orig_grad[0] ** 2 + orig_grad[1] ** 2)
        filt_grad_mag = np.sqrt(filt_grad[0] ** 2 + filt_grad[1] ** 2)

        # Filtered should have lower gradient variance (smoother)
        assert np.var(filt_grad_mag) < np.var(orig_grad_mag)

    def test_filter_handles_nan(self, device_manager, sample_interferogram):
        """Filter should preserve NaN positions."""
        # Add NaN values
        igram_with_nan = sample_interferogram.copy()
        nan_mask = np.random.rand(*igram_with_nan.shape) < 0.1
        igram_with_nan[nan_mask] = np.nan

        filt = GoldsteinFilter(device_manager, alpha=0.6, window_size=32)
        filtered = filt(igram_with_nan)

        # NaN positions should be preserved
        output_nan_mask = np.isnan(filtered)
        assert np.array_equal(nan_mask, output_nan_mask)

    def test_filter_handles_zeros(self, device_manager, sample_interferogram):
        """Filter should handle zero values without issues."""
        # Add zero values
        igram_with_zeros = sample_interferogram.copy()
        zero_mask = np.random.rand(*igram_with_zeros.shape) < 0.1
        igram_with_zeros[zero_mask] = 0

        filt = GoldsteinFilter(device_manager, alpha=0.6, window_size=32)

        # Should not raise an error
        filtered = filt(igram_with_zeros)

        assert filtered.shape == sample_interferogram.shape
        assert not np.any(np.isnan(filtered[~zero_mask]))

    def test_alpha_parameter(self, device_manager, sample_interferogram):
        """Higher alpha should result in stronger filtering."""
        filt_low = GoldsteinFilter(device_manager, alpha=0.2, window_size=32)
        filt_high = GoldsteinFilter(device_manager, alpha=0.8, window_size=32)

        filtered_low = filt_low(sample_interferogram)
        filtered_high = filt_high(sample_interferogram)

        # Higher alpha should produce smoother result (lower gradient variance)
        grad_low = np.gradient(np.angle(filtered_low))
        grad_high = np.gradient(np.angle(filtered_high))

        var_low = np.var(grad_low[0]) + np.var(grad_low[1])
        var_high = np.var(grad_high[0]) + np.var(grad_high[1])

        assert var_high < var_low

    def test_window_size_parameter(self, device_manager, sample_interferogram):
        """Different window sizes should produce valid results."""
        for ws in [16, 32, 64]:
            filt = GoldsteinFilter(device_manager, alpha=0.6, window_size=ws)
            filtered = filt(sample_interferogram)

            assert filtered.shape == sample_interferogram.shape
            assert not np.any(np.isnan(filtered))

    def test_small_image(self, device_manager):
        """Filter should handle images smaller than window size."""
        small_igram = np.exp(1j * np.random.randn(32, 32))

        filt = GoldsteinFilter(device_manager, alpha=0.6, window_size=64)
        filtered = filt(small_igram)

        assert filtered.shape == small_igram.shape

    def test_non_square_image(self, device_manager):
        """Filter should handle non-square images."""
        rect_igram = np.exp(1j * np.random.randn(100, 150))

        filt = GoldsteinFilter(device_manager, alpha=0.6, window_size=32)
        filtered = filt(rect_igram)

        assert filtered.shape == rect_igram.shape


class TestGoldsteinFilterAPI:
    """Tests for the goldstein_filter API function."""

    def test_basic_usage(self, sample_interferogram):
        """Basic API usage should work."""
        filtered = goldstein_filter(sample_interferogram, alpha=0.6, window_size=32)

        assert filtered.shape == sample_interferogram.shape
        assert np.iscomplexobj(filtered)

    def test_default_parameters(self, sample_interferogram):
        """Default parameters (alpha=0.6, window_size=64) should work."""
        filtered = goldstein_filter(sample_interferogram)

        assert filtered.shape == sample_interferogram.shape

    def test_with_nan_values(self, sample_interferogram):
        """API should handle NaN values correctly."""
        igram_with_nan = sample_interferogram.copy()
        nan_mask = np.random.rand(*igram_with_nan.shape) < 0.1
        igram_with_nan[nan_mask] = np.nan

        filtered = goldstein_filter(igram_with_nan, alpha=0.6, window_size=32)

        # NaN positions should be preserved
        assert np.array_equal(nan_mask, np.isnan(filtered))

    def test_device_cpu(self, sample_interferogram):
        """Explicit CPU device should work."""
        filtered = goldstein_filter(
            sample_interferogram, alpha=0.6, window_size=32, device="cpu"
        )

        assert filtered.shape == sample_interferogram.shape

    def test_returns_numpy(self, sample_interferogram):
        """API should return numpy array."""
        filtered = goldstein_filter(sample_interferogram)

        assert isinstance(filtered, np.ndarray)


class TestGoldsteinFilterIntegration:
    """Integration tests for filter + unwrap pipeline."""

    def test_filter_then_unwrap(self, sample_interferogram):
        """Filtering before unwrapping should work."""
        from rapidphase import unwrap

        # Filter first
        filtered = goldstein_filter(sample_interferogram, alpha=0.6, window_size=32)

        # Then unwrap
        unw, conncomp = unwrap(filtered, algorithm="dct", device="cpu")

        assert unw.shape == sample_interferogram.shape
        assert not np.any(np.isnan(unw))
