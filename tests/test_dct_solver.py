"""Tests for DCT-based phase unwrapping."""

import numpy as np
import pytest
import torch

from rapidphase.core.dct_solver import DCTUnwrapper
from rapidphase.device.manager import DeviceManager


class TestDCTUnwrapper:
    """Tests for DCTUnwrapper class."""

    @pytest.fixture
    def unwrapper(self, device_manager):
        """Create DCT unwrapper instance."""
        return DCTUnwrapper(device_manager)

    def test_unwrap_smooth_phase(self, unwrapper, simple_phase, wrapped_phase):
        """Test unwrapping a smooth phase ramp."""
        # Convert to tensor
        wrapped_t = unwrapper.dm.to_tensor(wrapped_phase)

        # Unwrap
        unwrapped_t = unwrapper.unwrap(wrapped_t)
        unwrapped = unwrapper.dm.to_numpy(unwrapped_t)

        # The unwrapped phase should have the same gradient as original
        # (may differ by a constant offset)
        grad_orig_x = np.diff(simple_phase, axis=1)
        grad_unw_x = np.diff(unwrapped, axis=1)

        # Check gradients match (ignoring boundary effects)
        inner_slice = (slice(5, -5), slice(5, -5))
        np.testing.assert_array_almost_equal(
            grad_orig_x[inner_slice],
            grad_unw_x[inner_slice],
            decimal=1,
        )

    def test_unwrap_preserves_shape(self, unwrapper, wrapped_phase):
        """Test that unwrapping preserves array shape."""
        wrapped_t = unwrapper.dm.to_tensor(wrapped_phase)
        unwrapped_t = unwrapper.unwrap(wrapped_t)

        assert unwrapped_t.shape == wrapped_t.shape

    def test_unwrap_constant_phase(self, unwrapper):
        """Test unwrapping constant phase (should remain constant)."""
        constant = np.ones((32, 32)) * 0.5
        constant_t = unwrapper.dm.to_tensor(constant)

        unwrapped_t = unwrapper.unwrap(constant_t)
        unwrapped = unwrapper.dm.to_numpy(unwrapped_t)

        # Should be nearly constant (allow for numerical precision)
        assert np.std(unwrapped) < 0.01

    def test_unwrap_callable(self, unwrapper, wrapped_phase):
        """Test the __call__ interface with numpy arrays."""
        unwrapped = unwrapper(wrapped_phase)

        assert isinstance(unwrapped, np.ndarray)
        assert unwrapped.shape == wrapped_phase.shape

    def test_coherence_parameter_ignored(self, unwrapper, wrapped_phase, coherence_map):
        """Test that DCT unwrapper ignores coherence (it's for IRLS)."""
        wrapped_t = unwrapper.dm.to_tensor(wrapped_phase)
        coh_t = unwrapper.dm.to_tensor(coherence_map)

        # Should work without error (coherence is ignored)
        unwrapped_t = unwrapper.unwrap(wrapped_t, coherence=coh_t)

        assert unwrapped_t.shape == wrapped_t.shape

    def test_eigenvalue_caching(self, unwrapper, wrapped_phase):
        """Test that eigenvalues are cached for same-sized inputs."""
        wrapped_t = unwrapper.dm.to_tensor(wrapped_phase)

        # First call should populate cache
        unwrapper.unwrap(wrapped_t)
        cache_size_1 = len(unwrapper._eigenvalue_cache)

        # Second call should reuse cache
        unwrapper.unwrap(wrapped_t)
        cache_size_2 = len(unwrapper._eigenvalue_cache)

        assert cache_size_1 == cache_size_2

    def test_clear_cache(self, unwrapper, wrapped_phase):
        """Test cache clearing."""
        wrapped_t = unwrapper.dm.to_tensor(wrapped_phase)
        unwrapper.unwrap(wrapped_t)

        assert len(unwrapper._eigenvalue_cache) > 0

        unwrapper.clear_cache()

        assert len(unwrapper._eigenvalue_cache) == 0

    def test_dct_idct_inverse(self, unwrapper):
        """Test that IDCT(DCT(x)) = x."""
        x = torch.randn(32, 32, dtype=unwrapper.dtype, device=unwrapper.device)

        x_dct = unwrapper._dct2(x)
        x_reconstructed = unwrapper._idct2(x_dct)

        # Should be close to original
        torch.testing.assert_close(x_reconstructed, x, rtol=1e-4, atol=1e-4)

    def test_unwrap_different_sizes(self, unwrapper):
        """Test unwrapping with different image sizes."""
        for size in [(32, 32), (64, 64), (48, 64), (100, 80)]:
            H, W = size
            y, x = np.meshgrid(
                np.linspace(0, 1, H),
                np.linspace(0, 1, W),
                indexing='ij'
            )
            phase = 4 * np.pi * (x + y)
            wrapped = np.arctan2(np.sin(phase), np.cos(phase))

            wrapped_t = unwrapper.dm.to_tensor(wrapped)
            unwrapped_t = unwrapper.unwrap(wrapped_t)

            assert unwrapped_t.shape == (H, W)


class TestDCTUnwrapperDevices:
    """Test DCT unwrapper on different devices."""

    def test_consistent_results_across_devices(
        self, wrapped_phase, available_device
    ):
        """Test that different devices give consistent results."""
        # CPU result
        dm_cpu = DeviceManager(device="cpu")
        unwrapper_cpu = DCTUnwrapper(dm_cpu)
        result_cpu = unwrapper_cpu(wrapped_phase)

        # Other device result
        if available_device != "cpu":
            dm_other = DeviceManager(device=available_device)
            unwrapper_other = DCTUnwrapper(dm_other)
            result_other = unwrapper_other(wrapped_phase)

            # Results should be very close
            np.testing.assert_array_almost_equal(
                result_cpu,
                result_other,
                decimal=4,
            )
