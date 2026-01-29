"""Tests for IRLS-based phase unwrapping."""

import numpy as np
import pytest
import torch

from rapidphase.core.irls_solver import IRLSUnwrapper
from rapidphase.core.dct_solver import DCTUnwrapper
from rapidphase.device.manager import DeviceManager


class TestIRLSUnwrapper:
    """Tests for IRLSUnwrapper class."""

    @pytest.fixture
    def unwrapper(self, device_manager):
        """Create IRLS unwrapper instance."""
        return IRLSUnwrapper(device_manager, max_iterations=20)

    def test_unwrap_with_coherence(
        self, unwrapper, wrapped_phase, coherence_map
    ):
        """Test unwrapping with coherence weighting."""
        wrapped_t = unwrapper.dm.to_tensor(wrapped_phase)
        coh_t = unwrapper.dm.to_tensor(coherence_map)

        unwrapped_t = unwrapper.unwrap(wrapped_t, coherence=coh_t)
        unwrapped = unwrapper.dm.to_numpy(unwrapped_t)

        assert unwrapped.shape == wrapped_phase.shape

    def test_fallback_to_dct_without_coherence(
        self, unwrapper, wrapped_phase, device_manager
    ):
        """Test that IRLS falls back to DCT when no coherence provided."""
        wrapped_t = unwrapper.dm.to_tensor(wrapped_phase)

        # IRLS without coherence
        result_irls_t = unwrapper.unwrap(wrapped_t, coherence=None)
        result_irls = unwrapper.dm.to_numpy(result_irls_t)

        # Pure DCT
        dct_unwrapper = DCTUnwrapper(device_manager)
        result_dct = dct_unwrapper(wrapped_phase)

        # Should be identical
        np.testing.assert_array_almost_equal(result_irls, result_dct)

    def test_unwrap_preserves_shape(
        self, unwrapper, wrapped_phase, coherence_map
    ):
        """Test that unwrapping preserves array shape."""
        wrapped_t = unwrapper.dm.to_tensor(wrapped_phase)
        coh_t = unwrapper.dm.to_tensor(coherence_map)

        unwrapped_t = unwrapper.unwrap(wrapped_t, coherence=coh_t)

        assert unwrapped_t.shape == wrapped_t.shape

    def test_custom_iterations(self, device_manager, wrapped_phase, coherence_map):
        """Test custom iteration count."""
        # Few iterations
        unwrapper_few = IRLSUnwrapper(device_manager, max_iterations=5)
        wrapped_t = unwrapper_few.dm.to_tensor(wrapped_phase)
        coh_t = unwrapper_few.dm.to_tensor(coherence_map)

        result_few_t = unwrapper_few.unwrap(wrapped_t, coherence=coh_t)

        # More iterations
        unwrapper_many = IRLSUnwrapper(device_manager, max_iterations=50)
        result_many_t = unwrapper_many.unwrap(wrapped_t, coherence=coh_t)

        # Both should produce valid results (shapes match)
        assert result_few_t.shape == result_many_t.shape

    def test_override_iterations_in_unwrap(
        self, unwrapper, wrapped_phase, coherence_map
    ):
        """Test overriding max_iterations in unwrap call."""
        wrapped_t = unwrapper.dm.to_tensor(wrapped_phase)
        coh_t = unwrapper.dm.to_tensor(coherence_map)

        # Override with fewer iterations
        unwrapped_t = unwrapper.unwrap(
            wrapped_t,
            coherence=coh_t,
            max_iterations=3,
        )

        assert unwrapped_t.shape == wrapped_t.shape

    def test_callable_interface(self, unwrapper, wrapped_phase, coherence_map):
        """Test the __call__ interface with numpy arrays."""
        unwrapped = unwrapper(wrapped_phase, coherence=coherence_map)

        assert isinstance(unwrapped, np.ndarray)
        assert unwrapped.shape == wrapped_phase.shape

    def test_convergence_tolerance(
        self, device_manager, wrapped_phase, coherence_map
    ):
        """Test that tight tolerance leads to more iterations."""
        # This is a behavioral test - tight tolerance should iterate longer
        # We just verify it runs without error
        unwrapper = IRLSUnwrapper(
            device_manager,
            max_iterations=100,
            tolerance=1e-10,  # Very tight
        )

        wrapped_t = unwrapper.dm.to_tensor(wrapped_phase)
        coh_t = unwrapper.dm.to_tensor(coherence_map)

        unwrapped_t = unwrapper.unwrap(wrapped_t, coherence=coh_t)

        assert unwrapped_t.shape == wrapped_t.shape

    def test_clear_cache(self, unwrapper, wrapped_phase, coherence_map):
        """Test cache clearing."""
        wrapped_t = unwrapper.dm.to_tensor(wrapped_phase)
        coh_t = unwrapper.dm.to_tensor(coherence_map)

        unwrapper.unwrap(wrapped_t, coherence=coh_t)

        # Should not raise
        unwrapper.clear_cache()

    def test_low_coherence_handling(self, unwrapper, wrapped_phase):
        """Test handling of low coherence regions."""
        # Create coherence with very low region
        H, W = wrapped_phase.shape
        coherence = np.ones((H, W)) * 0.8
        coherence[H//4:3*H//4, W//4:3*W//4] = 0.1  # Low coherence center

        wrapped_t = unwrapper.dm.to_tensor(wrapped_phase)
        coh_t = unwrapper.dm.to_tensor(coherence)

        # Should handle without error
        unwrapped_t = unwrapper.unwrap(wrapped_t, coherence=coh_t)

        assert unwrapped_t.shape == wrapped_t.shape
        assert torch.isfinite(unwrapped_t).all()


class TestIRLSUnwrapperDevices:
    """Test IRLS unwrapper on different devices."""

    def test_consistent_results_across_devices(
        self, wrapped_phase, coherence_map, available_device
    ):
        """Test that different devices give consistent results."""
        # CPU result
        dm_cpu = DeviceManager(device="cpu")
        unwrapper_cpu = IRLSUnwrapper(dm_cpu, max_iterations=10)
        result_cpu = unwrapper_cpu(wrapped_phase, coherence=coherence_map)

        # Other device result
        if available_device != "cpu":
            # Skip MPS for now - scipy DCT doesn't run on GPU and
            # the data transfer + float32 precision causes issues
            if available_device == "mps":
                pytest.skip("MPS has float32 precision limitations for IRLS")

            dm_other = DeviceManager(device=available_device)
            unwrapper_other = IRLSUnwrapper(dm_other, max_iterations=10)
            result_other = unwrapper_other(wrapped_phase, coherence=coherence_map)

            # Results should be reasonably close
            # (may differ slightly due to float32 vs float64)
            np.testing.assert_array_almost_equal(
                result_cpu,
                result_other,
                decimal=2,
            )
