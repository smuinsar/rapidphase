"""Tests for IRLS-CG phase unwrapping."""

import numpy as np
import pytest
import torch

from rapidphase.core.irls_cg_solver import IRLSCGUnwrapper
from rapidphase.core.dct_solver import DCTUnwrapper
from rapidphase.device.manager import DeviceManager


class TestIRLSCGUnwrapper:
    """Tests for IRLSCGUnwrapper class."""

    @pytest.fixture
    def unwrapper(self, device_manager):
        """Create IRLS-CG unwrapper instance."""
        return IRLSCGUnwrapper(device_manager, max_irls_iterations=10)

    def test_unwrap_with_coherence(
        self, unwrapper, wrapped_phase, coherence_map
    ):
        """Test unwrapping with coherence weighting."""
        wrapped_t = unwrapper.dm.to_tensor(wrapped_phase)
        coh_t = unwrapper.dm.to_tensor(coherence_map)

        unwrapped_t = unwrapper.unwrap(wrapped_t, coherence=coh_t)
        unwrapped = unwrapper.dm.to_numpy(unwrapped_t)

        assert unwrapped.shape == wrapped_phase.shape

    def test_unwrap_without_coherence(
        self, unwrapper, wrapped_phase
    ):
        """Test unwrapping without coherence (uniform weights)."""
        wrapped_t = unwrapper.dm.to_tensor(wrapped_phase)

        unwrapped_t = unwrapper.unwrap(wrapped_t, coherence=None)
        unwrapped = unwrapper.dm.to_numpy(unwrapped_t)

        assert unwrapped.shape == wrapped_phase.shape

    def test_unwrap_preserves_shape(
        self, unwrapper, wrapped_phase, coherence_map
    ):
        """Test that unwrapping preserves array shape."""
        wrapped_t = unwrapper.dm.to_tensor(wrapped_phase)
        coh_t = unwrapper.dm.to_tensor(coherence_map)

        unwrapped_t = unwrapper.unwrap(wrapped_t, coherence=coh_t)

        assert unwrapped_t.shape == wrapped_t.shape

    def test_similar_to_dct_on_clean_data(
        self, device_manager, simple_phase, wrapped_phase
    ):
        """Test that IRLS-CG gives similar results to DCT on clean data."""
        dct_unwrapper = DCTUnwrapper(device_manager)
        cg_unwrapper = IRLSCGUnwrapper(device_manager, max_irls_iterations=5)

        wrapped_t = dct_unwrapper.dm.to_tensor(wrapped_phase)

        result_dct = dct_unwrapper(wrapped_phase)
        result_cg = cg_unwrapper(wrapped_phase)

        # Results should be similar (both solve least squares)
        # Normalize to same mean for comparison
        result_dct_n = result_dct - np.mean(result_dct)
        result_cg_n = result_cg - np.mean(result_cg)

        # High correlation expected
        corr = np.corrcoef(result_dct_n.flatten(), result_cg_n.flatten())[0, 1]
        assert corr > 0.99

    def test_callable_interface(self, unwrapper, wrapped_phase, coherence_map):
        """Test the __call__ interface with numpy arrays."""
        unwrapped = unwrapper(wrapped_phase, coherence=coherence_map)

        assert isinstance(unwrapped, np.ndarray)
        assert unwrapped.shape == wrapped_phase.shape

    def test_custom_parameters(self, device_manager, wrapped_phase):
        """Test custom IRLS-CG parameters."""
        unwrapper = IRLSCGUnwrapper(
            device_manager,
            max_irls_iterations=5,
            max_cg_iterations=50,
            irls_tolerance=1e-3,
            cg_tolerance=1e-5,
            delta=0.5,
        )

        unwrapped = unwrapper(wrapped_phase)
        assert unwrapped.shape == wrapped_phase.shape

    def test_clear_cache(self, unwrapper, wrapped_phase):
        """Test cache clearing."""
        unwrapper(wrapped_phase)
        # Should not raise
        unwrapper.clear_cache()

    def test_finite_output(self, unwrapper, wrapped_phase, coherence_map):
        """Test that output contains no NaN or Inf values."""
        wrapped_t = unwrapper.dm.to_tensor(wrapped_phase)
        coh_t = unwrapper.dm.to_tensor(coherence_map)

        unwrapped_t = unwrapper.unwrap(wrapped_t, coherence=coh_t)

        assert torch.isfinite(unwrapped_t).all()


class TestIRLSCGAPI:
    """Tests for IRLS-CG via the API."""

    def test_unwrap_irls_cg_function(self, wrapped_phase, coherence_map):
        """Test the unwrap_irls_cg convenience function."""
        import rapidphase

        unw, conncomp = rapidphase.unwrap_irls_cg(
            np.exp(1j * wrapped_phase),
            coherence_map,
            device="cpu",
        )

        assert unw.shape == wrapped_phase.shape
        assert conncomp.shape == wrapped_phase.shape

    def test_unwrap_with_algorithm_irls_cg(self, wrapped_phase, coherence_map):
        """Test unwrap() with algorithm='irls_cg'."""
        import rapidphase

        unw, conncomp = rapidphase.unwrap(
            np.exp(1j * wrapped_phase),
            corr=coherence_map,
            algorithm="irls_cg",
            device="cpu",
        )

        assert unw.shape == wrapped_phase.shape
