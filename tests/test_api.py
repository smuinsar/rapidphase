"""Tests for the public API."""

import numpy as np
import pytest

import rapidphase
from rapidphase.api import unwrap, unwrap_dct, unwrap_irls, get_available_devices


class TestUnwrapAPI:
    """Tests for the main unwrap() function."""

    def test_unwrap_complex_interferogram(self, complex_interferogram):
        """Test unwrapping a complex interferogram."""
        unw, conncomp = unwrap(complex_interferogram, device="cpu")

        assert unw.shape == complex_interferogram.shape
        assert conncomp.shape == complex_interferogram.shape
        assert conncomp.dtype == np.int32

    def test_unwrap_wrapped_phase(self, wrapped_phase):
        """Test unwrapping already-wrapped phase."""
        unw, conncomp = unwrap(wrapped_phase, device="cpu")

        assert unw.shape == wrapped_phase.shape
        assert isinstance(unw, np.ndarray)

    def test_unwrap_with_coherence(self, wrapped_phase, coherence_map):
        """Test unwrapping with coherence (auto selects IRLS)."""
        unw, conncomp = unwrap(
            wrapped_phase,
            corr=coherence_map,
            device="cpu",
        )

        assert unw.shape == wrapped_phase.shape

    def test_unwrap_dct_algorithm(self, wrapped_phase):
        """Test explicit DCT algorithm selection."""
        unw, conncomp = unwrap(
            wrapped_phase,
            algorithm="dct",
            device="cpu",
        )

        assert unw.shape == wrapped_phase.shape

    def test_unwrap_irls_algorithm(self, wrapped_phase, coherence_map):
        """Test explicit IRLS algorithm selection."""
        unw, conncomp = unwrap(
            wrapped_phase,
            corr=coherence_map,
            algorithm="irls",
            device="cpu",
        )

        assert unw.shape == wrapped_phase.shape

    def test_unwrap_auto_algorithm_no_coherence(self, wrapped_phase):
        """Test auto algorithm without coherence selects DCT."""
        # Should use DCT (fast) when no coherence
        unw1, _ = unwrap(wrapped_phase, algorithm="auto", device="cpu")
        unw2, _ = unwrap(wrapped_phase, algorithm="dct", device="cpu")

        np.testing.assert_array_equal(unw1, unw2)

    def test_unwrap_auto_algorithm_with_coherence(self, wrapped_phase, coherence_map):
        """Test auto algorithm with coherence selects IRLS."""
        # Should use IRLS when coherence provided
        unw, _ = unwrap(
            wrapped_phase,
            corr=coherence_map,
            algorithm="auto",
            device="cpu",
        )

        assert unw.shape == wrapped_phase.shape

    def test_unwrap_invalid_algorithm(self, wrapped_phase):
        """Test that invalid algorithm raises error."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            unwrap(wrapped_phase, algorithm="invalid", device="cpu")

    def test_unwrap_with_tiling(self, wrapped_phase, coherence_map):
        """Test unwrapping with tiling for large images."""
        unw, conncomp = unwrap(
            wrapped_phase,
            corr=coherence_map,
            ntiles=(2, 2),
            tile_overlap=16,
            device="cpu",
        )

        assert unw.shape == wrapped_phase.shape

    def test_unwrap_nlooks_parameter(self, wrapped_phase, coherence_map):
        """Test nlooks parameter for IRLS."""
        unw1, _ = unwrap(
            wrapped_phase,
            corr=coherence_map,
            nlooks=1.0,
            device="cpu",
        )
        unw2, _ = unwrap(
            wrapped_phase,
            corr=coherence_map,
            nlooks=10.0,
            device="cpu",
        )

        # Both should produce valid results
        assert unw1.shape == wrapped_phase.shape
        assert unw2.shape == wrapped_phase.shape


class TestUnwrapDCT:
    """Tests for unwrap_dct() convenience function."""

    def test_unwrap_dct_basic(self, wrapped_phase):
        """Test basic DCT unwrapping."""
        unw, conncomp = unwrap_dct(wrapped_phase, device="cpu")

        assert unw.shape == wrapped_phase.shape

    def test_unwrap_dct_complex(self, complex_interferogram):
        """Test DCT unwrapping with complex input."""
        unw, conncomp = unwrap_dct(complex_interferogram, device="cpu")

        assert unw.shape == complex_interferogram.shape

    def test_unwrap_dct_with_tiling(self, wrapped_phase):
        """Test DCT unwrapping with tiling."""
        unw, conncomp = unwrap_dct(
            wrapped_phase,
            ntiles=(2, 2),
            tile_overlap=8,
            device="cpu",
        )

        assert unw.shape == wrapped_phase.shape


class TestUnwrapIRLS:
    """Tests for unwrap_irls() convenience function."""

    def test_unwrap_irls_basic(self, wrapped_phase, coherence_map):
        """Test basic IRLS unwrapping."""
        unw, conncomp = unwrap_irls(
            wrapped_phase,
            coherence_map,
            device="cpu",
        )

        assert unw.shape == wrapped_phase.shape

    def test_unwrap_irls_custom_params(self, wrapped_phase, coherence_map):
        """Test IRLS with custom parameters."""
        unw, conncomp = unwrap_irls(
            wrapped_phase,
            coherence_map,
            nlooks=5.0,
            max_iterations=30,
            tolerance=1e-3,
            device="cpu",
        )

        assert unw.shape == wrapped_phase.shape


class TestGetAvailableDevices:
    """Tests for get_available_devices() function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        devices = get_available_devices()

        assert isinstance(devices, dict)

    def test_cpu_always_available(self):
        """Test that CPU is always listed as available."""
        devices = get_available_devices()

        assert devices["cpu"] is True

    def test_contains_all_keys(self):
        """Test that all expected keys are present."""
        devices = get_available_devices()

        assert "cpu" in devices
        assert "cuda" in devices
        assert "mps" in devices
        assert "cuda_devices" in devices


class TestModuleExports:
    """Test that the module exports the expected API."""

    def test_unwrap_exported(self):
        """Test that unwrap is exported from rapidphase."""
        assert hasattr(rapidphase, "unwrap")
        assert callable(rapidphase.unwrap)

    def test_unwrap_dct_exported(self):
        """Test that unwrap_dct is exported."""
        assert hasattr(rapidphase, "unwrap_dct")
        assert callable(rapidphase.unwrap_dct)

    def test_unwrap_irls_exported(self):
        """Test that unwrap_irls is exported."""
        assert hasattr(rapidphase, "unwrap_irls")
        assert callable(rapidphase.unwrap_irls)

    def test_get_available_devices_exported(self):
        """Test that get_available_devices is exported."""
        assert hasattr(rapidphase, "get_available_devices")
        assert callable(rapidphase.get_available_devices)

    def test_device_manager_exported(self):
        """Test that DeviceManager is exported."""
        assert hasattr(rapidphase, "DeviceManager")

    def test_version_defined(self):
        """Test that version is defined."""
        assert hasattr(rapidphase, "__version__")
        assert isinstance(rapidphase.__version__, str)
