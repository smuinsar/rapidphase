"""Tests for phase operation utilities."""

import math

import numpy as np
import pytest
import torch

from rapidphase.utils.phase_ops import (
    wrap,
    rewrap,
    gradient,
    gradient_full,
    laplacian,
    compute_residues,
    integrate_phase,
)


class TestWrap:
    """Tests for wrap() function."""

    def test_wrap_within_range(self):
        """Test that values already in [-pi, pi] are unchanged."""
        phase = torch.tensor([0.0, 1.0, -1.0, math.pi - 0.1, -math.pi + 0.1])
        wrapped = wrap(phase)

        torch.testing.assert_close(wrapped, phase, rtol=1e-5, atol=1e-5)

    def test_wrap_positive_overflow(self):
        """Test wrapping values greater than pi."""
        phase = torch.tensor([math.pi + 0.5, 2 * math.pi, 3 * math.pi])
        wrapped = wrap(phase)

        # All should be in [-pi, pi]
        assert torch.all(wrapped >= -math.pi)
        assert torch.all(wrapped <= math.pi)

    def test_wrap_negative_overflow(self):
        """Test wrapping values less than -pi."""
        phase = torch.tensor([-math.pi - 0.5, -2 * math.pi, -3 * math.pi])
        wrapped = wrap(phase)

        # All should be in [-pi, pi]
        assert torch.all(wrapped >= -math.pi)
        assert torch.all(wrapped <= math.pi)

    def test_wrap_preserves_shape(self):
        """Test that wrapping preserves array shape."""
        phase = torch.randn(10, 20) * 10
        wrapped = wrap(phase)

        assert wrapped.shape == phase.shape

    def test_wrap_2d_array(self):
        """Test wrapping a 2D array."""
        phase = torch.tensor([[0.0, 4.0], [-4.0, 8.0]])
        wrapped = wrap(phase)

        assert wrapped.shape == (2, 2)
        assert torch.all(wrapped >= -math.pi)
        assert torch.all(wrapped <= math.pi)


class TestRewrap:
    """Tests for rewrap() function."""

    def test_rewrap_equals_wrap(self):
        """Test that rewrap gives same result as wrap."""
        phase = torch.randn(32, 32) * 5
        wrapped = wrap(phase)
        rewrapped = rewrap(phase)

        torch.testing.assert_close(wrapped, rewrapped)


class TestGradient:
    """Tests for gradient() function."""

    def test_gradient_constant_is_zero(self):
        """Test that gradient of constant is zero."""
        phase = torch.ones(10, 10) * 0.5
        grad_x, grad_y = gradient(phase)

        torch.testing.assert_close(grad_x, torch.zeros(10, 9))
        torch.testing.assert_close(grad_y, torch.zeros(9, 10))

    def test_gradient_linear_ramp_x(self):
        """Test gradient of linear ramp in x direction."""
        # Linear ramp with slope 0.1 in x direction
        x = torch.arange(10).float() * 0.1
        phase = x.unsqueeze(0).expand(10, 10)

        grad_x, grad_y = gradient(phase, wrap_result=False)

        # grad_x should be ~0.1 everywhere
        expected = torch.ones(10, 9) * 0.1
        torch.testing.assert_close(grad_x, expected, atol=1e-5, rtol=1e-5)

        # grad_y should be ~0
        torch.testing.assert_close(grad_y, torch.zeros(9, 10), atol=1e-5, rtol=1e-5)

    def test_gradient_output_shape(self):
        """Test gradient output shapes."""
        phase = torch.randn(20, 30)
        grad_x, grad_y = gradient(phase)

        assert grad_x.shape == (20, 29)
        assert grad_y.shape == (19, 30)

    def test_gradient_wrapping(self):
        """Test that gradient wraps large differences."""
        # Create phase with jump > pi
        phase = torch.zeros(5, 5)
        phase[:, 2:] = 2 * math.pi  # Jump of 2*pi

        grad_x, grad_y = gradient(phase, wrap_result=True)

        # Wrapped gradient should be near zero (2*pi wraps to 0)
        assert torch.all(torch.abs(grad_x) < 0.1)


class TestGradientFull:
    """Tests for gradient_full() function."""

    def test_gradient_full_same_shape(self):
        """Test that gradient_full returns same shape as input."""
        phase = torch.randn(20, 30)
        grad_x, grad_y = gradient_full(phase)

        assert grad_x.shape == phase.shape
        assert grad_y.shape == phase.shape

    def test_gradient_full_boundary_zeros(self):
        """Test that boundaries have zero gradient."""
        phase = torch.randn(10, 10)
        grad_x, grad_y = gradient_full(phase)

        # Last column of grad_x should be zero
        torch.testing.assert_close(grad_x[:, -1], torch.zeros(10))

        # Last row of grad_y should be zero
        torch.testing.assert_close(grad_y[-1, :], torch.zeros(10))


class TestLaplacian:
    """Tests for laplacian() function."""

    def test_laplacian_constant_is_zero(self):
        """Test that Laplacian of constant is zero."""
        phase = torch.ones(20, 20) * 0.5
        lap = laplacian(phase)

        # Should be nearly zero everywhere
        assert torch.allclose(lap, torch.zeros_like(lap), atol=1e-5)

    def test_laplacian_linear_is_zero(self):
        """Test that Laplacian of linear function is zero."""
        y, x = torch.meshgrid(
            torch.arange(20).float(),
            torch.arange(20).float(),
            indexing='ij'
        )
        phase = 0.1 * x + 0.2 * y

        lap = laplacian(phase)

        # Laplacian of linear function is zero (interior only)
        interior = lap[2:-2, 2:-2]
        assert torch.allclose(interior, torch.zeros_like(interior), atol=0.1)

    def test_laplacian_output_shape(self):
        """Test Laplacian output shape."""
        phase = torch.randn(30, 40)
        lap = laplacian(phase)

        assert lap.shape == phase.shape


class TestComputeResidues:
    """Tests for compute_residues() function."""

    def test_residues_smooth_phase(self):
        """Test that smooth phase has no residues."""
        # Smooth phase ramp (unwrapped)
        y, x = torch.meshgrid(
            torch.arange(20).float(),
            torch.arange(20).float(),
            indexing='ij'
        )
        phase = 0.1 * (x + y)
        wrapped = wrap(phase)

        residues = compute_residues(wrapped)

        # Should be nearly zero everywhere (no discontinuities)
        assert torch.allclose(residues, torch.zeros_like(residues), atol=0.1)

    def test_residues_output_shape(self):
        """Test residue output shape."""
        phase = torch.randn(30, 40)
        phase = wrap(phase)
        residues = compute_residues(phase)

        assert residues.shape == (29, 39)


class TestIntegratePhase:
    """Tests for integrate_phase() function."""

    def test_integrate_zero_gradients(self):
        """Test integration of zero gradients gives constant."""
        grad_x = torch.zeros(10, 9)
        grad_y = torch.zeros(9, 10)

        phase = integrate_phase(grad_x, grad_y, start_value=1.0)

        # Should be constant 1.0 everywhere
        torch.testing.assert_close(phase, torch.ones(10, 10))

    def test_integrate_output_shape(self):
        """Test integration output shape."""
        grad_x = torch.randn(20, 29)
        grad_y = torch.randn(19, 30)

        phase = integrate_phase(grad_x, grad_y)

        assert phase.shape == (20, 30)

    def test_integrate_recovers_linear_ramp(self):
        """Test that integration can recover a linear ramp."""
        # Create linear ramp
        y, x = torch.meshgrid(
            torch.arange(10).float(),
            torch.arange(10).float(),
            indexing='ij'
        )
        original = 0.1 * x

        # Compute gradients (not wrapped)
        grad_x, grad_y = gradient(original, wrap_result=False)

        # Integrate back
        recovered = integrate_phase(grad_x, grad_y, start_value=0.0)

        # Should match original
        torch.testing.assert_close(recovered, original, atol=1e-4, rtol=1e-4)
