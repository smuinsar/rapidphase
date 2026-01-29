"""
DCT-based least squares phase unwrapping.

This solver uses the Discrete Cosine Transform to solve the Poisson
equation for phase unwrapping. It's the fastest algorithm and works
well for most data with reasonable coherence.

The algorithm:
1. Compute the wrapped Laplacian of the phase
2. Transform to frequency domain using 2D DCT
3. Solve Poisson equation by division (embarrassingly parallel)
4. Transform back using inverse DCT

All operations are GPU-parallelizable via PyTorch.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from rapidphase.core.base import BaseUnwrapper
from rapidphase.utils.phase_ops import laplacian

if TYPE_CHECKING:
    from rapidphase.device.manager import DeviceManager

# Try to import scipy for optimized DCT
try:
    from scipy.fft import dctn, idctn
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class DCTUnwrapper(BaseUnwrapper):
    """
    DCT-based least squares phase unwrapper.

    This is the fastest unwrapping algorithm, using FFT-based DCT
    to solve the Poisson equation in O(n log n) time with full
    GPU parallelization.

    The method minimizes ||∇φ - W(∇ψ)||² where φ is the unwrapped
    phase, ψ is the wrapped phase, and W() wraps to [-π, π].
    """

    def __init__(self, device_manager: DeviceManager):
        """
        Initialize the DCT unwrapper.

        Parameters
        ----------
        device_manager : DeviceManager
            Device manager for GPU/CPU operations.
        """
        super().__init__(device_manager)
        self._eigenvalue_cache: dict[tuple[int, int], torch.Tensor] = {}

    def unwrap(
        self,
        phase: torch.Tensor,
        coherence: torch.Tensor | None = None,
        nan_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Unwrap phase using DCT-based least squares.

        Parameters
        ----------
        phase : torch.Tensor
            Wrapped phase of shape (H, W), values in [-pi, pi].
        coherence : torch.Tensor, optional
            Coherence map (not used by DCT solver, use IRLS for weighted).
        nan_mask : torch.Tensor, optional
            Boolean mask where True indicates NaN/invalid pixels.
            If not provided, detected automatically from phase.
        **kwargs
            Unused, for API compatibility.

        Returns
        -------
        torch.Tensor
            Unwrapped phase of shape (H, W).

        Notes
        -----
        The coherence parameter is accepted but not used. For coherence-
        weighted unwrapping, use IRLSUnwrapper instead.

        NaN values in the input phase are handled by zeroing gradients
        that involve NaN pixels. The unwrapped result will have NaN
        restored at the original invalid pixel locations.
        """
        H, W = phase.shape

        # Detect NaN mask if not provided
        if nan_mask is None:
            nan_mask = torch.isnan(phase)

        has_nans = nan_mask.any()

        # Replace NaN with 0 for safe computation
        if has_nans:
            phase_clean = phase.clone()
            phase_clean[nan_mask] = 0.0
        else:
            phase_clean = phase

        # Step 1: Compute wrapped Laplacian (parallel on GPU)
        # Pass NaN mask to properly handle invalid pixels
        rho = laplacian(phase_clean, nan_mask=nan_mask)

        # Step 2: 2D DCT (use scipy for accuracy, PyTorch for GPU ops)
        rho_dct = self._dct2(rho)

        # Step 3: Get eigenvalues for Poisson solve
        eigenvalues = self._get_eigenvalues(H, W)

        # Step 4: Solve in frequency domain (embarrassingly parallel)
        phi_dct = rho_dct / eigenvalues
        # Set DC component (mean) to zero
        phi_dct[0, 0] = 0.0

        # Step 5: Inverse DCT
        unwrapped = self._idct2(phi_dct)

        # Restore NaN at invalid pixel locations
        if has_nans:
            unwrapped[nan_mask] = float('nan')

        return unwrapped

    def _get_eigenvalues(self, H: int, W: int) -> torch.Tensor:
        """
        Get or compute cached eigenvalues for Poisson equation.

        The eigenvalues of the discrete Laplacian under DCT-II are:
        λ[i,j] = 2*cos(πi/H) + 2*cos(πj/W) - 4

        Parameters
        ----------
        H, W : int
            Dimensions of the phase array.

        Returns
        -------
        torch.Tensor
            Eigenvalue array of shape (H, W).
        """
        key = (H, W)
        if key not in self._eigenvalue_cache:
            # Compute eigenvalues
            i = torch.arange(H, dtype=self.dtype, device=self.device)
            j = torch.arange(W, dtype=self.dtype, device=self.device)

            # 2D eigenvalues via outer product
            lambda_i = 2 * torch.cos(math.pi * i / H) - 2
            lambda_j = 2 * torch.cos(math.pi * j / W) - 2

            eigenvalues = lambda_i.unsqueeze(1) + lambda_j.unsqueeze(0)

            # Avoid division by zero at DC component and near-zero eigenvalues
            # (can occur with float32 precision on large images)
            eigenvalues[0, 0] = 1.0  # Will be zeroed out anyway
            # Clamp small eigenvalues to avoid numerical instability
            min_eigenvalue = 1e-6
            eigenvalues = torch.where(
                torch.abs(eigenvalues) < min_eigenvalue,
                torch.sign(eigenvalues) * min_eigenvalue,
                eigenvalues,
            )
            # Handle exactly zero case (sign would be 0)
            eigenvalues = torch.where(
                eigenvalues == 0,
                torch.tensor(min_eigenvalue, dtype=self.dtype, device=self.device),
                eigenvalues,
            )

            self._eigenvalue_cache[key] = eigenvalues

        return self._eigenvalue_cache[key]

    def _dct2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D Type-II DCT.

        Uses scipy for accuracy when available, otherwise falls back
        to PyTorch FFT-based implementation.

        Parameters
        ----------
        x : torch.Tensor
            Input array of shape (H, W).

        Returns
        -------
        torch.Tensor
            DCT coefficients of shape (H, W).
        """
        if HAS_SCIPY:
            # Use scipy DCT (most accurate, CPU only)
            x_np = self.dm.to_numpy(x)
            result_np = dctn(x_np, type=2, norm='ortho')
            return self.dm.to_tensor(result_np)
        else:
            # Fallback to PyTorch implementation
            return self._dct_pytorch(self._dct_pytorch(x, dim=1), dim=0)

    def _idct2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D inverse Type-II DCT.

        Parameters
        ----------
        x : torch.Tensor
            DCT coefficients of shape (H, W).

        Returns
        -------
        torch.Tensor
            Reconstructed array of shape (H, W).
        """
        if HAS_SCIPY:
            # Use scipy IDCT (most accurate, CPU only)
            x_np = self.dm.to_numpy(x)
            result_np = idctn(x_np, type=2, norm='ortho')
            return self.dm.to_tensor(result_np)
        else:
            # Fallback to PyTorch implementation
            return self._idct_pytorch(self._idct_pytorch(x, dim=0), dim=1)

    def _dct_pytorch(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute 1D Type-II DCT using PyTorch FFT.

        Orthonormal DCT-II implementation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        dim : int
            Dimension along which to compute DCT.

        Returns
        -------
        torch.Tensor
            DCT coefficients.
        """
        N = x.shape[dim]
        x = x.movedim(dim, -1)

        # Reorder: even indices first, then odd indices reversed
        # v[k] = x[2k] for k < ceil(N/2)
        # v[k] = x[2N-2k-1] for k >= ceil(N/2)
        v = torch.empty_like(x)
        v[..., :(N + 1) // 2] = x[..., 0::2]
        if N > 1:
            v[..., (N + 1) // 2:] = x[..., (N - 1 - (N % 2))::-2]

        # FFT
        Vc = torch.fft.fft(v, dim=-1)

        # Phase factor
        k = torch.arange(N, dtype=self.dtype, device=self.device)
        factor = 2.0 * torch.exp(-1j * math.pi * k / (2 * N))

        # Orthonormal scaling
        scale = math.sqrt(1.0 / (2 * N))
        factor = factor * scale
        factor[0] = factor[0] / math.sqrt(2)

        result = (Vc * factor).real

        return result.movedim(-1, dim)

    def _idct_pytorch(self, X: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute 1D inverse Type-II DCT using PyTorch FFT.

        Orthonormal IDCT-II implementation.

        Parameters
        ----------
        X : torch.Tensor
            DCT coefficients.
        dim : int
            Dimension along which to compute IDCT.

        Returns
        -------
        torch.Tensor
            Reconstructed signal.
        """
        N = X.shape[dim]
        X = X.movedim(dim, -1)

        # Undo orthonormal scaling
        scale = math.sqrt(2 * N)
        X_scaled = X * scale
        X_scaled[..., 0] = X_scaled[..., 0] / math.sqrt(2)

        # Phase factor for DCT-III
        k = torch.arange(N, dtype=self.dtype, device=self.device)
        phase = torch.exp(1j * math.pi * k / (2 * N)) / 2.0
        phase[0] = phase[0] * 2

        X_complex = X_scaled * phase

        # Hermitian extension for real output
        complex_dtype = torch.complex128 if self.dtype == torch.float64 else torch.complex64
        X_full = torch.zeros(
            X.shape[:-1] + (N,),
            dtype=complex_dtype,
            device=self.device,
        )
        X_full[...] = X_complex

        # IFFT
        v = torch.fft.ifft(X_full, dim=-1).real * N

        # Reorder back
        x = torch.empty_like(v)
        x[..., 0::2] = v[..., :(N + 1) // 2]
        if N > 1:
            x[..., 1::2] = v[..., (N + 1) // 2:].flip(dims=[-1])

        return x.movedim(-1, dim)

    def clear_cache(self) -> None:
        """Clear the eigenvalue cache."""
        self._eigenvalue_cache.clear()
