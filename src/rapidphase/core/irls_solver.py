"""
Iteratively Reweighted Least Squares (IRLS) phase unwrapper.

This solver uses coherence-based weights to improve unwrapping quality
in low-coherence regions. It starts with the DCT solution and iteratively
refines it using weighted least squares.

The algorithm:
1. Initialize with DCT solution (unweighted)
2. Compute weights from coherence and residuals
3. Apply weighted smoothing to refine solution
4. Repeat until convergence

All operations are GPU-parallelizable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rapidphase.core.base import BaseUnwrapper
from rapidphase.core.dct_solver import DCTUnwrapper
from rapidphase.utils.phase_ops import wrap, gradient_full
from rapidphase.utils.quality import coherence_to_weights

if TYPE_CHECKING:
    from rapidphase.device.manager import DeviceManager


class IRLSUnwrapper(BaseUnwrapper):
    """
    Iteratively Reweighted Least Squares phase unwrapper.

    This algorithm provides coherence-weighted phase unwrapping,
    which is more robust to noise and low-coherence regions than
    the unweighted DCT method.

    The method minimizes Σ w[i,j] * ||∇φ[i,j] - W(∇ψ[i,j])||²
    where w[i,j] are weights derived from coherence.
    """

    def __init__(
        self,
        device_manager: DeviceManager,
        max_iterations: int = 50,
        tolerance: float = 1e-4,
        nlooks: float = 1.0,
    ):
        """
        Initialize the IRLS unwrapper.

        Parameters
        ----------
        device_manager : DeviceManager
            Device manager for GPU/CPU operations.
        max_iterations : int
            Maximum number of IRLS iterations.
        tolerance : float
            Convergence tolerance (relative change in solution).
        nlooks : float
            Number of looks for coherence-to-weight conversion.
        """
        super().__init__(device_manager)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.nlooks = nlooks

        # DCT solver for initialization
        self._dct_solver = DCTUnwrapper(device_manager)

    def unwrap(
        self,
        phase: torch.Tensor,
        coherence: torch.Tensor | None = None,
        nan_mask: torch.Tensor | None = None,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Unwrap phase using IRLS weighted least squares.

        Parameters
        ----------
        phase : torch.Tensor
            Wrapped phase of shape (H, W), values in [-pi, pi].
        coherence : torch.Tensor, optional
            Coherence map of shape (H, W), values in [0, 1].
            If None, falls back to unweighted DCT.
        nan_mask : torch.Tensor, optional
            Boolean mask where True indicates NaN/invalid pixels.
            If not provided, detected automatically from phase.
        max_iterations : int, optional
            Override default max iterations.
        tolerance : float, optional
            Override default convergence tolerance.
        **kwargs
            Unused, for API compatibility.

        Returns
        -------
        torch.Tensor
            Unwrapped phase of shape (H, W).
        """
        # Detect NaN mask if not provided
        if nan_mask is None:
            nan_mask = torch.isnan(phase)

        has_nans = nan_mask.any()

        # If no coherence provided, use unweighted DCT
        if coherence is None:
            return self._dct_solver.unwrap(phase, nan_mask=nan_mask)

        # Use defaults if not specified
        max_iter = max_iterations if max_iterations is not None else self.max_iterations
        tol = tolerance if tolerance is not None else self.tolerance

        H, W = phase.shape

        # Replace NaN with 0 for safe computation
        if has_nans:
            phase_clean = phase.clone()
            phase_clean[nan_mask] = 0.0
        else:
            phase_clean = phase

        # Initialize with DCT solution
        phi = self._dct_solver.unwrap(phase_clean, nan_mask=nan_mask)

        # Keep NaN positions zeroed during iterations (DCT restores NaN, we need to zero them)
        if has_nans:
            phi[nan_mask] = 0.0

        # Convert coherence to weights
        weights = coherence_to_weights(coherence, nlooks=self.nlooks)

        # Set weights to 0 for NaN pixels
        if has_nans:
            weights[nan_mask] = 0.0

        # Compute wrapped gradients of input phase
        grad_x_wrapped, grad_y_wrapped = gradient_full(
            phase_clean, wrap_result=True, nan_mask=nan_mask
        )

        # IRLS iterations: weighted Jacobi smoothing
        for iteration in range(max_iter):
            phi_old = phi.clone()

            # Update weights based on residuals
            weights = self._update_weights(
                phi, grad_x_wrapped, grad_y_wrapped, coherence, nan_mask
            )

            # Weighted Jacobi smoothing step
            phi = self._weighted_jacobi_step(
                phi, grad_x_wrapped, grad_y_wrapped, weights
            )

            # Keep NaN positions zeroed during iterations
            if has_nans:
                phi[nan_mask] = 0.0

            # Check convergence (exclude NaN pixels from norm computation)
            if has_nans:
                valid_mask = ~nan_mask
                diff_norm = torch.norm((phi - phi_old)[valid_mask])
                old_norm = torch.norm(phi_old[valid_mask])
            else:
                diff_norm = torch.norm(phi - phi_old)
                old_norm = torch.norm(phi_old)

            rel_change = diff_norm / (old_norm + 1e-10)
            if rel_change < tol:
                break

        # Restore NaN at invalid pixel locations
        if has_nans:
            phi[nan_mask] = float('nan')

        return phi

    def _update_weights(
        self,
        phi: torch.Tensor,
        grad_x_wrapped: torch.Tensor,
        grad_y_wrapped: torch.Tensor,
        coherence: torch.Tensor,
        nan_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Update weights based on residuals and coherence.

        Combines coherence-based weights with residual-based weights
        to down-weight pixels where unwrapping may have errors.
        """
        # Compute gradients of current solution
        grad_x_phi, grad_y_phi = gradient_full(phi, wrap_result=False, nan_mask=nan_mask)

        # Residuals: difference between unwrapped gradients and wrapped target
        res_x = grad_x_phi - grad_x_wrapped
        res_y = grad_y_phi - grad_y_wrapped

        # Residual magnitude
        residual_sq = res_x ** 2 + res_y ** 2

        # Residual-based weights (Huber-like)
        delta = 1.0
        residual_weights = torch.where(
            residual_sq < delta ** 2,
            torch.ones_like(residual_sq),
            delta / (torch.sqrt(residual_sq) + 1e-10),
        )

        # Combine with coherence weights
        coh_weights = coherence_to_weights(coherence, nlooks=self.nlooks)
        weights = coh_weights * residual_weights

        # Normalize and clamp
        weights = weights / (weights.max() + 1e-10)
        weights = torch.clamp(weights, min=1e-6)

        # Set weights to 0 for NaN pixels
        if nan_mask is not None and nan_mask.any():
            weights[nan_mask] = 0.0

        return weights

    def _weighted_jacobi_step(
        self,
        phi: torch.Tensor,
        grad_x_target: torch.Tensor,
        grad_y_target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a weighted Jacobi smoothing step.

        Updates phi to better match the target gradients, weighted by
        the reliability weights.
        """
        H, W = phi.shape

        # Pad phi for neighbor access
        phi_pad = torch.nn.functional.pad(
            phi.unsqueeze(0).unsqueeze(0),
            (1, 1, 1, 1),
            mode="replicate",
        ).squeeze()

        # Pad weights
        w_pad = torch.nn.functional.pad(
            weights.unsqueeze(0).unsqueeze(0),
            (1, 1, 1, 1),
            mode="replicate",
        ).squeeze()

        # Get neighbors
        phi_left = phi_pad[1:-1, :-2]
        phi_right = phi_pad[1:-1, 2:]
        phi_up = phi_pad[:-2, 1:-1]
        phi_down = phi_pad[2:, 1:-1]

        w_left = w_pad[1:-1, :-2]
        w_right = w_pad[1:-1, 2:]
        w_up = w_pad[:-2, 1:-1]
        w_down = w_pad[2:, 1:-1]

        # Target values from gradients
        # phi_right should equal phi + grad_x_target
        # phi_down should equal phi + grad_y_target
        target_from_left = phi_left + grad_x_target
        target_from_right = phi_right - grad_x_target
        target_from_up = phi_up + grad_y_target
        target_from_down = phi_down - grad_y_target

        # Weighted average of targets
        w_sum = w_left + w_right + w_up + w_down + 1e-10
        phi_new = (
            w_left * target_from_left
            + w_right * target_from_right
            + w_up * target_from_up
            + w_down * target_from_down
        ) / w_sum

        # Relaxation: blend with old value
        omega = 0.5  # Under-relaxation for stability
        phi_new = phi + omega * (phi_new - phi)

        return phi_new

    def clear_cache(self) -> None:
        """Clear internal caches."""
        self._dct_solver.clear_cache()
