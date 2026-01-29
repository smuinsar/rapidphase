"""
IRLS phase unwrapper with Conjugate Gradient solver.

This implements the IRLS algorithm using Conjugate Gradient (CG) for the
inner least-squares solve, which converges faster than Jacobi iteration.

The algorithm minimizes an approximation to the L1 norm:
    min Σ |∇φ - W(∇ψ)|

via iteratively reweighted least squares:
    min Σ w_ij * |∇φ_ij - W(∇ψ_ij)|²

where weights w_ij = 1/max(|residual_ij|, δ) are updated each iteration.

Reference: Based on the approach in bpauld/PhaseUnwrapping, simplified for
GPU efficiency while preserving the mathematical foundation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rapidphase.core.base import BaseUnwrapper
from rapidphase.core.dct_solver import DCTUnwrapper
from rapidphase.utils.phase_ops import wrap, gradient_full, laplacian
from rapidphase.utils.quality import coherence_to_weights

if TYPE_CHECKING:
    from rapidphase.device.manager import DeviceManager


class IRLSCGUnwrapper(BaseUnwrapper):
    """
    IRLS phase unwrapper using Conjugate Gradient solver.

    This provides faster convergence than the Jacobi-based IRLS by using
    CG to solve the weighted least-squares problem at each iteration.

    The method approximates L1-norm minimization:
        min Σ C_ij * |∇φ_ij - W(∇ψ_ij)|

    where C_ij are coherence-derived weights.
    """

    def __init__(
        self,
        device_manager: DeviceManager,
        max_irls_iterations: int = 20,
        max_cg_iterations: int = 100,
        irls_tolerance: float = 1e-4,
        cg_tolerance: float = 1e-6,
        delta: float = 0.1,
        nlooks: float = 1.0,
    ):
        """
        Initialize the IRLS-CG unwrapper.

        Parameters
        ----------
        device_manager : DeviceManager
            Device manager for GPU/CPU operations.
        max_irls_iterations : int
            Maximum number of outer IRLS iterations.
        max_cg_iterations : int
            Maximum CG iterations per IRLS step.
        irls_tolerance : float
            Convergence tolerance for IRLS (relative change).
        cg_tolerance : float
            Convergence tolerance for CG (relative residual).
        delta : float
            Smoothing parameter for IRLS weights. Smaller values
            approximate L1 more closely but may cause instability.
        nlooks : float
            Number of looks for coherence-to-weight conversion.
        """
        super().__init__(device_manager)
        self.max_irls_iterations = max_irls_iterations
        self.max_cg_iterations = max_cg_iterations
        self.irls_tolerance = irls_tolerance
        self.cg_tolerance = cg_tolerance
        self.delta = delta
        self.nlooks = nlooks

        # DCT solver for initialization
        self._dct_solver = DCTUnwrapper(device_manager)

    def unwrap(
        self,
        phase: torch.Tensor,
        coherence: torch.Tensor | None = None,
        nan_mask: torch.Tensor | None = None,
        max_irls_iterations: int | None = None,
        max_cg_iterations: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Unwrap phase using IRLS with CG solver.

        Parameters
        ----------
        phase : torch.Tensor
            Wrapped phase of shape (H, W), values in [-pi, pi].
        coherence : torch.Tensor, optional
            Coherence map of shape (H, W), values in [0, 1].
            If None, uses uniform weights (pure L1 approximation).
        nan_mask : torch.Tensor, optional
            Boolean mask where True indicates NaN/invalid pixels.
            If not provided, detected automatically from phase.
        max_irls_iterations : int, optional
            Override default max IRLS iterations.
        max_cg_iterations : int, optional
            Override default max CG iterations.

        Returns
        -------
        torch.Tensor
            Unwrapped phase of shape (H, W).
        """
        max_irls = max_irls_iterations or self.max_irls_iterations
        max_cg = max_cg_iterations or self.max_cg_iterations

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

        # Compute wrapped gradients (target)
        grad_x_target, grad_y_target = gradient_full(
            phase_clean, wrap_result=True, nan_mask=nan_mask
        )

        # Coherence-based weights (static throughout IRLS)
        if coherence is not None:
            coh_weights = coherence_to_weights(coherence, nlooks=self.nlooks)
        else:
            coh_weights = self.dm.ones((H, W))

        # Set weights to 0 for NaN pixels
        if has_nans:
            coh_weights[nan_mask] = 0.0

        # Initialize with DCT solution
        phi = self._dct_solver.unwrap(phase_clean, nan_mask=nan_mask)

        # Keep NaN positions zeroed during iterations (DCT restores NaN, we need to zero them)
        if has_nans:
            phi[nan_mask] = 0.0

        # IRLS iterations
        for irls_iter in range(max_irls):
            phi_old = phi.clone()

            # Compute current gradient residuals
            grad_x_phi, grad_y_phi = gradient_full(phi, wrap_result=False, nan_mask=nan_mask)
            res_x = grad_x_phi - grad_x_target
            res_y = grad_y_phi - grad_y_target

            # Update IRLS weights: w = C / max(|residual|, delta)
            residual_mag = torch.sqrt(res_x ** 2 + res_y ** 2 + 1e-10)
            irls_weights = coh_weights / torch.clamp(residual_mag, min=self.delta)

            # Normalize weights for numerical stability
            irls_weights = irls_weights / (irls_weights.max() + 1e-10)

            # Set weights to 0 for NaN pixels
            if has_nans:
                irls_weights[nan_mask] = 0.0

            # Solve weighted least squares using CG
            phi = self._cg_solve(
                phi,
                grad_x_target,
                grad_y_target,
                irls_weights,
                max_cg,
            )

            # Keep NaN positions zeroed during iterations
            if has_nans:
                phi[nan_mask] = 0.0

            # Check IRLS convergence (exclude NaN pixels)
            if has_nans:
                valid_mask = ~nan_mask
                diff_norm = torch.norm((phi - phi_old)[valid_mask])
                old_norm = torch.norm(phi_old[valid_mask])
            else:
                diff_norm = torch.norm(phi - phi_old)
                old_norm = torch.norm(phi_old)

            rel_change = diff_norm / (old_norm + 1e-10)
            if rel_change < self.irls_tolerance:
                break

        # Restore NaN at invalid pixel locations
        if has_nans:
            phi[nan_mask] = float('nan')

        return phi

    def _cg_solve(
        self,
        phi_init: torch.Tensor,
        grad_x_target: torch.Tensor,
        grad_y_target: torch.Tensor,
        weights: torch.Tensor,
        max_iterations: int,
    ) -> torch.Tensor:
        """
        Solve weighted least squares using Conjugate Gradient.

        Solves: min Σ w_ij * |∇φ_ij - g_ij|²

        This is equivalent to solving the weighted Poisson equation:
            div(w * ∇φ) = div(w * g)

        Parameters
        ----------
        phi_init : torch.Tensor
            Initial guess for φ.
        grad_x_target, grad_y_target : torch.Tensor
            Target gradients.
        weights : torch.Tensor
            Pixel weights.
        max_iterations : int
            Maximum CG iterations.

        Returns
        -------
        torch.Tensor
            Solution φ.
        """
        phi = phi_init.clone()

        # Compute RHS: div(w * g)
        # Using backward difference for divergence (adjoint of forward gradient)
        rhs = self._weighted_divergence(grad_x_target, grad_y_target, weights)

        # Initial residual: r = b - A*x
        Aphi = self._weighted_laplacian(phi, weights)
        r = rhs - Aphi

        # Initial search direction
        p = r.clone()

        # Initial residual norm squared
        r_norm_sq = torch.sum(r * r)
        r0_norm_sq = r_norm_sq.clone()

        for cg_iter in range(max_iterations):
            # Matrix-vector product: A*p
            Ap = self._weighted_laplacian(p, weights)

            # Step size
            pAp = torch.sum(p * Ap)
            if pAp.abs() < 1e-12:
                break
            alpha = r_norm_sq / pAp

            # Update solution and residual
            phi = phi + alpha * p
            r = r - alpha * Ap

            # New residual norm squared
            r_norm_sq_new = torch.sum(r * r)

            # Check convergence
            if r_norm_sq_new < self.cg_tolerance ** 2 * r0_norm_sq:
                break

            # Update search direction
            beta = r_norm_sq_new / r_norm_sq
            p = r + beta * p

            r_norm_sq = r_norm_sq_new

        return phi

    def _weighted_laplacian(
        self,
        phi: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted Laplacian: -div(w * ∇φ).

        This is the operator A in the linear system A*φ = b.
        """
        H, W = phi.shape

        # Forward gradients
        grad_x = torch.zeros_like(phi)
        grad_y = torch.zeros_like(phi)
        grad_x[:, :-1] = phi[:, 1:] - phi[:, :-1]
        grad_y[:-1, :] = phi[1:, :] - phi[:-1, :]

        # Weight the gradients
        wgrad_x = weights * grad_x
        wgrad_y = weights * grad_y

        # Backward divergence (negative adjoint of forward gradient)
        div = torch.zeros_like(phi)
        # x-component
        div[:, 1:] += wgrad_x[:, :-1]
        div[:, :-1] -= wgrad_x[:, :-1]
        # y-component
        div[1:, :] += wgrad_y[:-1, :]
        div[:-1, :] -= wgrad_y[:-1, :]

        return -div

    def _weighted_divergence(
        self,
        grad_x: torch.Tensor,
        grad_y: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted divergence: div(w * g).

        This is the RHS b in the linear system.
        """
        H, W = grad_x.shape

        # Weight the gradients
        wgrad_x = weights * grad_x
        wgrad_y = weights * grad_y

        # Backward divergence
        div = torch.zeros_like(grad_x)
        # x-component
        div[:, 1:] += wgrad_x[:, :-1]
        div[:, :-1] -= wgrad_x[:, :-1]
        # y-component
        div[1:, :] += wgrad_y[:-1, :]
        div[:-1, :] -= wgrad_y[:-1, :]

        return -div

    def clear_cache(self) -> None:
        """Clear internal caches."""
        self._dct_solver.clear_cache()
