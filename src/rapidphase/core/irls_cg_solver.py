"""
IRLS phase unwrapper with DCT-Preconditioned Conjugate Gradient solver.

This implements the IRLS algorithm using Preconditioned Conjugate Gradient (PCG)
for the inner least-squares solve. The DCT inverse of the unweighted Laplacian
serves as an effective preconditioner, reducing CG iterations from ~200 to ~10-20.

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

import math

import torch

from rapidphase.core.base import BaseUnwrapper
from rapidphase.core.dct_solver import DCTUnwrapper
from rapidphase.utils.phase_ops import wrap, gradient_full
from rapidphase.utils.quality import coherence_to_weights

if TYPE_CHECKING:
    from rapidphase.device.manager import DeviceManager


class IRLSCGUnwrapper(BaseUnwrapper):
    """
    IRLS phase unwrapper using DCT-Preconditioned Conjugate Gradient solver.

    This provides faster convergence than the Jacobi-based IRLS by using
    PCG to solve the weighted least-squares problem at each iteration.
    The DCT preconditioner dramatically reduces CG iteration count.

    The method approximates L1-norm minimization:
        min Σ C_ij * |∇φ_ij - W(∇ψ_ij)|

    where C_ij are coherence-derived weights.
    """

    def __init__(
        self,
        device_manager: DeviceManager,
        max_irls_iterations: int = 50,
        max_cg_iterations: int = 50,
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

        # DCT solver for initialization and preconditioning
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
        Unwrap phase using IRLS with DCT-Preconditioned CG solver.

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
        two_pi = 2 * math.pi

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

        # Set weights to 0 for NaN pixels AND NaN edges.
        # An edge (i,j)->(i,j+1) must have zero weight if either pixel is NaN.
        # This ensures the weighted Laplacian operator (A) and the RHS (b) are
        # consistent — otherwise CG diverges on large mostly-NaN tiles.
        if has_nans:
            coh_weights[nan_mask] = 0.0
            # Also precompute NaN edge masks for per-edge weight zeroing
            nan_edge_x = nan_mask.clone()
            nan_edge_x[:, :-1] = nan_edge_x[:, :-1] | nan_mask[:, 1:]
            nan_edge_y = nan_mask.clone()
            nan_edge_y[:-1, :] = nan_edge_y[:-1, :] | nan_mask[1:, :]

        # Initialize with DCT solution
        phi = self._dct_solver.unwrap(phase_clean, nan_mask=nan_mask)

        # Keep NaN positions zeroed during iterations
        if has_nans:
            phi[nan_mask] = 0.0

        # Adaptive delta schedule: start L2-like (large delta) for faster
        # early convergence, then decrease toward target for L1 approximation
        delta_schedule = [
            max(self.delta, 1.0),
            max(self.delta, 0.5),
            max(self.delta, 0.2),
        ]

        # IRLS iterations
        for irls_iter in range(max_irls):
            phi_old = phi.clone()

            # Select delta for this iteration
            if irls_iter < len(delta_schedule):
                current_delta = delta_schedule[irls_iter]
            else:
                current_delta = self.delta

            # Compute current gradient residuals
            grad_x_phi, grad_y_phi = gradient_full(phi, wrap_result=False, nan_mask=nan_mask)
            res_x = grad_x_phi - grad_x_target
            res_y = grad_y_phi - grad_y_target

            # Update IRLS weights per-edge: w = C / max(|residual|, delta)
            irls_weights_x = coh_weights / torch.clamp(torch.abs(res_x) + 1e-10, min=current_delta)
            irls_weights_y = coh_weights / torch.clamp(torch.abs(res_y) + 1e-10, min=current_delta)

            # Normalize weights for numerical stability
            max_w = max(irls_weights_x.max(), irls_weights_y.max()) + 1e-10
            irls_weights_x = irls_weights_x / max_w
            irls_weights_y = irls_weights_y / max_w

            # Set weights to 0 for NaN edges (either endpoint is NaN)
            if has_nans:
                irls_weights_x[nan_edge_x] = 0.0
                irls_weights_y[nan_edge_y] = 0.0

            # Solve weighted least squares using Preconditioned CG
            phi = self._cg_solve(
                phi,
                grad_x_target,
                grad_y_target,
                irls_weights_x,
                irls_weights_y,
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

        # Save continuous IRLS-CG solution for disambiguation
        phi_continuous = phi.clone()

        # Adjust DC offset before congruence projection to avoid ambiguous
        # rounding. The IRLS-CG solution has an arbitrary DC offset (Poisson
        # equation is defined up to a constant). If (phi - psi) / 2pi is near
        # a half-integer for many pixels, rounding will split them, creating
        # artificial discontinuities. Shift phi so the median fractional part
        # is centered at 0 (maximally far from the +/-0.5 rounding boundary).
        k_float = (phi - phase_clean) / two_pi
        frac = k_float - torch.round(k_float)  # fractional part in [-0.5, 0.5]
        if has_nans:
            valid_frac = frac[~nan_mask]
            dc_adjust = torch.median(valid_frac) * two_pi if valid_frac.numel() > 0 else 0.0
        else:
            dc_adjust = torch.median(frac) * two_pi
        phi = phi - dc_adjust
        phi_continuous = phi_continuous - dc_adjust

        # For tiled processing, return the continuous (pre-congruence) solution.
        # Congruence projection will be applied after merging all tiles to
        # avoid per-tile 2π transition inconsistencies at tile boundaries.
        if kwargs.get('return_continuous', False):
            if has_nans:
                phi_continuous[nan_mask] = float('nan')
            return phi_continuous

        # Project to nearest congruent solution:
        # unwrapped = wrapped + k * 2*pi, where k is the nearest integer
        k = torch.round((phi - phase_clean) / two_pi)
        phi = phase_clean + k * two_pi

        # Fix local consistency errors in the congruence projection.
        # Uses the continuous IRLS-CG solution to disambiguate expected
        # integer cycle differences between neighbors.
        phi = self._local_consistency_correction(
            phi, phase_clean, phi_continuous, coh_weights,
            nan_mask if has_nans else None
        )

        # Restore NaN at invalid pixel locations
        if has_nans:
            phi[nan_mask] = float('nan')

        return phi

    def _local_consistency_correction(
        self,
        phi: torch.Tensor,
        phase_clean: torch.Tensor,
        phi_continuous: torch.Tensor,
        coh_weights: torch.Tensor,
        nan_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Fix local 2pi consistency errors in the congruence projection.

        After congruence projection (k = round((phi_irls - psi) / 2pi)),
        some pixels may have k off by +/-1 due to rounding. This method
        detects and corrects such errors by checking each pixel's k against
        its neighbors.

        Uses the continuous IRLS-CG solution (phi_continuous) to compute
        unambiguous expected integer cycle differences, avoiding the
        rounding ambiguity in wrapped phase differences near +/-pi.

        Runs entirely on GPU with vectorized PyTorch operations.

        Parameters
        ----------
        phi : torch.Tensor
            Congruent unwrapped phase after projection (H, W).
        phase_clean : torch.Tensor
            Wrapped phase with NaN replaced by 0 (H, W).
        phi_continuous : torch.Tensor
            Pre-projection continuous IRLS-CG solution (H, W).
        coh_weights : torch.Tensor
            Coherence-derived weights (H, W), 0 at invalid pixels.
        nan_mask : torch.Tensor or None
            Boolean mask of invalid pixels.

        Returns
        -------
        torch.Tensor
            Corrected unwrapped phase.
        """
        two_pi = 2 * math.pi
        k = torch.round((phi - phase_clean) / two_pi)

        # Compute expected integer cycle differences from the continuous
        # IRLS-CG solution. Since phi_continuous is smooth and continuous,
        # its gradients give unambiguous integer offsets.
        raw_dx = phase_clean[:, 1:] - phase_clean[:, :-1]
        raw_dy = phase_clean[1:, :] - phase_clean[:-1, :]
        phi_dx = phi_continuous[:, 1:] - phi_continuous[:, :-1]
        phi_dy = phi_continuous[1:, :] - phi_continuous[:-1, :]

        # expected_dk = (continuous_gradient - wrapped_gradient) / 2pi
        # This is exact integer when phi_continuous is close to the true solution
        expected_dk_x = torch.round((phi_dx - raw_dx) / two_pi)
        expected_dk_y = torch.round((phi_dy - raw_dy) / two_pi)

        for _ in range(20):
            # Check horizontal consistency: k[i,j+1] - k[i,j] should equal expected_dk_x
            actual_dk_x = k[:, 1:] - k[:, :-1]
            actual_dk_y = k[1:, :] - k[:-1, :]

            err_x = actual_dk_x - expected_dk_x  # should be 0
            err_y = actual_dk_y - expected_dk_y

            n_bad = (err_x != 0).sum() + (err_y != 0).sum()
            if n_bad == 0:
                break

            # For inconsistent edges, adjust the lower-coherence pixel
            correction = torch.zeros_like(k)

            # Horizontal edges with error
            bad_x = err_x != 0
            # Left pixel is weaker -> adjust left pixel's k
            left_weaker = bad_x & (coh_weights[:, :-1] <= coh_weights[:, 1:])
            # Right pixel is weaker -> adjust right pixel's k
            right_weaker = bad_x & ~left_weaker & bad_x

            # Clamp err to +/-1 for stability
            err_x_clamped = torch.clamp(err_x, -1, 1)
            correction[:, :-1] += torch.where(left_weaker, err_x_clamped, torch.zeros_like(err_x_clamped))
            correction[:, 1:] -= torch.where(right_weaker, err_x_clamped, torch.zeros_like(err_x_clamped))

            # Vertical edges with error
            bad_y = err_y != 0
            top_weaker = bad_y & (coh_weights[:-1, :] <= coh_weights[1:, :])
            bottom_weaker = bad_y & ~top_weaker & bad_y

            err_y_clamped = torch.clamp(err_y, -1, 1)
            correction[:-1, :] += torch.where(top_weaker, err_y_clamped, torch.zeros_like(err_y_clamped))
            correction[1:, :] -= torch.where(bottom_weaker, err_y_clamped, torch.zeros_like(err_y_clamped))

            if nan_mask is not None:
                correction[nan_mask] = 0

            # Apply correction (clamp to +/-1 per step for stability)
            correction = torch.clamp(correction, -1, 1).round()

            if (correction == 0).all():
                break

            k += correction

        return phase_clean + k * two_pi

    def _apply_dct_preconditioner(self, r: torch.Tensor) -> torch.Tensor:
        """
        Apply DCT-based preconditioner: z = L^{-1} r.

        The unweighted Laplacian inverse (computed via DCT) serves as
        an effective preconditioner for the weighted Laplacian system.

        Parameters
        ----------
        r : torch.Tensor
            Residual vector of shape (H, W).

        Returns
        -------
        torch.Tensor
            Preconditioned residual z = L^{-1} r.
        """
        H, W = r.shape
        r_dct = self._dct_solver._dct2(r)
        eigenvalues = self._dct_solver._get_eigenvalues(H, W)
        z_dct = r_dct / eigenvalues
        z_dct[0, 0] = 0.0  # Zero DC component
        return self._dct_solver._idct2(z_dct)

    def _cg_solve(
        self,
        phi_init: torch.Tensor,
        grad_x_target: torch.Tensor,
        grad_y_target: torch.Tensor,
        weights_x: torch.Tensor,
        weights_y: torch.Tensor,
        max_iterations: int,
    ) -> torch.Tensor:
        """
        Solve weighted least squares using DCT-Preconditioned Conjugate Gradient.

        Solves: min Σ w_x_ij * |∂xφ_ij - gx_ij|² + w_y_ij * |∂yφ_ij - gy_ij|²

        This is equivalent to solving the weighted Poisson equation:
            div(W * ∇φ) = div(W * g)

        The DCT preconditioner (unweighted Laplacian inverse) dramatically
        reduces iteration count from ~200 to ~10-20.

        Parameters
        ----------
        phi_init : torch.Tensor
            Initial guess for φ.
        grad_x_target, grad_y_target : torch.Tensor
            Target gradients.
        weights_x, weights_y : torch.Tensor
            Per-edge weights for horizontal and vertical gradients.
        max_iterations : int
            Maximum CG iterations.

        Returns
        -------
        torch.Tensor
            Solution φ.
        """
        phi = phi_init.clone()

        # Compute RHS: div(W * g)
        rhs = self._weighted_divergence(grad_x_target, grad_y_target, weights_x, weights_y)

        # Initial residual: r = b - A*x
        Aphi = self._weighted_laplacian(phi, weights_x, weights_y)
        r = rhs - Aphi

        # Apply preconditioner: z = M^{-1} r
        z = self._apply_dct_preconditioner(r)

        # Initial search direction
        p = z.clone()

        # Initial preconditioned residual inner product
        rz = torch.sum(r * z)
        rz_init = rz.clone()

        for cg_iter in range(max_iterations):
            # Matrix-vector product: A*p
            Ap = self._weighted_laplacian(p, weights_x, weights_y)

            # Step size
            pAp = torch.sum(p * Ap)
            if pAp.abs() < 1e-12:
                break
            alpha = rz / pAp

            # Update solution and residual
            phi = phi + alpha * p
            r = r - alpha * Ap

            # Apply preconditioner to new residual
            z_new = self._apply_dct_preconditioner(r)

            # New preconditioned inner product
            rz_new = torch.sum(r * z_new)

            # Check convergence
            if rz_new.abs() < self.cg_tolerance ** 2 * rz_init.abs():
                break

            # Update search direction
            beta = rz_new / (rz + 1e-30)
            p = z_new + beta * p

            rz = rz_new

        return phi

    def _weighted_laplacian(
        self,
        phi: torch.Tensor,
        weights_x: torch.Tensor,
        weights_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted Laplacian: -div(W * ∇φ) with per-edge weights.

        This is the operator A in the linear system A*φ = b.
        Uses weights_x for horizontal edges and weights_y for vertical edges.
        """
        H, W = phi.shape

        # Forward gradients
        grad_x = torch.zeros_like(phi)
        grad_y = torch.zeros_like(phi)
        grad_x[:, :-1] = phi[:, 1:] - phi[:, :-1]
        grad_y[:-1, :] = phi[1:, :] - phi[:-1, :]

        # Weight each edge direction independently
        wgrad_x = weights_x * grad_x
        wgrad_y = weights_y * grad_y

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
        weights_x: torch.Tensor,
        weights_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted divergence: div(W * g) with per-edge weights.

        This is the RHS b in the linear system.
        """
        H, W = grad_x.shape

        # Weight each gradient component with its own weight
        wgrad_x = weights_x * grad_x
        wgrad_y = weights_y * grad_y

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
