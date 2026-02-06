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

import math

import numpy as np
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
        max_irls_iterations: int = 50,
        max_cg_iterations: int = 200,
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

            # Update IRLS weights per-edge: w = C / max(|residual|, delta)
            # Separate weights for horizontal (x) and vertical (y) edges
            irls_weights_x = coh_weights / torch.clamp(torch.abs(res_x) + 1e-10, min=self.delta)
            irls_weights_y = coh_weights / torch.clamp(torch.abs(res_y) + 1e-10, min=self.delta)

            # Normalize weights for numerical stability (shared scale)
            max_w = max(irls_weights_x.max(), irls_weights_y.max()) + 1e-10
            irls_weights_x = irls_weights_x / max_w
            irls_weights_y = irls_weights_y / max_w

            # Set weights to 0 for NaN pixels
            if has_nans:
                irls_weights_x[nan_mask] = 0.0
                irls_weights_y[nan_mask] = 0.0

            # Solve weighted least squares using CG
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

        # Project to nearest congruent solution:
        # unwrapped = wrapped + k * 2*pi, where k is the nearest integer
        k = torch.round((phi - phase_clean) / (2 * math.pi))
        phi = phase_clean + k * (2 * math.pi)

        # Correct branch cut routing via quality-guided flood fill.
        # The DCT initialization may route branch cuts differently from
        # optimal (MCF-style) routing. This BFS correction grows outward
        # from the highest-coherence pixel, adjusting each pixel's integer
        # cycle to be gradient-consistent with already-visited neighbors.
        phi = self._gradient_guided_correction(
            phi, phase_clean, coh_weights, nan_mask if has_nans else None
        )

        # Restore NaN at invalid pixel locations
        if has_nans:
            phi[nan_mask] = float('nan')

        return phi

    def _gradient_guided_correction(
        self,
        phi: torch.Tensor,
        phase_clean: torch.Tensor,
        coh_weights: torch.Tensor,
        nan_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Correct 2π branch cut errors via quality-guided graph integration.

        Instead of relying on the congruence projection k values (which may
        have branch cut routing errors), this method integrates the wrapped
        phase gradients from the seed pixel along a Dijkstra shortest-path
        tree weighted by inverse coherence. This routes branch cuts through
        low-coherence areas (similar to SNAPHU's MCF approach).

        Uses scipy's C-compiled Dijkstra for the traversal order, then
        propagates k values along the tree in a single pass.

        Parameters
        ----------
        phi : torch.Tensor
            Congruent unwrapped phase (H, W).
        phase_clean : torch.Tensor
            Wrapped phase with NaN replaced by 0 (H, W).
        coh_weights : torch.Tensor
            Coherence-derived weights (H, W), 0 at invalid pixels.
        nan_mask : torch.Tensor or None
            Boolean mask of invalid pixels.

        Returns
        -------
        torch.Tensor
            Corrected unwrapped phase.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra

        H, W = phi.shape
        device = phi.device
        two_pi = 2 * math.pi
        n_pixels = H * W

        # Move to CPU numpy
        psi = phase_clean.detach().cpu().numpy()
        coh = coh_weights.detach().cpu().numpy()
        nan_np = (
            nan_mask.detach().cpu().numpy()
            if nan_mask is not None
            else np.zeros((H, W), dtype=bool)
        )
        valid = ~nan_np

        # Wrapping integers between adjacent pixels
        # k[j+1] = k[j] + m_x[j]  means gradient-consistent unwrapping
        raw_dx = psi[:, 1:] - psi[:, :-1]
        m_x = -np.rint(raw_dx / two_pi).astype(np.int64)
        raw_dy = psi[1:, :] - psi[:-1, :]
        m_y = -np.rint(raw_dy / two_pi).astype(np.int64)

        # Seed: highest-coherence valid pixel
        phi_np = phi.detach().cpu().numpy()
        coh_masked = np.where(valid, coh, -1.0)
        seed_idx = int(np.argmax(coh_masked))
        seed_r, seed_c = seed_idx // W, seed_idx % W
        seed_flat = seed_r * W + seed_c
        seed_k = int(np.round(
            (phi_np[seed_r, seed_c] - psi[seed_r, seed_c]) / two_pi
        ))

        # Build sparse grid graph weighted by INVERSE coherence
        # Low cost = high coherence, so Dijkstra routes through high-coh areas
        # and branch cuts end up in low-coherence regions.
        # Horizontal edges: (i,j) <-> (i,j+1)
        h_valid = valid[:, :-1] & valid[:, 1:]
        hr, hc = np.where(h_valid)
        idx_l = hr * W + hc
        idx_r = hr * W + (hc + 1)
        min_coh_h = np.minimum(coh[hr, hc], coh[hr, hc + 1])
        cost_h = 1.0 / (min_coh_h + 1e-6)

        # Vertical edges: (i,j) <-> (i+1,j)
        v_valid = valid[:-1, :] & valid[1:, :]
        vr, vc = np.where(v_valid)
        idx_t = vr * W + vc
        idx_b = (vr + 1) * W + vc
        min_coh_v = np.minimum(coh[vr, vc], coh[vr + 1, vc])
        cost_v = 1.0 / (min_coh_v + 1e-6)

        # Symmetric adjacency
        rows = np.concatenate([idx_l, idx_r, idx_t, idx_b])
        cols = np.concatenate([idx_r, idx_l, idx_b, idx_t])
        costs = np.concatenate([cost_h, cost_h, cost_v, cost_v])

        graph = csr_matrix(
            (costs, (rows, cols)), shape=(n_pixels, n_pixels)
        )

        # Dijkstra from seed — quality-guided traversal (scipy C-compiled)
        dist, predecessors = dijkstra(
            graph, indices=seed_flat, return_predecessors=True
        )

        # Sort by distance = Dijkstra visitation order (high-coh pixels first)
        order = np.argsort(dist)
        # Keep only reachable valid pixels (finite distance), skip seed
        reachable = np.isfinite(dist[order]) & (order != seed_flat)
        pixels = order[reachable]
        preds = predecessors[pixels]

        r_pix = pixels // W
        c_pix = pixels % W
        r_par = preds // W
        c_par = preds % W

        m_inc = np.zeros(len(pixels), dtype=np.int64)

        # From left parent (r, c-1) -> pixel (r, c)
        mask = (r_pix == r_par) & (c_pix == c_par + 1)
        m_inc[mask] = m_x[r_pix[mask], c_par[mask]]

        # From right parent (r, c+1) -> pixel (r, c)
        mask = (r_pix == r_par) & (c_pix == c_par - 1)
        m_inc[mask] = -m_x[r_pix[mask], c_pix[mask]]

        # From above parent (r-1, c) -> pixel (r, c)
        mask = (r_pix == r_par + 1) & (c_pix == c_par)
        m_inc[mask] = m_y[r_par[mask], c_pix[mask]]

        # From below parent (r+1, c) -> pixel (r, c)
        mask = (r_pix == r_par - 1) & (c_pix == c_par)
        m_inc[mask] = -m_y[r_pix[mask], c_pix[mask]]

        # Propagate k along BFS tree order (single sequential pass)
        # Each pixel's parent was visited earlier, so flat_k[parent] is set.
        flat_k = np.zeros(n_pixels, dtype=np.int64)
        flat_k[seed_flat] = seed_k

        for i in range(len(pixels)):
            flat_k[pixels[i]] = flat_k[preds[i]] + m_inc[i]

        k = flat_k.reshape(H, W)

        # Reconstruct corrected phase
        k_tensor = torch.from_numpy(k.astype(np.float64)).to(
            dtype=phi.dtype, device=device
        )
        return phase_clean + k_tensor * two_pi

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
        Solve weighted least squares using Conjugate Gradient.

        Solves: min Σ w_x_ij * |∂xφ_ij - gx_ij|² + w_y_ij * |∂yφ_ij - gy_ij|²

        This is equivalent to solving the weighted Poisson equation:
            div(W * ∇φ) = div(W * g)

        with separate weights W_x, W_y for horizontal and vertical edges.

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
        # Using backward difference for divergence (adjoint of forward gradient)
        rhs = self._weighted_divergence(grad_x_target, grad_y_target, weights_x, weights_y)

        # Initial residual: r = b - A*x
        Aphi = self._weighted_laplacian(phi, weights_x, weights_y)
        r = rhs - Aphi

        # Initial search direction
        p = r.clone()

        # Initial residual norm squared
        r_norm_sq = torch.sum(r * r)
        r0_norm_sq = r_norm_sq.clone()

        for cg_iter in range(max_iterations):
            # Matrix-vector product: A*p
            Ap = self._weighted_laplacian(p, weights_x, weights_y)

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
