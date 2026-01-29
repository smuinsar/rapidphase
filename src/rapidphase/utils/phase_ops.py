"""
GPU-accelerated phase operations.

All operations are designed for parallel execution on GPU.
"""

from __future__ import annotations

import math

import torch


def wrap(phase: torch.Tensor) -> torch.Tensor:
    """
    Wrap phase values to the interval [-pi, pi].

    This operation is embarrassingly parallel - each pixel
    is computed independently.

    Parameters
    ----------
    phase : torch.Tensor
        Phase values (any range).

    Returns
    -------
    torch.Tensor
        Wrapped phase in [-pi, pi].
    """
    return torch.atan2(torch.sin(phase), torch.cos(phase))


def rewrap(phase_diff: torch.Tensor) -> torch.Tensor:
    """
    Re-wrap phase differences to [-pi, pi].

    Equivalent to wrap() but may be slightly faster for
    differences that are already close to [-pi, pi].

    Parameters
    ----------
    phase_diff : torch.Tensor
        Phase differences.

    Returns
    -------
    torch.Tensor
        Wrapped phase differences.
    """
    return wrap(phase_diff)


def gradient(
    phase: torch.Tensor,
    wrap_result: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute wrapped phase gradients in x and y directions.

    Uses forward differences: grad_x[i,j] = phase[i,j+1] - phase[i,j]

    Parameters
    ----------
    phase : torch.Tensor
        2D phase array of shape (H, W).
    wrap_result : bool
        If True (default), wrap gradients to [-pi, pi].

    Returns
    -------
    grad_x : torch.Tensor
        Gradient in x (column) direction, shape (H, W-1).
    grad_y : torch.Tensor
        Gradient in y (row) direction, shape (H-1, W).
    """
    # Forward differences (parallel computation)
    grad_x = phase[:, 1:] - phase[:, :-1]
    grad_y = phase[1:, :] - phase[:-1, :]

    if wrap_result:
        grad_x = wrap(grad_x)
        grad_y = wrap(grad_y)

    return grad_x, grad_y


def gradient_full(
    phase: torch.Tensor,
    wrap_result: bool = True,
    nan_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute wrapped phase gradients with same output shape as input.

    Uses forward differences with zero-padding at boundaries.

    Parameters
    ----------
    phase : torch.Tensor
        2D phase array of shape (H, W).
    wrap_result : bool
        If True (default), wrap gradients to [-pi, pi].
    nan_mask : torch.Tensor, optional
        Boolean mask where True indicates NaN/invalid pixels. Gradients
        involving NaN pixels are set to zero.

    Returns
    -------
    grad_x : torch.Tensor
        Gradient in x direction, shape (H, W).
    grad_y : torch.Tensor
        Gradient in y direction, shape (H, W).
    """
    H, W = phase.shape

    # Handle NaN values
    if nan_mask is None:
        nan_mask = torch.isnan(phase)

    has_nans = nan_mask.any()

    if has_nans:
        phase_clean = phase.clone()
        phase_clean[nan_mask] = 0.0
    else:
        phase_clean = phase

    # Initialize with zeros
    grad_x = torch.zeros_like(phase)
    grad_y = torch.zeros_like(phase)

    # Forward differences (last column/row remains zero)
    grad_x[:, :-1] = phase_clean[:, 1:] - phase_clean[:, :-1]
    grad_y[:-1, :] = phase_clean[1:, :] - phase_clean[:-1, :]

    if wrap_result:
        grad_x = wrap(grad_x)
        grad_y = wrap(grad_y)

    if has_nans:
        # Zero out gradients that involve NaN pixels
        # grad_x[:, i] = phase[:, i+1] - phase[:, i] -> invalid if either is NaN
        invalid_grad_x = torch.zeros_like(nan_mask)
        invalid_grad_x[:, :-1] = nan_mask[:, :-1] | nan_mask[:, 1:]
        invalid_grad_x[:, -1] = True  # Last column is always zero (boundary)

        invalid_grad_y = torch.zeros_like(nan_mask)
        invalid_grad_y[:-1, :] = nan_mask[:-1, :] | nan_mask[1:, :]
        invalid_grad_y[-1, :] = True  # Last row is always zero (boundary)

        grad_x[invalid_grad_x] = 0.0
        grad_y[invalid_grad_y] = 0.0

    return grad_x, grad_y


def laplacian(phase: torch.Tensor, nan_mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute the wrapped Laplacian of a phase field.

    The Laplacian is computed as:
        L[i,j] = wrap(phase[i+1,j] - phase[i,j]) - wrap(phase[i,j] - phase[i-1,j])
               + wrap(phase[i,j+1] - phase[i,j]) - wrap(phase[i,j] - phase[i,j-1])

    This is the key operation for least-squares phase unwrapping.

    Parameters
    ----------
    phase : torch.Tensor
        2D wrapped phase array of shape (H, W).
    nan_mask : torch.Tensor, optional
        Boolean mask where True indicates NaN/invalid pixels. Gradients
        involving NaN pixels are set to zero, preventing NaN propagation.

    Returns
    -------
    torch.Tensor
        Wrapped Laplacian of shape (H, W).
    """
    H, W = phase.shape

    # Handle NaN values: replace with 0 for computation, then zero out
    # gradients that involve NaN pixels
    if nan_mask is None:
        nan_mask = torch.isnan(phase)

    has_nans = nan_mask.any()

    if has_nans:
        # Replace NaN with 0 for safe computation
        phase_clean = phase.clone()
        phase_clean[nan_mask] = 0.0
    else:
        phase_clean = phase

    # Pad phase with replicated boundary conditions
    # This avoids boundary artifacts in the Laplacian
    phase_pad = torch.nn.functional.pad(
        phase_clean.unsqueeze(0).unsqueeze(0),
        (1, 1, 1, 1),
        mode="replicate",
    ).squeeze()

    # Compute wrapped differences (all parallel)
    # Forward differences in x
    dx_forward = wrap(phase_pad[1:-1, 2:] - phase_pad[1:-1, 1:-1])
    # Backward differences in x
    dx_backward = wrap(phase_pad[1:-1, 1:-1] - phase_pad[1:-1, :-2])
    # Forward differences in y
    dy_forward = wrap(phase_pad[2:, 1:-1] - phase_pad[1:-1, 1:-1])
    # Backward differences in y
    dy_backward = wrap(phase_pad[1:-1, 1:-1] - phase_pad[:-2, 1:-1])

    if has_nans:
        # Pad NaN mask similarly to identify invalid neighbors
        nan_mask_pad = torch.nn.functional.pad(
            nan_mask.unsqueeze(0).unsqueeze(0).float(),
            (1, 1, 1, 1),
            mode="constant",
            value=1.0,  # Treat out-of-bounds as NaN for gradient zeroing
        ).squeeze() > 0.5

        # Zero out gradients that involve NaN pixels
        # dx_forward: from [1:-1, 1:-1] to [1:-1, 2:] - invalid if either is NaN
        invalid_dx_forward = nan_mask_pad[1:-1, 1:-1] | nan_mask_pad[1:-1, 2:]
        # dx_backward: from [1:-1, :-2] to [1:-1, 1:-1] - invalid if either is NaN
        invalid_dx_backward = nan_mask_pad[1:-1, :-2] | nan_mask_pad[1:-1, 1:-1]
        # dy_forward: from [1:-1, 1:-1] to [2:, 1:-1] - invalid if either is NaN
        invalid_dy_forward = nan_mask_pad[1:-1, 1:-1] | nan_mask_pad[2:, 1:-1]
        # dy_backward: from [:-2, 1:-1] to [1:-1, 1:-1] - invalid if either is NaN
        invalid_dy_backward = nan_mask_pad[:-2, 1:-1] | nan_mask_pad[1:-1, 1:-1]

        dx_forward[invalid_dx_forward] = 0.0
        dx_backward[invalid_dx_backward] = 0.0
        dy_forward[invalid_dy_forward] = 0.0
        dy_backward[invalid_dy_backward] = 0.0

    # Laplacian = second differences
    lap = (dx_forward - dx_backward) + (dy_forward - dy_backward)

    return lap


def compute_residues(phase: torch.Tensor) -> torch.Tensor:
    """
    Compute phase residues (topological charges).

    Residues are locations where the wrapped phase gradient has
    non-zero circulation, indicating phase discontinuities.

    Parameters
    ----------
    phase : torch.Tensor
        2D wrapped phase array of shape (H, W).

    Returns
    -------
    torch.Tensor
        Residue map of shape (H-1, W-1).
        Values are approximately -1, 0, or +1 (normalized by 2*pi).
    """
    # Compute wrapped gradients
    grad_x, grad_y = gradient(phase, wrap_result=True)

    # Circulation around each 2x2 cell
    # Going clockwise: right, down, left, up
    circulation = (
        grad_x[:-1, :]       # right edge (top)
        + grad_y[:, 1:]      # down edge (right)
        - grad_x[1:, :]      # left edge (bottom, reversed)
        - grad_y[:, :-1]     # up edge (left, reversed)
    )

    # Normalize by 2*pi to get integer residues
    residues = circulation / (2 * math.pi)

    return residues


def integrate_phase(
    grad_x: torch.Tensor,
    grad_y: torch.Tensor,
    start_value: float = 0.0,
) -> torch.Tensor:
    """
    Integrate phase gradients to recover phase.

    Simple row-by-row integration (not the most accurate method,
    but useful for testing).

    Parameters
    ----------
    grad_x : torch.Tensor
        Gradient in x direction, shape (H, W-1).
    grad_y : torch.Tensor
        Gradient in y direction, shape (H-1, W).
    start_value : float
        Starting phase value at (0, 0).

    Returns
    -------
    torch.Tensor
        Integrated phase of shape (H, W).
    """
    H = grad_y.shape[0] + 1
    W = grad_x.shape[1] + 1

    phase = torch.zeros((H, W), dtype=grad_x.dtype, device=grad_x.device)
    phase[0, 0] = start_value

    # Integrate along first row
    phase[0, 1:] = torch.cumsum(grad_x[0, :], dim=0) + start_value

    # Integrate along first column
    phase[1:, 0] = torch.cumsum(grad_y[:, 0], dim=0) + start_value

    # Integrate the rest (row by row)
    for i in range(1, H):
        phase[i, 1:] = torch.cumsum(grad_x[i, :], dim=0) + phase[i, 0]

    return phase
