"""
Quality and coherence map utilities for weighted phase unwrapping.
"""

from __future__ import annotations

import torch


def compute_quality_map(
    phase: torch.Tensor,
    coherence: torch.Tensor | None = None,
    method: str = "phase_derivative_variance",
) -> torch.Tensor:
    """
    Compute a quality map for phase unwrapping.

    Quality maps indicate the reliability of each pixel, with higher
    values indicating more reliable phase measurements.

    Parameters
    ----------
    phase : torch.Tensor
        Wrapped phase of shape (H, W).
    coherence : torch.Tensor, optional
        Coherence/correlation map of shape (H, W), values in [0, 1].
    method : str
        Quality estimation method:
        - "coherence": Use coherence directly (requires coherence input)
        - "phase_derivative_variance": Based on local phase gradient variance

    Returns
    -------
    torch.Tensor
        Quality map of shape (H, W), values typically in [0, 1].
    """
    if method == "coherence":
        if coherence is None:
            raise ValueError("coherence input required for method='coherence'")
        return coherence.clone()

    elif method == "phase_derivative_variance":
        return _pdv_quality(phase, coherence)

    else:
        raise ValueError(f"Unknown quality method: {method}")


def _pdv_quality(
    phase: torch.Tensor,
    coherence: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute quality based on phase derivative variance.

    Lower variance indicates more reliable phase measurements.
    """
    from rapidphase.utils.phase_ops import wrap

    H, W = phase.shape

    # Compute wrapped gradients
    # Pad to keep same size
    phase_pad = torch.nn.functional.pad(
        phase.unsqueeze(0).unsqueeze(0),
        (1, 1, 1, 1),
        mode="replicate",
    ).squeeze()

    # Gradients in 4 directions
    dx_p = wrap(phase_pad[1:-1, 2:] - phase_pad[1:-1, 1:-1])
    dx_n = wrap(phase_pad[1:-1, 1:-1] - phase_pad[1:-1, :-2])
    dy_p = wrap(phase_pad[2:, 1:-1] - phase_pad[1:-1, 1:-1])
    dy_n = wrap(phase_pad[1:-1, 1:-1] - phase_pad[:-2, 1:-1])

    # Stack gradients
    grads = torch.stack([dx_p, dx_n, dy_p, dy_n], dim=0)

    # Variance of gradients at each pixel
    grad_var = torch.var(grads, dim=0)

    # Convert variance to quality (inverse relationship)
    # Use exponential decay: high variance -> low quality
    quality = torch.exp(-grad_var)

    # Combine with coherence if available
    if coherence is not None:
        quality = quality * coherence

    return quality


def coherence_to_weights(
    coherence: torch.Tensor,
    nlooks: float = 1.0,
    min_weight: float = 1e-6,
) -> torch.Tensor:
    """
    Convert coherence values to weights for IRLS unwrapping.

    Uses the relationship between coherence and phase variance
    to derive appropriate weights.

    Parameters
    ----------
    coherence : torch.Tensor
        Coherence values in [0, 1].
    nlooks : float
        Number of looks (averaging) used in coherence estimation.
        Higher values give more reliable coherence estimates.
    min_weight : float
        Minimum weight to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Weights suitable for least-squares unwrapping.
    """
    # Clamp coherence to valid range
    coh = torch.clamp(coherence, min=0.0, max=1.0)

    # Phase variance approximation (Rodriguez & Martin, 1992):
    # sigma_phi^2 ≈ (1 - coh^2) / (2 * nlooks * coh^2)
    # Weight = 1 / variance ∝ coh^2 / (1 - coh^2)

    # Avoid division by zero
    coh_sq = coh ** 2
    one_minus_coh_sq = torch.clamp(1 - coh_sq, min=1e-10)

    # Weight proportional to inverse variance
    weights = (nlooks * coh_sq) / one_minus_coh_sq

    # Normalize to [0, 1] range and apply minimum
    weights = weights / (weights.max() + 1e-10)
    weights = torch.clamp(weights, min=min_weight)

    return weights


def estimate_coherence_from_complex(
    igram: torch.Tensor,
    window_size: int = 5,
) -> torch.Tensor:
    """
    Estimate coherence from complex interferogram using local averaging.

    Parameters
    ----------
    igram : torch.Tensor
        Complex interferogram.
    window_size : int
        Size of averaging window (should be odd).

    Returns
    -------
    torch.Tensor
        Estimated coherence map.
    """
    # Create averaging kernel
    kernel_size = window_size
    kernel = torch.ones(
        (1, 1, kernel_size, kernel_size),
        dtype=igram.real.dtype,
        device=igram.device,
    ) / (kernel_size ** 2)

    # Pad input
    pad = kernel_size // 2

    # Compute local averages
    # |<igram>|^2
    igram_4d = igram.unsqueeze(0).unsqueeze(0)
    real_avg = torch.nn.functional.conv2d(
        igram.real.unsqueeze(0).unsqueeze(0),
        kernel,
        padding=pad,
    ).squeeze()
    imag_avg = torch.nn.functional.conv2d(
        igram.imag.unsqueeze(0).unsqueeze(0),
        kernel,
        padding=pad,
    ).squeeze()
    mean_igram_sq = real_avg ** 2 + imag_avg ** 2

    # <|igram|^2>
    intensity = (igram.real ** 2 + igram.imag ** 2)
    mean_intensity = torch.nn.functional.conv2d(
        intensity.unsqueeze(0).unsqueeze(0),
        kernel,
        padding=pad,
    ).squeeze()

    # Coherence = |<igram>| / sqrt(<|igram|^2>)
    coherence = torch.sqrt(mean_igram_sq / (mean_intensity + 1e-10))
    coherence = torch.clamp(coherence, 0.0, 1.0)

    return coherence
