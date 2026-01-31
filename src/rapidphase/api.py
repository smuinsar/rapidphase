"""
Public API for RapidPhase phase unwrapping.

This module provides the main entry points for phase unwrapping,
with GPU acceleration options.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from rapidphase.device.manager import DeviceManager, DeviceType, DeviceInfo
from rapidphase.core.dct_solver import DCTUnwrapper
from rapidphase.core.irls_solver import IRLSUnwrapper
from rapidphase.core.irls_cg_solver import IRLSCGUnwrapper
from rapidphase.filtering.goldstein import GoldsteinFilter
from rapidphase.tiling.tile_manager import TileManager

AlgorithmType = Literal["dct", "irls", "irls_cg", "auto"]


def unwrap(
    igram: np.ndarray,
    corr: np.ndarray | None = None,
    nlooks: float = 1.0,
    algorithm: AlgorithmType = "auto",
    device: DeviceType = "auto",
    ntiles: tuple[int, int] | None = None,
    tile_overlap: int | None = None,
    n_gpus: int | None = None,
    max_iterations: int = 50,
    tolerance: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unwrap interferometric phase using GPU-accelerated algorithms.

    This is the main entry point for phase unwrapping, designed to be
    compatible with snaphu-py while providing GPU acceleration.

    Parameters
    ----------
    igram : np.ndarray
        Input interferogram. Can be:
        - Complex array: phase is extracted as np.angle(igram)
        - Real array: assumed to be wrapped phase in [-pi, pi]
    corr : np.ndarray, optional
        Coherence/correlation map, values in [0, 1].
        If provided with algorithm="auto", uses IRLS for weighting.
    nlooks : float
        Number of looks used in coherence estimation.
        Used for coherence-to-weight conversion in IRLS.
    algorithm : str
        Unwrapping algorithm:
        - "dct": Fast unweighted DCT-based least squares
        - "irls": IRLS with Jacobi smoother (simple, GPU-friendly)
        - "irls_cg": IRLS with Conjugate Gradient (faster convergence)
        - "auto": Selects based on whether coherence is provided
    device : str
        Compute device: "cuda", "mps", "cpu", or "auto" (default).
        "auto" selects best available (CUDA > MPS > CPU).
    ntiles : tuple of int, optional
        If provided, splits image into (nrow, ncol) tiles for processing.
        Useful for large images that don't fit in GPU memory.
    tile_overlap : int, optional
        Overlap in pixels between tiles. If None (default), automatically
        calculated as 10% of the smaller tile dimension for good blending.
    n_gpus : int, optional
        Number of GPUs to use for parallel tile processing. If None (default),
        uses all available GPUs. Set this explicitly on HPC systems where
        torch.cuda.device_count() may return more GPUs than allocated.
    max_iterations : int
        Maximum IRLS iterations (default 50, only for algorithm="irls").
    tolerance : float
        IRLS convergence tolerance (default 1e-4).

    Returns
    -------
    unw : np.ndarray
        Unwrapped phase.
    conncomp : np.ndarray
        Connected component labels (currently all ones, for API compatibility).

    Examples
    --------
    Basic usage with auto device selection:

    >>> import rapidphase
    >>> unw, conncomp = rapidphase.unwrap(igram, corr, nlooks=5.0)

    Explicit GPU selection:

    >>> unw, conncomp = rapidphase.unwrap(igram, corr, device="cuda")

    Fast DCT algorithm (no coherence weighting):

    >>> unw, conncomp = rapidphase.unwrap(igram, algorithm="dct")

    Large image with tiling:

    >>> unw, conncomp = rapidphase.unwrap(
    ...     igram, corr,
    ...     ntiles=(4, 4),
    ...     tile_overlap=64,
    ... )
    """
    # Initialize device manager
    dm = DeviceManager(device)

    # Extract phase from complex interferogram
    if np.iscomplexobj(igram):
        phase = np.angle(igram).astype(np.float64)
        # NaN mask from complex interferogram (NaN in real or imag part)
        nan_mask = np.isnan(igram.real) | np.isnan(igram.imag)
    else:
        phase = igram.astype(np.float64)
        nan_mask = np.isnan(phase)

    # Handle coherence
    coherence = None
    if corr is not None:
        coherence = np.clip(np.nan_to_num(corr.astype(np.float64), nan=0.0), 0.0, 1.0)
        # Also mark NaN in coherence as invalid
        nan_mask = nan_mask | np.isnan(corr)

    # Select algorithm
    if algorithm == "auto":
        algorithm = "irls" if coherence is not None else "dct"

    # Create unwrapper
    if algorithm == "dct":
        unwrapper = DCTUnwrapper(dm)
    elif algorithm == "irls":
        unwrapper = IRLSUnwrapper(
            dm,
            max_iterations=max_iterations,
            tolerance=tolerance,
            nlooks=nlooks,
        )
    elif algorithm == "irls_cg":
        unwrapper = IRLSCGUnwrapper(
            dm,
            max_irls_iterations=max_iterations,
            irls_tolerance=tolerance,
            nlooks=nlooks,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Process with or without tiling
    if ntiles is not None:
        # Memory-efficient path: keep data as numpy, only convert tiles to GPU
        n_rows, n_cols = ntiles
        tile_h = phase.shape[0] // n_rows
        tile_w = phase.shape[1] // n_cols
        min_tile_dim = min(tile_h, tile_w)

        # Auto-calculate overlap if not specified (10% of min tile dimension)
        if tile_overlap is None:
            tile_overlap = max(64, int(min_tile_dim * 0.1))
        elif tile_overlap < min_tile_dim * 0.05:
            # Warn if user-specified overlap is too small
            import warnings
            warnings.warn(
                f"tile_overlap={tile_overlap} is small relative to tile size "
                f"({tile_h}x{tile_w}). Consider using overlap of at least "
                f"{int(min_tile_dim * 0.1)} pixels for better blending.",
                UserWarning,
            )

        tile_manager = TileManager(dm, ntiles=ntiles, overlap=tile_overlap)

        # Create unwrapper factory for multi-GPU support
        def create_unwrapper_for_device(device_manager):
            if algorithm == "dct":
                return DCTUnwrapper(device_manager)
            elif algorithm == "irls":
                return IRLSUnwrapper(
                    device_manager,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    nlooks=nlooks,
                )
            else:  # irls_cg
                return IRLSCGUnwrapper(
                    device_manager,
                    max_irls_iterations=max_iterations,
                    irls_tolerance=tolerance,
                    nlooks=nlooks,
                )

        # Use numpy-based tiling to avoid loading entire image to GPU
        unw = tile_manager.process_tiled_numpy(
            phase,
            unwrapper_factory=create_unwrapper_for_device,
            coherence=coherence,
            nan_mask=nan_mask,
            n_gpus=n_gpus,
            verbose=True,
        )
    else:
        # Standard path: convert entire image to GPU tensors
        phase_t = dm.to_tensor(phase)
        coh_t = dm.to_tensor(coherence) if coherence is not None else None
        nan_mask_t = dm.to_tensor(nan_mask.astype(np.float32)) > 0.5  # Convert to bool tensor

        unw_t = unwrapper.unwrap(phase_t, coh_t, nan_mask=nan_mask_t)

        # Convert back to numpy
        unw = dm.to_numpy(unw_t)

    # Create connected component array
    # Mark NaN regions as component 0 (invalid), valid regions as 1
    conncomp = np.ones(unw.shape, dtype=np.int32)
    conncomp[nan_mask] = 0

    return unw, conncomp


def unwrap_dct(
    igram: np.ndarray,
    device: DeviceType = "auto",
    ntiles: tuple[int, int] | None = None,
    tile_overlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unwrap phase using fast DCT-based least squares.

    This is the fastest algorithm but does not use coherence weighting.
    Best for data with generally good coherence.

    Parameters
    ----------
    igram : np.ndarray
        Input interferogram (complex or phase).
    device : str
        Compute device: "cuda", "mps", "cpu", or "auto".
    ntiles : tuple of int, optional
        Tile configuration for large images.
    tile_overlap : int, optional
        Overlap between tiles. Auto-calculated if None.

    Returns
    -------
    unw : np.ndarray
        Unwrapped phase.
    conncomp : np.ndarray
        Connected component labels.

    See Also
    --------
    unwrap : Main unwrapping function with all options.
    unwrap_irls : Weighted unwrapping using coherence.
    """
    return unwrap(
        igram,
        corr=None,
        algorithm="dct",
        device=device,
        ntiles=ntiles,
        tile_overlap=tile_overlap,
    )


def unwrap_irls(
    igram: np.ndarray,
    corr: np.ndarray | None = None,
    nlooks: float = 1.0,
    device: DeviceType = "auto",
    max_iterations: int = 50,
    tolerance: float = 1e-4,
    ntiles: tuple[int, int] | None = None,
    tile_overlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unwrap phase using IRLS with coherence weighting.

    This algorithm uses coherence information to weight the least-squares
    solution, providing better results in low-coherence regions.

    Parameters
    ----------
    igram : np.ndarray
        Input interferogram (complex or phase).
    corr : np.ndarray, optional
        Coherence map, values in [0, 1]. If None, falls back to DCT.
    nlooks : float
        Number of looks for weight conversion.
    device : str
        Compute device: "cuda", "mps", "cpu", or "auto".
    max_iterations : int
        Maximum IRLS iterations.
    tolerance : float
        Convergence tolerance.
    ntiles : tuple of int, optional
        Tile configuration for large images.
    tile_overlap : int, optional
        Overlap between tiles. Auto-calculated if None.

    Returns
    -------
    unw : np.ndarray
        Unwrapped phase.
    conncomp : np.ndarray
        Connected component labels.

    See Also
    --------
    unwrap : Main unwrapping function with all options.
    unwrap_dct : Fast unweighted unwrapping.
    """
    return unwrap(
        igram,
        corr=corr,
        nlooks=nlooks,
        algorithm="irls",
        device=device,
        max_iterations=max_iterations,
        tolerance=tolerance,
        ntiles=ntiles,
        tile_overlap=tile_overlap,
    )


def unwrap_irls_cg(
    igram: np.ndarray,
    corr: np.ndarray | None = None,
    nlooks: float = 1.0,
    device: DeviceType = "auto",
    max_iterations: int = 20,
    tolerance: float = 1e-4,
    delta: float = 0.1,
    ntiles: tuple[int, int] | None = None,
    tile_overlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unwrap phase using IRLS with Conjugate Gradient solver.

    This algorithm uses CG for faster convergence than the Jacobi-based IRLS.
    It approximates L1-norm minimization through iterative reweighting,
    which is more robust to outliers than pure L2.

    Parameters
    ----------
    igram : np.ndarray
        Input interferogram (complex or phase).
    corr : np.ndarray, optional
        Coherence map, values in [0, 1]. If None, uses uniform weights.
    nlooks : float
        Number of looks for weight conversion.
    device : str
        Compute device: "cuda", "mps", "cpu", or "auto".
    max_iterations : int
        Maximum IRLS iterations (default 20).
    tolerance : float
        Convergence tolerance.
    delta : float
        Smoothing parameter for IRLS weights. Smaller values approximate
        L1 more closely but may cause instability (default 0.1).
    ntiles : tuple of int, optional
        Tile configuration for large images.
    tile_overlap : int, optional
        Overlap between tiles. Auto-calculated if None.

    Returns
    -------
    unw : np.ndarray
        Unwrapped phase.
    conncomp : np.ndarray
        Connected component labels.

    See Also
    --------
    unwrap : Main unwrapping function with all options.
    unwrap_irls : Jacobi-based IRLS (simpler but slower convergence).
    unwrap_dct : Fast unweighted unwrapping.
    """
    return unwrap(
        igram,
        corr=corr,
        nlooks=nlooks,
        algorithm="irls_cg",
        device=device,
        max_iterations=max_iterations,
        tolerance=tolerance,
        ntiles=ntiles,
        tile_overlap=tile_overlap,
    )


def get_available_devices() -> dict:
    """
    Get information about available compute devices.

    Returns
    -------
    dict
        Dictionary with device availability:
        - 'cpu': Always True
        - 'cuda': True if CUDA is available
        - 'mps': True if MPS (Apple Silicon) is available
        - 'cuda_devices': List of CUDA device info (if available)

    Examples
    --------
    >>> import rapidphase
    >>> devices = rapidphase.get_available_devices()
    >>> print(devices)
    {'cpu': True, 'cuda': True, 'mps': False, 'cuda_devices': [...]}
    """
    return DeviceManager.get_available_devices().to_dict()


def goldstein_filter(
    igram: np.ndarray,
    alpha: float = 0.6,
    window_size: int = 64,
    overlap: float = 0.75,
    patch_batch_size: int = 1024,
    device: DeviceType = "auto",
) -> np.ndarray:
    """
    Apply Goldstein adaptive filter to reduce interferogram noise.

    The Goldstein filter is an adaptive frequency-domain filter that reduces
    phase noise while preserving fringe structure. It is typically applied
    before phase unwrapping to improve results.

    Parameters
    ----------
    igram : np.ndarray
        Complex interferogram of shape (H, W). NaN and zero values are
        handled automatically by masking during filtering.
    alpha : float
        Filter strength exponent (default 0.6). Higher values apply stronger
        filtering. Typical range is 0.2 to 1.0. Use lower values (0.2-0.4)
        for high-quality data, higher values (0.6-1.0) for noisy data.
    window_size : int
        Size of filtering window in pixels (default 64). Must be positive.
        Larger windows provide more spectral resolution but less spatial
        adaptivity. Common values are 32, 64, or 128.
    overlap : float
        Fractional overlap between adjacent windows (default 0.75).
        Higher overlap reduces edge artifacts but increases computation.
        Values between 0.5 and 0.875 work well.
    patch_batch_size : int
        Number of patches to process at once (default 1024). Reduce this
        value if encountering GPU out-of-memory errors with large images.
        For very large images on limited GPU memory, try 256 or 128.
    device : str
        Compute device: "cuda", "mps", "cpu", or "auto" (default).
        "auto" selects best available (CUDA > MPS > CPU).

    Returns
    -------
    np.ndarray
        Filtered complex interferogram of same shape. NaN positions from
        input are preserved in output.

    Notes
    -----
    The filter works by:
    1. Dividing the interferogram into overlapping windows
    2. For each window, computing the 2D FFT
    3. Smoothing the power spectrum
    4. Applying an adaptive filter H = (smoothed_power)^alpha
    5. Inverse FFT and weighted overlap-add

    NaN and zero values are masked during processing and restored afterward.
    The triangular window weighting ensures smooth blending between patches.

    For large images on GPU, patches are processed in batches to manage memory.
    If you encounter out-of-memory errors, reduce patch_batch_size (e.g., 256
    or 128) rather than switching to CPU.

    Examples
    --------
    Basic filtering before unwrapping:

    >>> import rapidphase
    >>> filtered = rapidphase.goldstein_filter(igram, alpha=0.6)
    >>> unw, conncomp = rapidphase.unwrap(filtered, corr)

    Stronger filtering for noisy data:

    >>> filtered = rapidphase.goldstein_filter(igram, alpha=0.8, window_size=64)

    GPU-accelerated filtering:

    >>> filtered = rapidphase.goldstein_filter(igram, device="cuda")

    Large image with limited GPU memory:

    >>> filtered = rapidphase.goldstein_filter(igram, patch_batch_size=256, device="cuda")

    References
    ----------
    Goldstein, R.M. and Werner, C.L. (1998). Radar interferogram filtering
    for geophysical applications. Geophysical Research Letters, 25(21).

    See Also
    --------
    unwrap : Main phase unwrapping function.
    """
    # Initialize device manager
    dm = DeviceManager(device)

    # Create filter
    filt = GoldsteinFilter(
        dm,
        alpha=alpha,
        window_size=window_size,
        overlap=overlap,
        patch_batch_size=patch_batch_size,
    )

    # Apply filter and return numpy array
    return filt(igram, return_numpy=True)
