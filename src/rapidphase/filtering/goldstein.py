"""
Goldstein adaptive filter for interferogram noise reduction.

The Goldstein filter applies frequency-domain filtering with adaptive
power spectrum weighting, reducing noise while preserving fringe structure.

Reference:
    Goldstein, R.M. and Werner, C.L. (1998). Radar interferogram filtering
    for geophysical applications. Geophysical Research Letters, 25(21).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import torch

# Try to import tqdm for progress bars
try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

if TYPE_CHECKING:
    from rapidphase.device.manager import DeviceManager


class GoldsteinFilter:
    """
    GPU-accelerated Goldstein adaptive filter for interferograms.

    This filter reduces phase noise in interferograms by applying an adaptive
    frequency-domain filter based on the local power spectrum. The filter
    strength adapts to the local signal-to-noise ratio.

    Parameters
    ----------
    device_manager : DeviceManager
        Device manager for GPU/CPU operations.
    alpha : float
        Filter strength exponent (default 0.6). Higher values apply stronger
        filtering. Typical range is 0.2 to 1.0.
    window_size : int
        Size of the filtering window in pixels (default 64). Must be a
        power of 2 for efficient FFT. Larger windows provide more spectral
        resolution but less spatial adaptivity.
    overlap : float
        Fractional overlap between adjacent windows (default 0.75).
        Higher overlap reduces edge artifacts but increases computation.
    patch_batch_size : int
        Number of patches to process at once (default 1024). Reduce this
        value if encountering GPU memory issues with large images.

    Attributes
    ----------
    alpha : float
        Filter strength exponent.
    window_size : int
        Size of filtering windows.
    overlap : float
        Window overlap fraction.
    patch_batch_size : int
        Batch size for patch processing.

    Examples
    --------
    >>> import rapidphase
    >>> from rapidphase.device import DeviceManager
    >>> from rapidphase.filtering import GoldsteinFilter
    >>>
    >>> dm = DeviceManager("auto")
    >>> filt = GoldsteinFilter(dm, alpha=0.6, window_size=64)
    >>> filtered = filt.filter(interferogram)

    For large images with limited GPU memory:

    >>> filt = GoldsteinFilter(dm, alpha=0.6, patch_batch_size=256)
    >>> filtered = filt.filter(large_interferogram)
    """

    def __init__(
        self,
        device_manager: DeviceManager,
        alpha: float = 0.6,
        window_size: int = 64,
        overlap: float = 0.75,
        patch_batch_size: int = 1024,
    ):
        self.dm = device_manager
        self.alpha = alpha
        self.window_size = window_size
        self.overlap = overlap
        self.patch_batch_size = patch_batch_size

        # Pre-compute triangular window for smooth blending
        self._window_2d = self._create_window()

    def _create_window(self) -> torch.Tensor:
        """Create 2D triangular window for smooth patch blending."""
        # Create 1D triangular window
        n = self.window_size
        window_1d = torch.linspace(0, 1, n // 2 + 1, device=self.dm.device)
        window_1d = torch.cat([window_1d[:-1], window_1d.flip(0)])
        if len(window_1d) < n:
            window_1d = torch.cat([window_1d, window_1d[-1:]])
        window_1d = window_1d[:n]

        # Create 2D window via outer product
        window_2d = torch.outer(window_1d, window_1d)

        return window_2d.to(self.dm.dtype)

    def _smooth_power_spectrum(self, power: torch.Tensor) -> torch.Tensor:
        """
        Smooth power spectrum using 3x3 box filter.

        Parameters
        ----------
        power : torch.Tensor
            Power spectrum of shape (..., H, W).

        Returns
        -------
        torch.Tensor
            Smoothed power spectrum.
        """
        # Create 3x3 averaging kernel
        kernel = torch.ones(1, 1, 3, 3, device=self.dm.device, dtype=self.dm.dtype) / 9.0

        # Handle batched input
        original_shape = power.shape
        if power.dim() == 2:
            power = power.unsqueeze(0).unsqueeze(0)
        elif power.dim() == 3:
            power = power.unsqueeze(1)

        # Apply convolution with padding
        smoothed = torch.nn.functional.conv2d(power, kernel, padding=1)

        # Restore original shape
        if len(original_shape) == 2:
            smoothed = smoothed.squeeze(0).squeeze(0)
        elif len(original_shape) == 3:
            smoothed = smoothed.squeeze(1)

        return smoothed

    def _extract_patches_unfold(
        self, data: torch.Tensor, step: int
    ) -> tuple[torch.Tensor, int, int]:
        """
        Extract overlapping patches using efficient unfold operation.

        Parameters
        ----------
        data : torch.Tensor
            Input data of shape (H, W).
        step : int
            Step size between patches.

        Returns
        -------
        patches : torch.Tensor
            Extracted patches of shape (N, window_size, window_size).
        n_rows : int
            Number of patch rows.
        n_cols : int
            Number of patch columns.
        """
        H, W = data.shape
        ws = self.window_size

        # Check if image is smaller than window
        if H < ws or W < ws:
            # Image smaller than window - pad and process as single patch
            padded = torch.zeros(ws, ws, dtype=data.dtype, device=data.device)
            padded[:H, :W] = data
            return padded.unsqueeze(0), 1, 1

        # Use unfold for efficient patch extraction
        # unfold(dimension, size, step) extracts patches along that dimension
        # Result shape: (H - ws) // step + 1, (W - ws) // step + 1, ws, ws
        patches = data.unfold(0, ws, step).unfold(1, ws, step)
        n_rows, n_cols = patches.shape[0], patches.shape[1]

        # Reshape to (N, ws, ws) where N = n_rows * n_cols
        patches = patches.contiguous().view(-1, ws, ws)

        return patches, n_rows, n_cols

    def _filter_patches_batched(
        self,
        patches: torch.Tensor,
        nan_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Goldstein filter to a batch of patches.

        Parameters
        ----------
        patches : torch.Tensor
            Complex patches of shape (N, window_size, window_size).
        nan_masks : torch.Tensor
            Boolean mask indicating NaN positions, shape (N, window_size, window_size).

        Returns
        -------
        torch.Tensor
            Filtered patches of same shape.
        """
        # Skip entirely NaN patches
        valid_ratio = (~nan_masks).to(self.dm.dtype).mean(dim=(1, 2))
        mostly_valid = valid_ratio > 0.5

        # Initialize output
        filtered = torch.zeros_like(patches)

        if not mostly_valid.any():
            return filtered

        # Process only valid patches
        valid_patches = patches[mostly_valid]
        valid_nan_masks = nan_masks[mostly_valid]

        # Replace NaN with zero for FFT
        clean_patches = valid_patches.clone()
        clean_patches[valid_nan_masks] = 0

        # Forward FFT
        spectrum = torch.fft.fftshift(torch.fft.fft2(clean_patches), dim=(-2, -1))

        # Compute power spectrum
        power = torch.abs(spectrum) ** 2

        # Smooth power spectrum
        smoothed_power = self._smooth_power_spectrum(power)

        # Compute adaptive filter: H = smoothed_power^alpha
        # Normalize to prevent amplitude explosion
        max_power = smoothed_power.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-10)
        normalized_power = smoothed_power / max_power
        H = normalized_power ** self.alpha

        # Apply filter
        filtered_spectrum = spectrum * H

        # Inverse FFT
        filtered_patches = torch.fft.ifft2(
            torch.fft.ifftshift(filtered_spectrum, dim=(-2, -1))
        )

        # Store results
        filtered[mostly_valid] = filtered_patches

        return filtered

    def filter(
        self,
        interferogram: torch.Tensor,
        return_numpy: bool = False,
    ) -> torch.Tensor:
        """
        Apply Goldstein adaptive filter to an interferogram.

        Parameters
        ----------
        interferogram : torch.Tensor or np.ndarray
            Complex interferogram of shape (H, W). NaN values are handled
            by masking during filtering and restored afterward.
        return_numpy : bool
            If True, return result as numpy array.

        Returns
        -------
        torch.Tensor or np.ndarray
            Filtered interferogram of same shape. NaN positions from input
            are preserved in output.
        """
        import numpy as np

        # Determine complex dtype based on device
        complex_dtype = torch.complex64 if self.dm.dtype == torch.float32 else torch.complex128

        # Convert input to tensor if needed
        if isinstance(interferogram, np.ndarray):
            # Handle complex numpy array
            if np.iscomplexobj(interferogram):
                # Convert to appropriate complex type for the device
                if self.dm.dtype == torch.float32:
                    interferogram = torch.from_numpy(interferogram.astype(np.complex64)).to(self.dm.device)
                else:
                    interferogram = torch.from_numpy(interferogram.astype(np.complex128)).to(self.dm.device)
            else:
                interferogram = self.dm.to_tensor(interferogram)
        else:
            # Ensure tensor is on correct device and dtype
            interferogram = interferogram.to(device=self.dm.device, dtype=complex_dtype)

        H, W = interferogram.shape

        # Create NaN mask (handle both real and imaginary parts)
        if interferogram.is_complex():
            nan_mask = torch.isnan(interferogram.real) | torch.isnan(interferogram.imag)
        else:
            nan_mask = torch.isnan(interferogram)

        # Create zero mask
        zero_mask = interferogram == 0

        # Combined invalid mask
        invalid_mask = nan_mask | zero_mask

        # Replace invalid values with zero for processing
        clean_data = interferogram.clone()
        clean_data[invalid_mask] = 0

        # Ensure complex type
        if not clean_data.is_complex():
            clean_data = clean_data.to(complex_dtype)

        # Calculate step size
        step = max(1, int(self.window_size * (1 - self.overlap)))

        # Use reflection padding to handle edges properly
        # Pad by half window size on each side to ensure full coverage
        pad_size = self.window_size // 2

        # For reflection padding, pad size must be < input dimension
        # Use replicate padding for very small images
        if H <= pad_size or W <= pad_size:
            # For very small images, use replicate padding
            pad_mode = 'replicate'
        else:
            pad_mode = 'reflect'

        # Pad the data (maintains phase continuity at edges)
        # PyTorch pad format: (left, right, top, bottom)
        padded_data = torch.nn.functional.pad(
            clean_data.unsqueeze(0).unsqueeze(0),
            (pad_size, pad_size, pad_size, pad_size),
            mode=pad_mode
        ).squeeze(0).squeeze(0)

        # Pad the invalid mask with False (reflected pixels are valid)
        padded_mask = torch.nn.functional.pad(
            invalid_mask.unsqueeze(0).unsqueeze(0).float(),
            (pad_size, pad_size, pad_size, pad_size),
            mode=pad_mode
        ).squeeze(0).squeeze(0) > 0.5

        pH, pW = padded_data.shape

        # Additional padding to ensure we cover the entire padded area
        extra_pad_h = (step - (pH - self.window_size) % step) % step
        extra_pad_w = (step - (pW - self.window_size) % step) % step

        if extra_pad_h > 0 or extra_pad_w > 0:
            # For additional padding, use replicate if dimensions are small
            extra_pad_mode = 'replicate' if pH <= extra_pad_h or pW <= extra_pad_w else 'reflect'
            padded_data = torch.nn.functional.pad(
                padded_data.unsqueeze(0).unsqueeze(0),
                (0, extra_pad_w, 0, extra_pad_h),
                mode=extra_pad_mode
            ).squeeze(0).squeeze(0)
            padded_mask = torch.nn.functional.pad(
                padded_mask.unsqueeze(0).unsqueeze(0).float(),
                (0, extra_pad_w, 0, extra_pad_h),
                mode=extra_pad_mode
            ).squeeze(0).squeeze(0) > 0.5

        pH, pW = padded_data.shape
        ws = self.window_size

        # Extract patches using efficient unfold operation
        patches, n_patch_rows, n_patch_cols = self._extract_patches_unfold(padded_data, step)
        mask_patches, _, _ = self._extract_patches_unfold(padded_mask.to(self.dm.dtype), step)
        mask_patches = mask_patches > 0.5  # Convert back to bool

        num_patches = patches.shape[0]

        # Process patches in batches to manage GPU memory
        # Collect all filtered patches
        all_filtered = []
        all_weights = []

        window_2d_complex = self._window_2d.to(complex_dtype)

        for batch_start in range(0, num_patches, self.patch_batch_size):
            batch_end = min(batch_start + self.patch_batch_size, num_patches)

            # Get batch of patches
            batch_patches = patches[batch_start:batch_end]
            batch_masks = mask_patches[batch_start:batch_end]

            # Filter this batch
            filtered_batch = self._filter_patches_batched(batch_patches, batch_masks)

            # Apply window weighting (vectorized for whole batch)
            weighted_batch = filtered_batch * window_2d_complex.unsqueeze(0)

            # Compute weight patches (vectorized)
            # Weight is window * valid_mask
            valid_masks = (~batch_masks).to(self.dm.dtype)  # (B, ws, ws)
            weight_batch = self._window_2d.unsqueeze(0) * valid_masks  # (B, ws, ws)

            all_filtered.append(weighted_batch)
            all_weights.append(weight_batch)

            # Free intermediate memory
            del filtered_batch, batch_patches, batch_masks

        # Concatenate all batches
        all_filtered = torch.cat(all_filtered, dim=0)  # (N, ws, ws)
        all_weights = torch.cat(all_weights, dim=0)  # (N, ws, ws)

        # Use fold to efficiently accumulate patches
        # fold expects input of shape (batch, C * kernel_h * kernel_w, L)
        # where L is the number of patches
        # Output shape is (batch, C, H, W)

        # Reshape filtered patches for fold: (N, ws, ws) -> (1, ws*ws, N)
        # Handle complex by separating real and imaginary parts
        filtered_real = all_filtered.real.reshape(num_patches, -1).T.unsqueeze(0)  # (1, ws*ws, N)
        filtered_imag = all_filtered.imag.reshape(num_patches, -1).T.unsqueeze(0)  # (1, ws*ws, N)
        weights_flat = all_weights.reshape(num_patches, -1).T.unsqueeze(0)  # (1, ws*ws, N)

        # Use fold to accumulate (this sums overlapping regions automatically)
        output_real = torch.nn.functional.fold(
            filtered_real,
            output_size=(pH, pW),
            kernel_size=ws,
            stride=step
        ).squeeze(0).squeeze(0)

        output_imag = torch.nn.functional.fold(
            filtered_imag,
            output_size=(pH, pW),
            kernel_size=ws,
            stride=step
        ).squeeze(0).squeeze(0)

        weights_sum = torch.nn.functional.fold(
            weights_flat,
            output_size=(pH, pW),
            kernel_size=ws,
            stride=step
        ).squeeze(0).squeeze(0)

        # Combine real and imaginary parts
        output = torch.complex(output_real, output_imag)

        # Free memory
        del all_filtered, all_weights, filtered_real, filtered_imag, weights_flat
        if self.dm.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Normalize by weights
        valid_weights = weights_sum > 1e-10
        output[valid_weights] = output[valid_weights] / weights_sum[valid_weights]

        # Crop to original size (accounting for reflection padding offset)
        output = output[pad_size:pad_size + H, pad_size:pad_size + W]

        # Restore NaN values
        output[nan_mask] = float('nan')

        if return_numpy:
            return self.dm.to_numpy(output)

        return output

    def __call__(
        self,
        interferogram,
        return_numpy: bool = True,
    ):
        """
        Convenience method for filtering with numpy input/output.

        Parameters
        ----------
        interferogram : np.ndarray or torch.Tensor
            Complex interferogram.
        return_numpy : bool
            If True (default), return numpy array.

        Returns
        -------
        np.ndarray or torch.Tensor
            Filtered interferogram.
        """
        return self.filter(interferogram, return_numpy=return_numpy)

    @property
    def device(self) -> torch.device:
        """The device used for computation."""
        return self.dm.device

    @property
    def dtype(self) -> torch.dtype:
        """The dtype used for computation."""
        return self.dm.dtype


def _get_gpu_count() -> int:
    """Get number of available GPUs using torch.cuda."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def filter_multi_gpu(
    igram: np.ndarray,
    alpha: float = 0.6,
    window_size: int = 64,
    overlap: float = 0.75,
    patch_batch_size: int = 1024,
    n_gpus: int | None = None,
    device: str = "auto",
    ntiles: tuple[int, int] | str = "auto",
    verbose: bool = True,
) -> np.ndarray:
    """
    Apply Goldstein filter using multiple GPUs in parallel with tiling.

    For large images, splits into tiles that fit in GPU memory. Each tile
    is processed separately and results are merged with smooth blending.

    Parameters
    ----------
    igram : np.ndarray
        Complex interferogram of shape (H, W).
    alpha : float
        Filter strength exponent (default 0.6).
    window_size : int
        Size of filtering window in pixels (default 64).
    overlap : float
        Fractional overlap between adjacent windows (default 0.75).
    patch_batch_size : int
        Number of patches to process at once per GPU (default 1024).
    n_gpus : int, optional
        Number of GPUs to use. If None, uses all available GPUs.
    device : str
        Fallback device when multi-GPU is not available: "cuda", "mps",
        "cpu", or "auto" (default).
    ntiles : tuple of int or str
        Number of tiles in (row, col) directions. Use "auto" (default) to
        automatically determine based on image size and available GPU memory.
        For explicit control, use e.g. (2, 2) for 4 tiles.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    np.ndarray
        Filtered complex interferogram of same shape.
    """
    import threading

    from rapidphase.device.manager import DeviceManager

    H, W = igram.shape

    # Determine number of GPUs to use
    available_gpus = _get_gpu_count()

    if n_gpus is not None:
        if n_gpus > available_gpus and available_gpus > 0:
            import warnings
            warnings.warn(
                f"Requested {n_gpus} GPUs but only {available_gpus} detected. "
                f"Using {available_gpus} GPUs.",
                UserWarning,
            )
        use_n_gpus = min(n_gpus, available_gpus) if available_gpus > 0 else 0
    else:
        # n_gpus=None -> use all available GPUs
        use_n_gpus = available_gpus

    # Determine tiling strategy
    if ntiles == "auto":
        ntiles = _auto_compute_ntiles(H, W, use_n_gpus, device)

    n_row_tiles, n_col_tiles = ntiles
    n_total_tiles = n_row_tiles * n_col_tiles

    # If only one tile, use single-device path directly
    if n_total_tiles == 1:
        dm = DeviceManager(device if use_n_gpus == 0 else "cuda")
        filt = GoldsteinFilter(
            dm,
            alpha=alpha,
            window_size=window_size,
            overlap=overlap,
            patch_batch_size=patch_batch_size,
        )
        return filt(igram, return_numpy=True)

    # Use tiled processing
    return _filter_tiled(
        igram,
        alpha=alpha,
        window_size=window_size,
        overlap=overlap,
        patch_batch_size=patch_batch_size,
        ntiles=ntiles,
        n_gpus=use_n_gpus,
        device=device,
        verbose=verbose,
    )


def _auto_compute_ntiles(
    H: int,
    W: int,
    n_gpus: int,
    device: str,
) -> tuple[int, int]:
    """
    Automatically compute number of tiles based on image size and GPU memory.

    Targets tile sizes that fit comfortably in GPU memory with safety margin.
    """
    # Estimate memory per pixel for Goldstein filter
    # Complex input + FFT workspace + accumulator buffers
    # ~40 bytes per pixel (conservative estimate)
    bytes_per_pixel = 40

    # Get available GPU memory (use first GPU as reference)
    if n_gpus > 0 and torch.cuda.is_available():
        try:
            # Get free memory on first GPU
            free_memory = torch.cuda.mem_get_info(0)[0]
        except Exception:
            # Fallback: assume 8GB available
            free_memory = 8 * 1024**3
    else:
        # CPU: use 4GB as conservative limit
        free_memory = 4 * 1024**3

    # Use only 50% of available memory for safety
    usable_memory = free_memory * 0.5

    # Calculate max pixels per tile
    max_pixels_per_tile = int(usable_memory / bytes_per_pixel)

    # Current image size
    total_pixels = H * W

    # If image fits in memory, use 1x1 tiling (or n_gpus horizontal strips)
    if total_pixels <= max_pixels_per_tile:
        if n_gpus > 1:
            # Use horizontal strips for multi-GPU
            return (n_gpus, 1)
        return (1, 1)

    # Calculate number of tiles needed
    tiles_needed = max(1, int(np.ceil(total_pixels / max_pixels_per_tile)))

    # Try to balance tiles across both dimensions
    # Prefer aspect ratio similar to original image
    aspect_ratio = H / W if W > 0 else 1.0

    n_row_tiles = max(1, int(np.sqrt(tiles_needed * aspect_ratio)))
    n_col_tiles = max(1, int(np.ceil(tiles_needed / n_row_tiles)))

    # Ensure we have enough tiles
    while n_row_tiles * n_col_tiles < tiles_needed:
        if n_row_tiles <= n_col_tiles:
            n_row_tiles += 1
        else:
            n_col_tiles += 1

    # If using multiple GPUs, adjust to distribute evenly
    if n_gpus > 1:
        total_tiles = n_row_tiles * n_col_tiles
        # Round up to multiple of n_gpus for even distribution
        if total_tiles % n_gpus != 0:
            extra = n_gpus - (total_tiles % n_gpus)
            # Add extra tiles to the smaller dimension
            if n_row_tiles <= n_col_tiles:
                n_row_tiles += int(np.ceil(extra / n_col_tiles))
            else:
                n_col_tiles += int(np.ceil(extra / n_row_tiles))

    return (n_row_tiles, n_col_tiles)


def _filter_tiled(
    igram: np.ndarray,
    alpha: float,
    window_size: int,
    overlap: float,
    patch_batch_size: int,
    ntiles: tuple[int, int],
    n_gpus: int,
    device: str,
    verbose: bool,
) -> np.ndarray:
    """
    Apply Goldstein filter using tiled processing.

    Keeps full image on CPU, transfers tiles to GPU one at a time.
    Supports multi-GPU parallel processing.
    """
    import threading

    from rapidphase.device.manager import DeviceManager

    H, W = igram.shape
    n_row_tiles, n_col_tiles = ntiles
    n_total_tiles = n_row_tiles * n_col_tiles

    # Tile overlap size (use 2x window_size for smooth blending)
    tile_overlap = window_size * 2

    # Compute tile boundaries
    tiles = _compute_tile_boundaries(H, W, n_row_tiles, n_col_tiles, tile_overlap)

    if verbose:
        if n_gpus > 1:
            print(f"Goldstein filtering: {n_total_tiles} tiles ({n_row_tiles}x{n_col_tiles}) on {n_gpus} GPUs...")
        else:
            device_name = device if n_gpus == 0 else "cuda"
            print(f"Goldstein filtering: {n_total_tiles} tiles ({n_row_tiles}x{n_col_tiles}) on {device_name}...")

    if n_gpus > 1:
        # Multi-GPU parallel processing
        return _filter_tiled_multi_gpu(
            igram, tiles, alpha, window_size, overlap, patch_batch_size,
            n_gpus, tile_overlap, verbose
        )
    else:
        # Single-device sequential processing
        return _filter_tiled_single_device(
            igram, tiles, alpha, window_size, overlap, patch_batch_size,
            device, tile_overlap, verbose
        )


def _compute_tile_boundaries(
    H: int,
    W: int,
    n_row_tiles: int,
    n_col_tiles: int,
    tile_overlap: int,
) -> list[dict]:
    """Compute tile boundaries with overlap."""
    base_tile_h = H // n_row_tiles
    base_tile_w = W // n_col_tiles

    tiles = []

    for i in range(n_row_tiles):
        for j in range(n_col_tiles):
            # Data region (what this tile is responsible for)
            data_row_start = i * base_tile_h
            data_row_end = (i + 1) * base_tile_h if i < n_row_tiles - 1 else H
            data_col_start = j * base_tile_w
            data_col_end = (j + 1) * base_tile_w if j < n_col_tiles - 1 else W

            # Extended region with overlap
            row_start = max(0, data_row_start - tile_overlap)
            row_end = min(H, data_row_end + tile_overlap)
            col_start = max(0, data_col_start - tile_overlap)
            col_end = min(W, data_col_end + tile_overlap)

            tiles.append({
                'row_idx': i,
                'col_idx': j,
                'row_start': row_start,
                'row_end': row_end,
                'col_start': col_start,
                'col_end': col_end,
                'data_row_start': data_row_start,
                'data_row_end': data_row_end,
                'data_col_start': data_col_start,
                'data_col_end': data_col_end,
            })

    return tiles


def _filter_tiled_single_device(
    igram: np.ndarray,
    tiles: list[dict],
    alpha: float,
    window_size: int,
    overlap: float,
    patch_batch_size: int,
    device: str,
    tile_overlap: int,
    verbose: bool,
) -> np.ndarray:
    """Process tiles sequentially on a single device."""
    from rapidphase.device.manager import DeviceManager

    H, W = igram.shape
    n_tiles = len(tiles)

    # Create device manager and filter
    dm = DeviceManager(device)
    filt = GoldsteinFilter(
        dm,
        alpha=alpha,
        window_size=window_size,
        overlap=overlap,
        patch_batch_size=patch_batch_size,
    )

    # Process tiles
    processed_tiles = []

    # Set up progress iteration
    if verbose and HAS_TQDM:
        tile_iter = tqdm(
            enumerate(tiles),
            total=n_tiles,
            desc=f"Filtering tiles ({n_tiles})",
            unit="tile",
        )
    else:
        tile_iter = enumerate(tiles)

    for idx, tile in tile_iter:
        if verbose and not HAS_TQDM:
            print(f"  Tile {idx + 1}/{n_tiles}...")

        # Extract tile data (stays on CPU)
        tile_data = igram[
            tile['row_start']:tile['row_end'],
            tile['col_start']:tile['col_end']
        ].copy()

        # Process on GPU
        filtered_tile = filt(tile_data, return_numpy=True)

        # Store result
        processed_tiles.append({
            'data': filtered_tile,
            'info': tile,
        })

        # Clear GPU cache
        dm.clear_cache()

    if verbose and not HAS_TQDM:
        print("  Merging tiles...")

    # Merge tiles
    return _merge_tiles(processed_tiles, (H, W))


def _filter_tiled_multi_gpu(
    igram: np.ndarray,
    tiles: list[dict],
    alpha: float,
    window_size: int,
    overlap: float,
    patch_batch_size: int,
    n_gpus: int,
    tile_overlap: int,
    verbose: bool,
) -> np.ndarray:
    """Process tiles in parallel across multiple GPUs."""
    import threading

    from rapidphase.device.manager import DeviceManager

    H, W = igram.shape
    n_tiles = len(tiles)
    gpu_ids = list(range(n_gpus))

    # Group tiles by GPU (round-robin assignment)
    tiles_per_gpu: list[list[tuple[int, dict]]] = [[] for _ in range(n_gpus)]
    for i, tile in enumerate(tiles):
        tiles_per_gpu[i % n_gpus].append((i, tile))

    # Results storage
    results: list[dict | None] = [None] * n_tiles

    # Progress tracking
    progress_lock = threading.Lock()
    completed_count = [0]

    # Create tqdm progress bar if available
    pbar = None
    if verbose and HAS_TQDM:
        pbar = tqdm(
            total=n_tiles,
            desc=f"Filtering ({n_gpus} GPUs)",
            unit="tile",
        )

    def process_gpu_tiles(gpu_id: int, tile_list: list[tuple[int, dict]]):
        """Worker function to process tiles on a specific GPU."""
        # Create device manager and filter for this GPU
        dm = DeviceManager(f"cuda:{gpu_id}")
        filt = GoldsteinFilter(
            dm,
            alpha=alpha,
            window_size=window_size,
            overlap=overlap,
            patch_batch_size=patch_batch_size,
        )

        for tile_idx, tile in tile_list:
            # Extract tile data (stays on CPU)
            tile_data = igram[
                tile['row_start']:tile['row_end'],
                tile['col_start']:tile['col_end']
            ].copy()

            # Process on GPU
            filtered_tile = filt(tile_data, return_numpy=True)

            # Store result
            results[tile_idx] = {
                'data': filtered_tile,
                'info': tile,
            }

            # Update progress
            with progress_lock:
                completed_count[0] += 1
                if pbar is not None:
                    pbar.update(1)
                elif verbose:
                    print(f"  Tile {completed_count[0]}/{n_tiles} (GPU {gpu_id})")

        # Clean up GPU memory
        dm.clear_cache()

    # Process tiles in parallel
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = [
            executor.submit(process_gpu_tiles, gpu_ids[i], tiles_per_gpu[i])
            for i in range(n_gpus)
        ]
        for future in futures:
            future.result()

    if pbar is not None:
        pbar.close()

    if verbose and not HAS_TQDM:
        print("  Merging tiles...")

    # Convert to list and merge
    processed_tiles = [r for r in results if r is not None]
    return _merge_tiles(processed_tiles, (H, W))


def _merge_tiles(
    processed_tiles: list[dict],
    shape: tuple[int, int],
) -> np.ndarray:
    """
    Merge filtered tiles with smooth blending.

    Uses cosine ramp blending in overlap regions based on DATA boundaries.
    """
    H, W = shape

    # Determine output dtype from first result
    output_dtype = processed_tiles[0]['data'].dtype

    # Accumulators
    output = np.zeros((H, W), dtype=output_dtype)
    weight_sum = np.zeros((H, W), dtype=np.float64)

    for result in processed_tiles:
        tile_data = result['data']
        info = result['info']

        tile_H, tile_W = tile_data.shape

        # Create 2D blend weights
        weights = np.ones((tile_H, tile_W), dtype=np.float64)

        # Top overlap region: ramp from 0 at row_start to 1 at data_row_start
        top_overlap = info['data_row_start'] - info['row_start']
        if top_overlap > 0:
            t = np.linspace(0, 1, top_overlap + 1)[1:]
            ramp = 0.5 * (1 - np.cos(t * np.pi))
            weights[:top_overlap, :] *= ramp[:, np.newaxis]

        # Bottom overlap region: ramp from 1 at data_row_end to 0 at row_end
        bottom_overlap = info['row_end'] - info['data_row_end']
        if bottom_overlap > 0:
            t = np.linspace(0, 1, bottom_overlap + 1)[1:]
            ramp = 0.5 * (1 - np.cos(t * np.pi))
            weights[-bottom_overlap:, :] *= ramp[::-1, np.newaxis]

        # Left overlap region: ramp from 0 at col_start to 1 at data_col_start
        left_overlap = info['data_col_start'] - info['col_start']
        if left_overlap > 0:
            t = np.linspace(0, 1, left_overlap + 1)[1:]
            ramp = 0.5 * (1 - np.cos(t * np.pi))
            weights[:, :left_overlap] *= ramp[np.newaxis, :]

        # Right overlap region: ramp from 1 at data_col_end to 0 at col_end
        right_overlap = info['col_end'] - info['data_col_end']
        if right_overlap > 0:
            t = np.linspace(0, 1, right_overlap + 1)[1:]
            ramp = 0.5 * (1 - np.cos(t * np.pi))
            weights[:, -right_overlap:] *= ramp[::-1][np.newaxis, :]

        # Handle NaN values
        nan_mask = np.isnan(tile_data)
        effective_weights = weights.copy()
        effective_weights[nan_mask] = 0.0

        safe_data = tile_data.copy()
        safe_data[nan_mask] = 0.0

        # Accumulate
        output[info['row_start']:info['row_end'], info['col_start']:info['col_end']] += (
            effective_weights * safe_data
        )
        weight_sum[info['row_start']:info['row_end'], info['col_start']:info['col_end']] += (
            effective_weights
        )

    # Normalize by weight sum
    valid_output = weight_sum > 1e-10
    output[valid_output] = output[valid_output] / weight_sum[valid_output]
    output[~valid_output] = np.nan

    return output
