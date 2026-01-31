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

    def _extract_patches(
        self, data: torch.Tensor, step: int
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """
        Extract overlapping patches from the input.

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
        positions : list of tuple
            (row, col) position of each patch's top-left corner.
        """
        H, W = data.shape
        ws = self.window_size
        patches = []
        positions = []

        for i in range(0, H - ws + 1, step):
            for j in range(0, W - ws + 1, step):
                patch = data[i : i + ws, j : j + ws]
                patches.append(patch)
                positions.append((i, j))

        if not patches:
            # Image smaller than window - pad and process as single patch
            padded = torch.zeros(ws, ws, dtype=data.dtype, device=data.device)
            padded[:H, :W] = data
            return padded.unsqueeze(0), [(0, 0)]

        return torch.stack(patches), positions

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

        # Extract patches
        patches, positions = self._extract_patches(padded_data, step)
        mask_patches, _ = self._extract_patches(padded_mask.to(self.dm.dtype), step)
        mask_patches = mask_patches > 0.5  # Convert back to bool

        # Determine complex dtype for output
        complex_dtype = torch.complex64 if self.dm.dtype == torch.float32 else torch.complex128

        # Initialize output accumulators
        output = torch.zeros(pH, pW, dtype=complex_dtype, device=self.dm.device)
        weights = torch.zeros(pH, pW, dtype=self.dm.dtype, device=self.dm.device)

        window_2d = self._window_2d.to(complex_dtype)
        num_patches = len(positions)

        # Process patches in batches to manage GPU memory
        for batch_start in range(0, num_patches, self.patch_batch_size):
            batch_end = min(batch_start + self.patch_batch_size, num_patches)

            # Get batch of patches
            batch_patches = patches[batch_start:batch_end]
            batch_masks = mask_patches[batch_start:batch_end]

            # Filter this batch
            filtered_batch = self._filter_patches_batched(batch_patches, batch_masks)

            # Accumulate weighted patches from this batch
            for local_idx, global_idx in enumerate(range(batch_start, batch_end)):
                i, j = positions[global_idx]

                # Apply window weighting
                weighted_patch = filtered_batch[local_idx] * window_2d
                weight_patch = self._window_2d * (~batch_masks[local_idx]).to(self.dm.dtype)

                # Accumulate
                output[i : i + self.window_size, j : j + self.window_size] += weighted_patch
                weights[i : i + self.window_size, j : j + self.window_size] += weight_patch

            # Free memory from this batch
            del filtered_batch, batch_patches, batch_masks
            if self.dm.device.type == 'cuda':
                torch.cuda.empty_cache()

        # Normalize by weights
        valid_weights = weights > 1e-10
        output[valid_weights] = output[valid_weights] / weights[valid_weights]

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
    verbose: bool = True,
) -> np.ndarray:
    """
    Apply Goldstein filter using multiple GPUs in parallel.

    Splits the image into horizontal strips, processes each strip on a
    separate GPU, and merges the results with smooth blending.

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

    # Fall back to single device if multi-GPU not available
    if use_n_gpus <= 1:
        dm = DeviceManager(device)
        filt = GoldsteinFilter(
            dm,
            alpha=alpha,
            window_size=window_size,
            overlap=overlap,
            patch_batch_size=patch_batch_size,
        )
        return filt(igram, return_numpy=True)

    gpu_ids = list(range(use_n_gpus))

    if verbose:
        print(f"Goldstein filtering on {use_n_gpus} GPUs (IDs: {gpu_ids}) in parallel...")

    # Calculate strip overlap (use window_size for smooth blending)
    strip_overlap = window_size * 2

    # Calculate strip boundaries
    base_strip_height = H // use_n_gpus
    strips = []

    for i in range(use_n_gpus):
        # Data region (what this strip is responsible for)
        data_row_start = i * base_strip_height
        data_row_end = (i + 1) * base_strip_height if i < use_n_gpus - 1 else H

        # Extended region with overlap
        row_start = max(0, data_row_start - strip_overlap)
        row_end = min(H, data_row_end + strip_overlap)

        strips.append({
            'gpu_id': gpu_ids[i],
            'row_start': row_start,
            'row_end': row_end,
            'data_row_start': data_row_start,
            'data_row_end': data_row_end,
        })

    # Results storage
    results: list[dict | None] = [None] * use_n_gpus

    # Progress tracking
    progress_lock = threading.Lock()
    completed_count = [0]

    def process_strip(strip_idx: int, strip_info: dict):
        """Worker function to process a strip on a specific GPU."""
        gpu_id = strip_info['gpu_id']

        # Create device manager and filter for this GPU
        dm = DeviceManager(f"cuda:{gpu_id}")
        filt = GoldsteinFilter(
            dm,
            alpha=alpha,
            window_size=window_size,
            overlap=overlap,
            patch_batch_size=patch_batch_size,
        )

        # Extract strip data
        strip_data = igram[strip_info['row_start']:strip_info['row_end'], :].copy()

        # Apply filter
        filtered_strip = filt(strip_data, return_numpy=True)

        # Store result with metadata
        results[strip_idx] = {
            'data': filtered_strip,
            'info': strip_info,
        }

        # Clean up GPU memory
        dm.clear_cache()

        # Update progress
        with progress_lock:
            completed_count[0] += 1
            if verbose:
                print(f"  Strip {completed_count[0]}/{use_n_gpus} complete (GPU {gpu_id})")

    # Process strips in parallel
    with ThreadPoolExecutor(max_workers=use_n_gpus) as executor:
        futures = [
            executor.submit(process_strip, i, strips[i])
            for i in range(use_n_gpus)
        ]
        for future in futures:
            future.result()

    if verbose:
        print("Merging strips...")

    # Merge strips with smooth blending
    output = _merge_strips(results, (H, W), strip_overlap)

    if verbose:
        print("Done.")

    return output


def _merge_strips(
    results: list[dict],
    shape: tuple[int, int],
    overlap: int,
) -> np.ndarray:
    """
    Merge filtered strips with smooth blending in overlap regions.

    Parameters
    ----------
    results : list of dict
        List of {'data': filtered_strip, 'info': strip_info} dicts.
    shape : tuple of int
        Output image shape (H, W).
    overlap : int
        Overlap size in pixels.

    Returns
    -------
    np.ndarray
        Merged filtered image.
    """
    H, W = shape

    # Determine output dtype from first result
    output_dtype = results[0]['data'].dtype

    # Accumulators
    output = np.zeros((H, W), dtype=output_dtype)
    weight_sum = np.zeros((H, W), dtype=np.float64)

    for result in results:
        strip_data = result['data']
        info = result['info']

        strip_H = strip_data.shape[0]

        # Create blend weights for this strip
        weights = np.ones(strip_H, dtype=np.float64)

        # Top overlap region: ramp from 0 to 1
        top_overlap = info['data_row_start'] - info['row_start']
        if top_overlap > 0:
            t = np.linspace(0, 1, top_overlap + 1)[1:]
            ramp = 0.5 * (1 - np.cos(t * np.pi))
            weights[:top_overlap] = ramp

        # Bottom overlap region: ramp from 1 to 0
        bottom_overlap = info['row_end'] - info['data_row_end']
        if bottom_overlap > 0:
            t = np.linspace(0, 1, bottom_overlap + 1)[1:]
            ramp = 0.5 * (1 - np.cos(t * np.pi))
            weights[-bottom_overlap:] = ramp[::-1]

        # Expand weights to 2D
        weights_2d = weights[:, np.newaxis] * np.ones((1, W))

        # Handle NaN values
        nan_mask = np.isnan(strip_data)
        effective_weights = weights_2d.copy()
        effective_weights[nan_mask] = 0.0

        safe_data = strip_data.copy()
        safe_data[nan_mask] = 0.0

        # Accumulate
        output[info['row_start']:info['row_end'], :] += effective_weights * safe_data
        weight_sum[info['row_start']:info['row_end'], :] += effective_weights

    # Normalize by weight sum
    valid_output = weight_sum > 1e-10
    output[valid_output] = output[valid_output] / weight_sum[valid_output]
    output[~valid_output] = np.nan

    return output
