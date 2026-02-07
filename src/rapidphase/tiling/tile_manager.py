"""
Tile manager for processing large images.

Splits large interferograms into overlapping tiles for memory-efficient
GPU processing, then merges results with phase alignment and feathered blending.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

# Try to import tqdm for progress bars, fall back to simple iteration
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

if TYPE_CHECKING:
    from rapidphase.device.manager import DeviceManager


@dataclass
class TileInfo:
    """Information about a single tile."""

    row_idx: int
    col_idx: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    # Actual data region (excluding overlap padding)
    data_row_start: int
    data_row_end: int
    data_col_start: int
    data_col_end: int


class TileManager:
    """
    Manages tiling and merging for large image processing.

    Splits images into overlapping tiles, processes them (potentially
    in parallel on GPU), aligns phase offsets between tiles, and merges
    with smooth blending to avoid discontinuities at tile boundaries.
    """

    def __init__(
        self,
        device_manager: DeviceManager,
        ntiles: tuple[int, int] = (2, 2),
        overlap: int = 64,
    ):
        """
        Initialize the tile manager.

        Parameters
        ----------
        device_manager : DeviceManager
            Device manager for GPU operations.
        ntiles : tuple of int
            Number of tiles in (row, col) directions.
        overlap : int
            Overlap in pixels between adjacent tiles.
        """
        self.dm = device_manager
        self.ntiles = ntiles
        self.overlap = overlap

    def _get_gpu_count(self) -> int:
        """
        Get number of available GPUs using torch.cuda.

        Returns
        -------
        int
            Number of available CUDA GPUs (0 if CUDA not available).
        """
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 0

    def compute_tiles(
        self,
        shape: tuple[int, int],
    ) -> list[TileInfo]:
        """
        Compute tile boundaries for a given image shape.

        Parameters
        ----------
        shape : tuple of int
            Image shape (H, W).

        Returns
        -------
        list of TileInfo
            List of tile information objects.
        """
        H, W = shape
        n_rows, n_cols = self.ntiles

        # Base tile sizes (without overlap)
        base_tile_h = H // n_rows
        base_tile_w = W // n_cols

        tiles = []

        for i in range(n_rows):
            for j in range(n_cols):
                # Data region (what this tile is responsible for)
                data_row_start = i * base_tile_h
                data_row_end = (i + 1) * base_tile_h if i < n_rows - 1 else H
                data_col_start = j * base_tile_w
                data_col_end = (j + 1) * base_tile_w if j < n_cols - 1 else W

                # Extended region with overlap
                row_start = max(0, data_row_start - self.overlap)
                row_end = min(H, data_row_end + self.overlap)
                col_start = max(0, data_col_start - self.overlap)
                col_end = min(W, data_col_end + self.overlap)

                tiles.append(TileInfo(
                    row_idx=i,
                    col_idx=j,
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                    data_row_start=data_row_start,
                    data_row_end=data_row_end,
                    data_col_start=data_col_start,
                    data_col_end=data_col_end,
                ))

        return tiles

    def extract_tile(
        self,
        image: torch.Tensor,
        tile: TileInfo,
    ) -> torch.Tensor:
        """
        Extract a tile from an image.

        Parameters
        ----------
        image : torch.Tensor
            Full image of shape (H, W).
        tile : TileInfo
            Tile information.

        Returns
        -------
        torch.Tensor
            Extracted tile.
        """
        return image[tile.row_start:tile.row_end, tile.col_start:tile.col_end].clone()

    def _align_tile_phases(
        self,
        tiles_data: list[tuple[TileInfo, torch.Tensor]],
    ) -> list[tuple[TileInfo, torch.Tensor]]:
        """
        Align phase offsets between adjacent tiles.

        Each tile from phase unwrapping has an arbitrary constant offset
        (integer multiple of 2*pi). This method computes and applies
        corrections so tiles have consistent phase values.

        Uses priority BFS: tiles with more valid overlap pixels are
        aligned first to avoid propagating errors through NaN-heavy links.

        Parameters
        ----------
        tiles_data : list of (TileInfo, Tensor)
            List of (tile_info, unwrapped_phase) pairs.

        Returns
        -------
        list of (TileInfo, Tensor)
            Aligned tiles with consistent phase offsets.
        """
        if len(tiles_data) <= 1:
            return tiles_data

        import heapq
        import math

        n_rows, n_cols = self.ntiles
        two_pi = 2 * math.pi

        # Create a dictionary for easy lookup by (row, col) index
        tile_dict = {}
        for tile, data in tiles_data:
            tile_dict[(tile.row_idx, tile.col_idx)] = (tile, data.clone())

        # Pick reference tile: the one with the most valid (non-NaN) pixels
        ref_idx = max(
            tile_dict.keys(),
            key=lambda idx: int((~torch.isnan(tile_dict[idx][1])).sum().item()),
        )

        aligned = set()
        offsets = {ref_idx: 0.0}
        aligned.add(ref_idx)

        pq: list[tuple[int, tuple[int, int], float]] = []

        def add_neighbors(curr_idx):
            curr_tile, curr_data = tile_dict[curr_idx]
            curr_offset = offsets[curr_idx]
            neighbors = [
                (curr_idx[0], curr_idx[1] + 1),
                (curr_idx[0] + 1, curr_idx[1]),
                (curr_idx[0], curr_idx[1] - 1),
                (curr_idx[0] - 1, curr_idx[1]),
            ]
            for neighbor_idx in neighbors:
                if neighbor_idx in aligned or neighbor_idx not in tile_dict:
                    continue
                neighbor_tile, neighbor_data = tile_dict[neighbor_idx]

                # Full overlap region
                or_s = max(curr_tile.row_start, neighbor_tile.row_start)
                or_e = min(curr_tile.row_end, neighbor_tile.row_end)
                oc_s = max(curr_tile.col_start, neighbor_tile.col_start)
                oc_e = min(curr_tile.col_end, neighbor_tile.col_end)
                if or_s >= or_e or oc_s >= oc_e:
                    continue

                curr_overlap = curr_data[
                    or_s - curr_tile.row_start:or_e - curr_tile.row_start,
                    oc_s - curr_tile.col_start:oc_e - curr_tile.col_start
                ]
                neighbor_overlap = neighbor_data[
                    or_s - neighbor_tile.row_start:or_e - neighbor_tile.row_start,
                    oc_s - neighbor_tile.col_start:oc_e - neighbor_tile.col_start
                ]

                diff = (curr_overlap + curr_offset) - neighbor_overlap
                valid_mask = ~torch.isnan(diff)
                n_valid = int(valid_mask.sum().item())

                if n_valid > 0:
                    # Per-pixel rounding + mode: robust to DC offset differences
                    # between tiles. Each pixel's diff should be close to N*2pi,
                    # so round per-pixel then take the most common integer.
                    k_per_pixel = torch.round(diff[valid_mask] / two_pi).long()
                    k_min = k_per_pixel.min().item()
                    k_shifted = k_per_pixel - k_min
                    counts = torch.bincount(k_shifted)
                    k_mode = int(counts.argmax().item()) + k_min
                    offset = k_mode * two_pi
                    heapq.heappush(pq, (-n_valid, neighbor_idx, offset))

        add_neighbors(ref_idx)

        while pq:
            neg_nvalid, neighbor_idx, offset = heapq.heappop(pq)
            if neighbor_idx in aligned:
                continue
            offsets[neighbor_idx] = offset
            aligned.add(neighbor_idx)
            add_neighbors(neighbor_idx)

        # Apply offsets to all tiles
        aligned_tiles = []
        for tile, data in tiles_data:
            idx = (tile.row_idx, tile.col_idx)
            offset = offsets.get(idx, 0.0)
            aligned_data = data + offset
            aligned_tiles.append((tile, aligned_data))

        return aligned_tiles

    def create_blend_weights(
        self,
        tile_shape: tuple[int, int],
        tile: TileInfo,
        full_shape: tuple[int, int],
    ) -> torch.Tensor:
        """
        Create blending weights for a tile.

        Uses cosine feathering at boundaries to ensure smooth transitions.

        Parameters
        ----------
        tile_shape : tuple of int
            Shape of the tile (tile_H, tile_W).
        tile : TileInfo
            Tile information.
        full_shape : tuple of int
            Shape of the full image (H, W).

        Returns
        -------
        torch.Tensor
            Blend weights of shape (tile_H, tile_W).
        """
        tile_H, tile_W = tile_shape
        H, W = full_shape

        weights = self.dm.ones((tile_H, tile_W))

        # Feather width (use most of the overlap for smooth blending)
        feather = max(1, int(self.overlap * 0.8))

        # Top edge feathering
        if tile.row_start > 0:
            ramp = self._cosine_ramp(feather, tile_H)
            weights = weights * ramp.unsqueeze(1)

        # Bottom edge feathering
        if tile.row_end < H:
            ramp = self._cosine_ramp(feather, tile_H)
            ramp = torch.flip(ramp, dims=[0])
            weights = weights * ramp.unsqueeze(1)

        # Left edge feathering
        if tile.col_start > 0:
            ramp = self._cosine_ramp(feather, tile_W)
            weights = weights * ramp.unsqueeze(0)

        # Right edge feathering
        if tile.col_end < W:
            ramp = self._cosine_ramp(feather, tile_W)
            ramp = torch.flip(ramp, dims=[0])
            weights = weights * ramp.unsqueeze(0)

        return weights

    def _cosine_ramp(self, feather: int, length: int) -> torch.Tensor:
        """Create a cosine ramp from small positive value to 1."""
        ramp = self.dm.ones((length,))
        if feather > 0 and feather < length:
            # Use linspace from small positive value to 1 to avoid exact zeros
            t = torch.linspace(0, 1, feather + 1, dtype=self.dm.dtype, device=self.dm.device)[1:]
            # Cosine ramp: small value at start, 1 at end
            ramp[:feather] = 0.5 * (1 - torch.cos(t * 3.14159265))
        return ramp

    def merge_tiles(
        self,
        tiles_data: list[tuple[TileInfo, torch.Tensor]],
        shape: tuple[int, int],
        align_phases: bool = True,
    ) -> torch.Tensor:
        """
        Merge processed tiles back into a full image.

        First aligns phase offsets between adjacent tiles, then uses
        weighted averaging with feathered blend weights.

        Parameters
        ----------
        tiles_data : list of (TileInfo, Tensor)
            List of (tile_info, processed_tile) pairs.
        shape : tuple of int
            Output image shape (H, W).
        align_phases : bool
            If True, align phase offsets between tiles before merging.

        Returns
        -------
        torch.Tensor
            Merged image of shape (H, W).
        """
        H, W = shape

        # Align phase offsets between tiles
        if align_phases and len(tiles_data) > 1:
            tiles_data = self._align_tile_phases(tiles_data)

        # Accumulators for weighted average
        result = self.dm.zeros((H, W))
        weight_sum = self.dm.zeros((H, W))

        for tile, data in tiles_data:
            tile_shape = data.shape
            weights = self.create_blend_weights(tile_shape, tile, shape)

            # Create mask for valid (non-NaN) pixels
            valid_mask = ~torch.isnan(data)

            # Zero out weights for NaN pixels
            effective_weights = weights.clone()
            effective_weights[~valid_mask] = 0.0

            # Replace NaN with 0 for accumulation (will be masked by weights)
            safe_data = data.clone()
            safe_data[~valid_mask] = 0.0

            # Add weighted contribution
            result[tile.row_start:tile.row_end, tile.col_start:tile.col_end] += (
                effective_weights * safe_data
            )
            weight_sum[tile.row_start:tile.row_end, tile.col_start:tile.col_end] += (
                effective_weights
            )

        # Normalize by weight sum
        # Pixels with zero weight sum had no valid data - set to NaN
        valid_output = weight_sum > 1e-10
        result[valid_output] = result[valid_output] / weight_sum[valid_output]
        result[~valid_output] = float('nan')

        return result

    def process_tiled(
        self,
        image: torch.Tensor,
        process_fn: Callable[[torch.Tensor], torch.Tensor],
        coherence: torch.Tensor | None = None,
        nan_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Process an image tile-by-tile.

        Parameters
        ----------
        image : torch.Tensor
            Input image of shape (H, W).
        process_fn : callable
            Function to apply to each tile.
            Should take (tile, coherence_tile, nan_mask_tile) and return processed tile.
        coherence : torch.Tensor, optional
            Coherence map of shape (H, W).
        nan_mask : torch.Tensor, optional
            Boolean mask of shape (H, W) indicating invalid pixels.

        Returns
        -------
        torch.Tensor
            Processed image of shape (H, W).
        """
        H, W = image.shape
        tiles = self.compute_tiles((H, W))

        processed_tiles = []

        for tile in tiles:
            # Extract tile
            tile_data = self.extract_tile(image, tile)

            # Extract coherence tile if available
            coh_tile = None
            if coherence is not None:
                coh_tile = self.extract_tile(coherence, tile)

            # Extract nan_mask tile if available
            nan_mask_tile = None
            if nan_mask is not None:
                nan_mask_tile = self.extract_tile(nan_mask, tile)

            # Process tile with available arguments
            if coh_tile is not None and nan_mask_tile is not None:
                result = process_fn(tile_data, coh_tile, nan_mask_tile)
            elif coh_tile is not None:
                result = process_fn(tile_data, coh_tile)
            elif nan_mask_tile is not None:
                result = process_fn(tile_data, nan_mask_tile)
            else:
                result = process_fn(tile_data)

            processed_tiles.append((tile, result))

        # Merge results with phase alignment
        return self.merge_tiles(processed_tiles, (H, W), align_phases=True)

    def process_tiled_numpy(
        self,
        image: np.ndarray,
        unwrapper_factory: Callable,
        coherence: np.ndarray | None = None,
        nan_mask: np.ndarray | None = None,
        n_gpus: int | None = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Process an image tile-by-tile with memory-efficient numpy I/O.

        This method keeps the full image in CPU memory (as numpy arrays) and only
        transfers individual tiles to GPU for processing. This enables processing
        of images much larger than GPU memory.

        Automatically uses all available CUDA GPUs for parallel processing.

        Parameters
        ----------
        image : np.ndarray
            Input image of shape (H, W) as numpy array.
        unwrapper_factory : callable
            Factory function that takes a DeviceManager and returns an unwrapper.
            This allows creating device-specific unwrappers for multi-GPU processing.
        coherence : np.ndarray, optional
            Coherence map of shape (H, W) as numpy array.
        nan_mask : np.ndarray, optional
            Boolean mask of shape (H, W) indicating invalid pixels.
        n_gpus : int, optional
            Number of GPUs to use. If None, uses all available GPUs.
            Set explicitly on HPC systems where not all GPUs may be allocated.
        verbose : bool
            If True, print progress information.

        Returns
        -------
        np.ndarray
            Processed image of shape (H, W) as numpy array.
        """
        H, W = image.shape
        tiles = self.compute_tiles((H, W))
        n_tiles = len(tiles)

        # Determine number of GPUs to use
        # Priority: n_gpus parameter > torch.cuda.device_count()
        available_gpus = self._get_gpu_count()

        if n_gpus is not None:
            # User explicitly specified GPU count - trust it but cap at available
            if n_gpus > available_gpus and available_gpus > 0:
                import warnings
                warnings.warn(
                    f"Requested {n_gpus} GPUs but only {available_gpus} detected. "
                    f"Using {available_gpus} GPUs.",
                    UserWarning,
                )
            use_n_gpus = min(n_gpus, available_gpus) if available_gpus > 0 else 0
        else:
            # Auto-detect using torch.cuda
            use_n_gpus = available_gpus

        use_gpu_ids = list(range(use_n_gpus))

        if use_n_gpus > 1 and self.dm.device_type == "cuda":
            # Multi-GPU parallel processing
            return self._process_tiled_multi_gpu(
                image, unwrapper_factory, coherence, nan_mask, tiles, use_gpu_ids, verbose
            )
        else:
            # Single device processing
            return self._process_tiled_single_device(
                image, unwrapper_factory, coherence, nan_mask, tiles, verbose
            )

    def _process_tiled_single_device(
        self,
        image: np.ndarray,
        unwrapper_factory: Callable,
        coherence: np.ndarray | None,
        nan_mask: np.ndarray | None,
        tiles: list[TileInfo],
        verbose: bool,
    ) -> np.ndarray:
        """Single-device tile processing."""
        H, W = image.shape
        n_tiles = len(tiles)

        # Create unwrapper for this device
        unwrapper = unwrapper_factory(self.dm)

        # Store processed tiles as numpy arrays (CPU memory)
        processed_tiles_np: list[tuple[TileInfo, np.ndarray]] = []

        # Set up progress iteration
        if verbose and HAS_TQDM:
            tile_iter = tqdm(
                enumerate(tiles),
                total=n_tiles,
                desc=f"Unwrapping tiles ({self.ntiles[0]}x{self.ntiles[1]})",
                unit="tile",
            )
        elif verbose:
            print(f"Processing {n_tiles} tiles ({self.ntiles[0]}x{self.ntiles[1]})...")
            tile_iter = enumerate(tiles)
        else:
            tile_iter = enumerate(tiles)

        for i, tile in tile_iter:
            if verbose and not HAS_TQDM and (i % max(1, n_tiles // 10) == 0 or i == n_tiles - 1):
                print(f"  Tile {i + 1}/{n_tiles} ({100 * (i + 1) // n_tiles}%)")

            # Extract tile from numpy array (stays on CPU)
            tile_data_np = image[tile.row_start:tile.row_end, tile.col_start:tile.col_end].copy()

            # Extract coherence tile if available
            coh_tile_np = None
            if coherence is not None:
                coh_tile_np = coherence[tile.row_start:tile.row_end, tile.col_start:tile.col_end].copy()

            # Extract nan_mask tile if available
            nan_mask_tile_np = None
            if nan_mask is not None:
                nan_mask_tile_np = nan_mask[tile.row_start:tile.row_end, tile.col_start:tile.col_end].copy()

            # Convert tile to GPU tensor
            tile_data_t = self.dm.to_tensor(tile_data_np)
            coh_tile_t = self.dm.to_tensor(coh_tile_np) if coh_tile_np is not None else None
            nan_mask_tile_t = None
            if nan_mask_tile_np is not None:
                nan_mask_tile_t = self.dm.to_tensor(nan_mask_tile_np.astype(np.float32)) > 0.5

            # Process tile on GPU
            # For tiled processing, get continuous (pre-congruence) solution.
            # Congruence projection is applied after merging to avoid per-tile
            # 2pi transition inconsistencies at tile boundaries.
            result_t = unwrapper.unwrap(
                tile_data_t, coh_tile_t, nan_mask=nan_mask_tile_t,
                return_continuous=True,
            )

            # Move result back to CPU numpy and store
            result_np = self.dm.to_numpy(result_t)
            processed_tiles_np.append((tile, result_np))

            # Clear GPU tensors to free memory (but not too frequently - it's expensive)
            del tile_data_t, coh_tile_t, nan_mask_tile_t, result_t
            # Only clear cache every 4 tiles or at the end
            if (i + 1) % 4 == 0 or i == n_tiles - 1:
                self.dm.clear_cache()

        if verbose and not HAS_TQDM:
            print("  Merging tiles with phase alignment...")

        # Merge continuous solutions, then apply congruence projection on
        # the full merged result for globally consistent 2pi transitions.
        result = self._merge_tiles_numpy(
            processed_tiles_np, (H, W), verbose=verbose,
            wrapped_phase=image, coherence=coherence, nan_mask=nan_mask,
        )

        if verbose and not HAS_TQDM:
            print("  Done.")

        return result

    def _process_tiled_multi_gpu(
        self,
        image: np.ndarray,
        unwrapper_factory: Callable,
        coherence: np.ndarray | None,
        nan_mask: np.ndarray | None,
        tiles: list[TileInfo],
        gpu_ids: list[int],
        verbose: bool,
    ) -> np.ndarray:
        """Multi-GPU parallel tile processing."""
        from rapidphase.device.manager import DeviceManager
        import threading

        H, W = image.shape
        n_tiles = len(tiles)
        n_gpus = len(gpu_ids)

        if verbose:
            print(f"Processing {n_tiles} tiles on {n_gpus} GPUs (IDs: {gpu_ids}) in parallel...")

        # Group tiles by GPU (round-robin assignment)
        tiles_per_gpu: list[list[tuple[int, TileInfo]]] = [[] for _ in range(n_gpus)]
        for i, tile in enumerate(tiles):
            tiles_per_gpu[i % n_gpus].append((i, tile))

        # Results storage (thread-safe via list index assignment)
        results: list[tuple[TileInfo, np.ndarray] | None] = [None] * n_tiles

        # Progress tracking
        progress_lock = threading.Lock()
        completed_count = [0]  # Use list to allow modification in nested function

        # Progress bar for tqdm
        pbar = None
        if verbose and HAS_TQDM:
            pbar = tqdm(
                total=n_tiles,
                desc=f"Unwrapping ({self.ntiles[0]}x{self.ntiles[1]} tiles, {n_gpus} GPUs)",
                unit="tile",
            )

        def process_gpu_tiles(gpu_id: int, tile_list: list[tuple[int, TileInfo]]):
            """Worker function to process all tiles assigned to a specific GPU."""
            # Create device manager and unwrapper for this GPU
            dm = DeviceManager(f"cuda:{gpu_id}")
            unwrapper = unwrapper_factory(dm)

            for tile_idx, tile in tile_list:
                # Extract tile data
                tile_data_np = image[tile.row_start:tile.row_end, tile.col_start:tile.col_end].copy()

                coh_tile_np = None
                if coherence is not None:
                    coh_tile_np = coherence[tile.row_start:tile.row_end, tile.col_start:tile.col_end].copy()

                nan_mask_tile_np = None
                if nan_mask is not None:
                    nan_mask_tile_np = nan_mask[tile.row_start:tile.row_end, tile.col_start:tile.col_end].copy()

                # Convert to tensors on this GPU
                tile_data_t = dm.to_tensor(tile_data_np)
                coh_tile_t = dm.to_tensor(coh_tile_np) if coh_tile_np is not None else None
                nan_mask_tile_t = None
                if nan_mask_tile_np is not None:
                    nan_mask_tile_t = dm.to_tensor(nan_mask_tile_np.astype(np.float32)) > 0.5

                # Process tile using the GPU-specific unwrapper
                # Return continuous solution for global congruence projection
                result_t = unwrapper.unwrap(
                    tile_data_t, coh_tile_t, nan_mask=nan_mask_tile_t,
                    return_continuous=True,
                )

                # Move result back to CPU
                result_np = dm.to_numpy(result_t)

                # Store result
                results[tile_idx] = (tile, result_np)

                # Clean up GPU memory
                del tile_data_t, coh_tile_t, nan_mask_tile_t, result_t

                # Update progress
                with progress_lock:
                    completed_count[0] += 1
                    if pbar is not None:
                        pbar.update(1)
                    elif verbose and (completed_count[0] % max(1, n_tiles // 10) == 0 or completed_count[0] == n_tiles):
                        print(f"  Tile {completed_count[0]}/{n_tiles} ({100 * completed_count[0] // n_tiles}%)")

            # Final cache clear for this GPU
            dm.clear_cache()

        # Process tiles in parallel - one thread per GPU
        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            futures = [
                executor.submit(process_gpu_tiles, gpu_ids[i], tiles_per_gpu[i])
                for i in range(n_gpus)
            ]
            # Wait for all to complete
            for future in futures:
                future.result()

        if pbar is not None:
            pbar.close()

        # Convert to list of tuples
        processed_tiles_np = [r for r in results if r is not None]

        if verbose:
            print("Aligning tile phases...")

        # Merge continuous solutions, then apply congruence projection
        result = self._merge_tiles_numpy(
            processed_tiles_np, (H, W), verbose=verbose,
            wrapped_phase=image, coherence=coherence, nan_mask=nan_mask,
        )

        if verbose and not HAS_TQDM:
            print("  Done.")

        return result

    def _compute_overlap_offset_numpy(
        self,
        curr_tile: TileInfo,
        curr_data: np.ndarray,
        curr_offset: float,
        neighbor_tile: TileInfo,
        neighbor_data: np.ndarray,
        is_continuous: bool = False,
    ) -> tuple[float, int]:
        """
        Compute phase offset between two overlapping tiles.

        For continuous solutions (pre-congruence), uses median of the
        difference — no rounding needed since DC offset is arbitrary.

        For congruent solutions, uses per-pixel rounding + mode to find
        the correct integer multiple of 2π.

        Returns
        -------
        offset : float
            Phase offset to apply to neighbor tile.
        n_valid : int
            Number of valid overlap pixels used.
        """
        # Find overlapping region in global coordinates
        overlap_row_start = max(curr_tile.row_start, neighbor_tile.row_start)
        overlap_row_end = min(curr_tile.row_end, neighbor_tile.row_end)
        overlap_col_start = max(curr_tile.col_start, neighbor_tile.col_start)
        overlap_col_end = min(curr_tile.col_end, neighbor_tile.col_end)

        if overlap_row_start >= overlap_row_end or overlap_col_start >= overlap_col_end:
            return 0.0, 0

        # Subsample large overlaps for speed. For NISAR-scale tiles, overlaps
        # can be 1706 × 8532 = 14.5M pixels. We only need ~10K-50K samples
        # for a robust median/mode estimate.
        oh = overlap_row_end - overlap_row_start
        ow = overlap_col_end - overlap_col_start
        n_pixels = oh * ow
        max_samples = 50000
        if n_pixels > max_samples:
            # Stride to get ~max_samples pixels
            stride = max(1, int(np.sqrt(n_pixels / max_samples)))
        else:
            stride = 1

        # Extract overlap regions in local coordinates (with stride)
        curr_overlap = curr_data[
            overlap_row_start - curr_tile.row_start:overlap_row_end - curr_tile.row_start:stride,
            overlap_col_start - curr_tile.col_start:overlap_col_end - curr_tile.col_start:stride
        ]
        neighbor_overlap = neighbor_data[
            overlap_row_start - neighbor_tile.row_start:overlap_row_end - neighbor_tile.row_start:stride,
            overlap_col_start - neighbor_tile.col_start:overlap_col_end - neighbor_tile.col_start:stride
        ]

        diff = (curr_overlap + curr_offset) - neighbor_overlap
        valid_mask = ~np.isnan(diff)
        # Report full valid count (not subsampled) for priority ordering
        n_valid = int(valid_mask.sum()) * (stride * stride)

        if valid_mask.any():
            if is_continuous:
                # Continuous solutions differ by an arbitrary DC offset.
                # Median gives a robust estimate of this constant offset.
                offset = float(np.median(diff[valid_mask]))
            else:
                # Congruent solutions differ by integer multiples of 2π.
                # Per-pixel rounding + mode is robust to local disagreements.
                two_pi = 2 * np.pi
                k_per_pixel = np.round(diff[valid_mask] / two_pi).astype(np.int64)
                k_min = int(k_per_pixel.min())
                k_shifted = k_per_pixel - k_min
                counts = np.bincount(k_shifted)
                k_mode = int(np.argmax(counts)) + k_min
                offset = k_mode * two_pi
            return offset, n_valid
        else:
            return 0.0, 0

    def _align_tile_phases_numpy(
        self,
        tiles_data: list[tuple[TileInfo, np.ndarray]],
        is_continuous: bool = False,
    ) -> list[tuple[TileInfo, np.ndarray]]:
        """
        Align phase offsets between adjacent tiles (numpy/CPU version).

        For continuous solutions, aligns using median offset (arbitrary DC).
        For congruent solutions, aligns using mode-based 2π rounding.

        Uses BFS with priority: tiles with more valid overlap pixels are
        aligned first to avoid propagating errors through NaN-heavy links.
        """
        if len(tiles_data) <= 1:
            return tiles_data

        n_rows, n_cols = self.ntiles

        # Create a dictionary for easy lookup by (row, col) index
        # No copy needed: data is read-only during alignment, offsets are
        # tracked separately and applied at the end.
        tile_dict = {}
        for tile, data in tiles_data:
            tile_dict[(tile.row_idx, tile.col_idx)] = (tile, data)

        # Pick reference tile: the one with the most valid (non-NaN) pixels.
        # Use subsampled count for speed on large tiles.
        ref_idx = max(
            tile_dict.keys(),
            key=lambda idx: int(np.sum(~np.isnan(tile_dict[idx][1][::10, ::10]))),
        )

        # Track which tiles have been aligned
        aligned = set()
        offsets = {ref_idx: 0.0}  # Reference tile has zero offset
        aligned.add(ref_idx)

        # Priority queue: (negative n_valid for max-first, neighbor_idx, offset)
        import heapq
        pq: list[tuple[int, tuple[int, int], float]] = []

        def add_neighbors(curr_idx):
            """Add unaligned neighbors to the priority queue."""
            curr_tile, curr_data = tile_dict[curr_idx]
            curr_offset = offsets[curr_idx]
            neighbors = [
                (curr_idx[0], curr_idx[1] + 1),
                (curr_idx[0] + 1, curr_idx[1]),
                (curr_idx[0], curr_idx[1] - 1),
                (curr_idx[0] - 1, curr_idx[1]),
            ]
            for neighbor_idx in neighbors:
                if neighbor_idx in aligned or neighbor_idx not in tile_dict:
                    continue
                neighbor_tile, neighbor_data = tile_dict[neighbor_idx]
                offset, n_valid = self._compute_overlap_offset_numpy(
                    curr_tile, curr_data, curr_offset,
                    neighbor_tile, neighbor_data,
                    is_continuous=is_continuous,
                )
                if n_valid > 0:
                    # Use negative n_valid so heapq gives us the largest first
                    heapq.heappush(pq, (-n_valid, neighbor_idx, offset))

        add_neighbors(ref_idx)

        while pq:
            neg_nvalid, neighbor_idx, offset = heapq.heappop(pq)

            if neighbor_idx in aligned:
                continue  # Already aligned through a better path

            offsets[neighbor_idx] = offset
            aligned.add(neighbor_idx)
            add_neighbors(neighbor_idx)

        # Apply offsets to all tiles
        aligned_tiles = []
        for tile, data in tiles_data:
            idx = (tile.row_idx, tile.col_idx)
            offset = offsets.get(idx, 0.0)
            aligned_data = data + offset
            aligned_tiles.append((tile, aligned_data))

        return aligned_tiles

    def _merge_tiles_numpy(
        self,
        tiles_data: list[tuple[TileInfo, np.ndarray]],
        shape: tuple[int, int],
        verbose: bool = False,
        wrapped_phase: np.ndarray | None = None,
        coherence: np.ndarray | None = None,
        nan_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Merge processed tiles back into a full image (numpy/CPU version).

        When wrapped_phase is provided, tiles are treated as continuous
        (pre-congruence) solutions: aligned with median offset, merged with
        blend weights, then congruence projection + local consistency
        correction are applied on the full merged result.

        The blending strategy uses the DATA boundaries (not extended boundaries)
        to determine where feathering occurs. Each tile has full weight in the
        center of its data region and ramps down in the overlap regions.
        """
        H, W = shape
        is_continuous = wrapped_phase is not None

        # Align phase offsets between tiles
        if len(tiles_data) > 1:
            if verbose:
                print("Aligning tile phases...")
            tiles_data = self._align_tile_phases_numpy(
                tiles_data, is_continuous=is_continuous
            )

        # Accumulators for weighted average (CPU memory)
        result = np.zeros((H, W), dtype=np.float64)
        weight_sum = np.zeros((H, W), dtype=np.float64)

        for tile, data in tiles_data:
            tile_H, tile_W = data.shape

            # Create blend weights based on DATA boundaries
            # The overlap region is between row_start and data_row_start (top)
            # and between data_row_end and row_end (bottom)
            weights = np.ones((tile_H, tile_W), dtype=np.float64)

            # Top overlap region: ramp from 0 at row_start to 1 at data_row_start
            top_overlap = tile.data_row_start - tile.row_start
            if top_overlap > 0:
                t = np.linspace(0, 1, top_overlap + 1)[1:]  # exclude 0, include 1
                ramp = 0.5 * (1 - np.cos(t * np.pi))
                weights[:top_overlap, :] *= ramp[:, np.newaxis]

            # Bottom overlap region: ramp from 1 at data_row_end to 0 at row_end
            bottom_overlap = tile.row_end - tile.data_row_end
            if bottom_overlap > 0:
                t = np.linspace(0, 1, bottom_overlap + 1)[1:]
                ramp = 0.5 * (1 - np.cos(t * np.pi))
                ramp = ramp[::-1]  # reverse: 1 -> 0
                weights[-bottom_overlap:, :] *= ramp[:, np.newaxis]

            # Left overlap region: ramp from 0 at col_start to 1 at data_col_start
            left_overlap = tile.data_col_start - tile.col_start
            if left_overlap > 0:
                t = np.linspace(0, 1, left_overlap + 1)[1:]
                ramp = 0.5 * (1 - np.cos(t * np.pi))
                weights[:, :left_overlap] *= ramp[np.newaxis, :]

            # Right overlap region: ramp from 1 at data_col_end to 0 at col_end
            right_overlap = tile.col_end - tile.data_col_end
            if right_overlap > 0:
                t = np.linspace(0, 1, right_overlap + 1)[1:]
                ramp = 0.5 * (1 - np.cos(t * np.pi))
                ramp = ramp[::-1]
                weights[:, -right_overlap:] *= ramp[np.newaxis, :]

            # Create mask for valid (non-NaN) pixels
            valid_mask = ~np.isnan(data)

            # Zero out weights for NaN pixels
            effective_weights = weights.copy()
            effective_weights[~valid_mask] = 0.0

            # Replace NaN with 0 for accumulation (will be masked by weights)
            safe_data = data.copy()
            safe_data[~valid_mask] = 0.0

            # Add weighted contribution
            result[tile.row_start:tile.row_end, tile.col_start:tile.col_end] += (
                effective_weights * safe_data
            )
            weight_sum[tile.row_start:tile.row_end, tile.col_start:tile.col_end] += (
                effective_weights
            )

        # Normalize by weight sum
        # Pixels with zero weight sum had no valid data - set to NaN
        valid_output = weight_sum > 1e-10
        result[valid_output] = result[valid_output] / weight_sum[valid_output]
        result[~valid_output] = np.nan

        # For continuous solutions, apply congruence projection on the full
        # merged result. This ensures globally consistent 2π transitions
        # instead of per-tile projections that can disagree at boundaries.
        if is_continuous and wrapped_phase is not None:
            if verbose:
                print("Applying global congruence projection...")
            result = self._apply_congruence_numpy(
                result, wrapped_phase, coherence, nan_mask, verbose=verbose
            )

        return result

    def _apply_congruence_numpy(
        self,
        phi_continuous: np.ndarray,
        wrapped_phase: np.ndarray,
        coherence: np.ndarray | None,
        nan_mask: np.ndarray | None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Apply congruence projection and local consistency correction.

        Projects the continuous solution to the nearest congruent solution
        (wrapped_phase + k * 2π) and fixes local consistency errors.

        Parameters
        ----------
        phi_continuous : np.ndarray
            Merged continuous solution (H, W).
        wrapped_phase : np.ndarray
            Original wrapped phase (H, W).
        coherence : np.ndarray or None
            Coherence map for weighting consistency correction.
        nan_mask : np.ndarray or None
            Boolean mask of invalid pixels.
        verbose : bool
            Print progress information.

        Returns
        -------
        np.ndarray
            Congruent unwrapped phase.
        """
        two_pi = 2 * np.pi

        # Handle NaN: replace with 0 for computation
        if nan_mask is None:
            nan_mask = np.isnan(wrapped_phase) | np.isnan(phi_continuous)
        has_nans = np.any(nan_mask)

        phase_clean = np.where(nan_mask, 0.0, wrapped_phase) if has_nans else wrapped_phase
        phi_cont_clean = np.where(nan_mask, 0.0, phi_continuous) if has_nans else phi_continuous.copy()

        # DC offset adjustment: center fractional parts away from ±0.5
        # Use subsampled median for speed on large images
        k_float = (phi_cont_clean - phase_clean) / two_pi
        frac = k_float - np.round(k_float)
        if has_nans:
            # Subsample for speed: median of 100K samples is sufficient
            valid_indices = np.where(~nan_mask.ravel())[0]
            if valid_indices.size > 0:
                if valid_indices.size > 100000:
                    sample_idx = valid_indices[::max(1, valid_indices.size // 100000)]
                else:
                    sample_idx = valid_indices
                dc_adjust = float(np.median(frac.ravel()[sample_idx])) * two_pi
            else:
                dc_adjust = 0.0
        else:
            if frac.size > 100000:
                stride = max(1, int(np.sqrt(frac.size / 100000)))
                dc_adjust = float(np.median(frac[::stride, ::stride])) * two_pi
            else:
                dc_adjust = float(np.median(frac)) * two_pi
        phi_cont_clean = phi_cont_clean - dc_adjust

        # Congruence projection: k = round((phi - psi) / 2π)
        k = np.round((phi_cont_clean - phase_clean) / two_pi)
        result = phase_clean + k * two_pi

        # Coherence weights for local consistency
        if coherence is not None:
            coh_weights = np.clip(np.nan_to_num(coherence, nan=0.0), 0.0, 1.0)
        else:
            coh_weights = np.ones_like(result)
        if has_nans:
            coh_weights[nan_mask] = 0.0

        # Local consistency correction (numpy version)
        result = self._local_consistency_correction_numpy(
            result, phase_clean, phi_cont_clean, coh_weights,
            nan_mask if has_nans else None, verbose=verbose,
        )

        # Restore NaN
        if has_nans:
            result[nan_mask] = np.nan

        return result

    def _local_consistency_correction_numpy(
        self,
        phi: np.ndarray,
        phase_clean: np.ndarray,
        phi_continuous: np.ndarray,
        coh_weights: np.ndarray,
        nan_mask: np.ndarray | None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Fix local 2π consistency errors using coarse-to-fine approach.

        After congruence projection, some pixels may have k off by ±1.
        A single fine-level iteration only propagates corrections by 1 pixel,
        so large 2π islands (thousands of pixels wide) need a coarse-level
        pass first.

        Approach:
        1. Downsample k, expected_dk, coh_weights by stride S
        2. Run many iterations on the coarse grid (fast, small arrays)
        3. Upsample coarse corrections to full resolution
        4. Run a few fine-level iterations for edge cleanup
        """
        import time
        t0 = time.time()
        two_pi = 2 * np.pi
        H, W = phi.shape

        # Use int16 for k — values are small integers, saves 4x memory
        k = np.round((phi - phase_clean) / two_pi).astype(np.int16)

        # Expected dk from continuous solution (also int16)
        raw_dx = phase_clean[:, 1:] - phase_clean[:, :-1]
        raw_dy = phase_clean[1:, :] - phase_clean[:-1, :]
        phi_dx = phi_continuous[:, 1:] - phi_continuous[:, :-1]
        phi_dy = phi_continuous[1:, :] - phi_continuous[:-1, :]

        expected_dk_x = np.round((phi_dx - raw_dx) / two_pi).astype(np.int16)
        expected_dk_y = np.round((phi_dy - raw_dy) / two_pi).astype(np.int16)

        # Free temporaries
        del raw_dx, raw_dy, phi_dx, phi_dy

        # --- Coarse-to-fine: downsample and correct at coarse level first ---
        stride = 32
        if H > stride * 4 and W > stride * 4:
            if verbose:
                print(f"  Coarse-level correction (stride={stride})...")

            # Downsample k by taking every stride-th pixel
            k_coarse = k[::stride, ::stride].copy()
            Hc, Wc = k_coarse.shape

            # Coarse expected_dk: sum fine-level expected_dk along stride path
            # dk between coarse pixels (i, j) and (i, j+1) in full-res is:
            #   sum of expected_dk_x[i*S, j*S], expected_dk_x[i*S, j*S+1], ..., expected_dk_x[i*S, (j+1)*S-1]
            # Use cumulative sum for efficient computation
            cumdk_x = np.zeros((H, W), dtype=np.int32)
            cumdk_x[:, 1:] = np.cumsum(expected_dk_x, axis=1)
            cumdk_y = np.zeros((H, W), dtype=np.int32)
            cumdk_y[1:, :] = np.cumsum(expected_dk_y, axis=0)

            # Coarse expected dk_x: between columns j*S and (j+1)*S at row i*S
            coarse_edk_x = np.zeros((Hc, Wc - 1), dtype=np.int16)
            for jc in range(Wc - 1):
                j0 = jc * stride
                j1 = min((jc + 1) * stride, W - 1)
                row_idx = np.minimum(np.arange(Hc) * stride, H - 1)
                coarse_edk_x[:, jc] = (cumdk_x[row_idx, j1] - cumdk_x[row_idx, j0]).astype(np.int16)

            # Coarse expected dk_y: between rows i*S and (i+1)*S at col j*S
            coarse_edk_y = np.zeros((Hc - 1, Wc), dtype=np.int16)
            for ic in range(Hc - 1):
                i0 = ic * stride
                i1 = min((ic + 1) * stride, H - 1)
                col_idx = np.minimum(np.arange(Wc) * stride, W - 1)
                coarse_edk_y[ic, :] = (cumdk_y[i1, col_idx] - cumdk_y[i0, col_idx]).astype(np.int16)

            del cumdk_x, cumdk_y

            # Coarse coherence (min along path for conservative weighting)
            coh_coarse = coh_weights[::stride, ::stride].copy()

            # Coarse NaN mask
            nan_coarse = nan_mask[::stride, ::stride] if nan_mask is not None else None

            # Run many iterations at coarse level (small grid, fast)
            coarse_max_iter = 200
            for iteration in range(coarse_max_iter):
                err_x = (k_coarse[:, 1:] - k_coarse[:, :-1]) - coarse_edk_x
                err_y = (k_coarse[1:, :] - k_coarse[:-1, :]) - coarse_edk_y

                n_bad = int(np.count_nonzero(err_x)) + int(np.count_nonzero(err_y))
                if n_bad == 0:
                    break

                correction = np.zeros_like(k_coarse)

                bad_x = err_x != 0
                if bad_x.any():
                    err_x_c = np.clip(err_x, -1, 1)
                    left_weaker = bad_x & (coh_coarse[:, :-1] <= coh_coarse[:, 1:])
                    correction[:, :-1] += np.where(left_weaker, err_x_c, 0)
                    right_weaker = bad_x & ~left_weaker
                    correction[:, 1:] -= np.where(right_weaker, err_x_c, 0)

                bad_y = err_y != 0
                if bad_y.any():
                    err_y_c = np.clip(err_y, -1, 1)
                    top_weaker = bad_y & (coh_coarse[:-1, :] <= coh_coarse[1:, :])
                    correction[:-1, :] += np.where(top_weaker, err_y_c, 0)
                    bottom_weaker = bad_y & ~top_weaker
                    correction[1:, :] -= np.where(bottom_weaker, err_y_c, 0)

                if nan_coarse is not None:
                    correction[nan_coarse] = 0

                np.clip(correction, -1, 1, out=correction)
                if not correction.any():
                    break

                k_coarse += correction.astype(np.int16)

            if verbose:
                print(f"  Coarse level: {iteration + 1} iterations, {n_bad} bad edges ({time.time()-t0:.1f}s)")

            # Upsample coarse corrections to full resolution
            # Correction = k_coarse_corrected - k_coarse_original
            k_coarse_orig = k[::stride, ::stride]
            dk_coarse = (k_coarse - k_coarse_orig).astype(np.int16)

            if dk_coarse.any():
                # Nearest-neighbor upsample: each coarse pixel covers a stride x stride block
                # Use repeat for efficient upsampling
                dk_full = np.repeat(np.repeat(dk_coarse, stride, axis=0), stride, axis=1)
                # Trim to original size
                dk_full = dk_full[:H, :W]

                if nan_mask is not None:
                    dk_full[nan_mask] = 0

                k += dk_full.astype(np.int16)
                del dk_full

            del k_coarse, k_coarse_orig, dk_coarse, coh_coarse

        # --- Fine-level iterations for edge cleanup ---
        fine_max_iter = 5

        for iteration in range(fine_max_iter):
            err_x = (k[:, 1:] - k[:, :-1]) - expected_dk_x
            err_y = (k[1:, :] - k[:-1, :]) - expected_dk_y

            n_bad = int(np.count_nonzero(err_x)) + int(np.count_nonzero(err_y))
            if verbose:
                elapsed = time.time() - t0
                print(f"  Fine consistency iter {iteration}: {n_bad} bad edges ({elapsed:.1f}s)")
            if n_bad == 0:
                break

            correction = np.zeros_like(k)

            bad_x = err_x != 0
            if bad_x.any():
                err_x_c = np.clip(err_x, -1, 1)
                left_weaker = bad_x & (coh_weights[:, :-1] <= coh_weights[:, 1:])
                correction[:, :-1] += np.where(left_weaker, err_x_c, 0)
                right_weaker = bad_x & ~left_weaker
                correction[:, 1:] -= np.where(right_weaker, err_x_c, 0)
                del err_x_c, left_weaker, right_weaker

            bad_y = err_y != 0
            if bad_y.any():
                err_y_c = np.clip(err_y, -1, 1)
                top_weaker = bad_y & (coh_weights[:-1, :] <= coh_weights[1:, :])
                correction[:-1, :] += np.where(top_weaker, err_y_c, 0)
                bottom_weaker = bad_y & ~top_weaker
                correction[1:, :] -= np.where(bottom_weaker, err_y_c, 0)
                del err_y_c, top_weaker, bottom_weaker

            del bad_x, bad_y, err_x, err_y

            if nan_mask is not None:
                correction[nan_mask] = 0

            np.clip(correction, -1, 1, out=correction)
            if not correction.any():
                break

            k += correction.astype(np.int16)
            del correction

        if verbose:
            print(f"  Local consistency: done ({time.time()-t0:.1f}s)")

        return phase_clean + k.astype(np.float64) * two_pi

    def process_tiled_batch(
        self,
        image: torch.Tensor,
        process_fn: Callable[[torch.Tensor], torch.Tensor],
        coherence: torch.Tensor | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """
        Process tiles in batches for better GPU utilization.

        Parameters
        ----------
        image : torch.Tensor
            Input image of shape (H, W).
        process_fn : callable
            Function to apply to batched tiles.
            Should take tensor of shape (B, H, W) and return same shape.
        coherence : torch.Tensor, optional
            Coherence map of shape (H, W).
        batch_size : int, optional
            Batch size. If None, process all tiles at once.

        Returns
        -------
        torch.Tensor
            Processed image of shape (H, W).
        """
        H, W = image.shape
        tiles = self.compute_tiles((H, W))
        n_tiles = len(tiles)

        if batch_size is None:
            batch_size = n_tiles

        # Find maximum tile dimensions
        max_h = max(t.row_end - t.row_start for t in tiles)
        max_w = max(t.col_end - t.col_start for t in tiles)

        processed_tiles = []

        # Process in batches
        for batch_start in range(0, n_tiles, batch_size):
            batch_end = min(batch_start + batch_size, n_tiles)
            batch_tiles = tiles[batch_start:batch_end]
            B = len(batch_tiles)

            # Create padded batch tensor
            batch_data = self.dm.zeros((B, max_h, max_w))

            for i, tile in enumerate(batch_tiles):
                tile_data = self.extract_tile(image, tile)
                th, tw = tile_data.shape
                batch_data[i, :th, :tw] = tile_data

            # Process batch
            batch_result = process_fn(batch_data)

            # Extract results
            for i, tile in enumerate(batch_tiles):
                th = tile.row_end - tile.row_start
                tw = tile.col_end - tile.col_start
                processed_tiles.append((tile, batch_result[i, :th, :tw]))

        # Merge results with phase alignment
        return self.merge_tiles(processed_tiles, (H, W), align_phases=True)
