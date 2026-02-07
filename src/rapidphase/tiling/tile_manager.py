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

        # Hard-stitch: each tile contributes only its DATA region.
        result = self.dm.zeros((H, W))
        result[:] = float('nan')

        for tile, data in tiles_data:
            dr0 = tile.data_row_start - tile.row_start
            dr1 = tile.data_row_end - tile.row_start
            dc0 = tile.data_col_start - tile.col_start
            dc1 = tile.data_col_end - tile.col_start

            result[
                tile.data_row_start:tile.data_row_end,
                tile.data_col_start:tile.data_col_end,
            ] = data[dr0:dr1, dc0:dc1]

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

            # Process tile on GPU (full unwrap with per-tile congruence)
            result_t = unwrapper.unwrap(tile_data_t, coh_tile_t, nan_mask=nan_mask_tile_t)

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

        # Merge results with phase alignment
        result = self._merge_tiles_numpy(processed_tiles_np, (H, W), verbose=verbose)

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

                # Process tile using the GPU-specific unwrapper (full unwrap with congruence)
                result_t = unwrapper.unwrap(tile_data_t, coh_tile_t, nan_mask=nan_mask_tile_t)

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

        # Merge results with phase alignment
        result = self._merge_tiles_numpy(processed_tiles_np, (H, W), verbose=verbose)

        if verbose and not HAS_TQDM:
            print("  Done.")

        return result

    def _align_tile_phases_numpy(
        self,
        tiles_data: list[tuple[TileInfo, np.ndarray]],
        verbose: bool = False,
    ) -> list[tuple[TileInfo, np.ndarray]]:
        """
        Align phase offsets between adjacent tiles (numpy/CPU version).

        Each tile has an arbitrary phase offset (integer multiple of 2π).
        Computes the dominant 2π multiple in a thin center strip of the
        overlap and applies it as the correction.

        Uses BFS from the tile with the most valid pixels.
        """
        if len(tiles_data) <= 1:
            return tiles_data

        n_rows, n_cols = self.ntiles

        # Create a dictionary for easy lookup by (row, col) index
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
        offsets = {ref_idx: 0.0}
        aligned.add(ref_idx)

        # BFS to propagate alignment
        queue = [ref_idx]

        while queue:
            curr_idx = queue.pop(0)
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

                # Find overlapping region in global coordinates
                overlap_row_start = max(curr_tile.row_start, neighbor_tile.row_start)
                overlap_row_end = min(curr_tile.row_end, neighbor_tile.row_end)
                overlap_col_start = max(curr_tile.col_start, neighbor_tile.col_start)
                overlap_col_end = min(curr_tile.col_end, neighbor_tile.col_end)

                if overlap_row_start >= overlap_row_end or overlap_col_start >= overlap_col_end:
                    continue

                oh = overlap_row_end - overlap_row_start
                ow = overlap_col_end - overlap_col_start

                # Sample a thin strip at the CENTER of the overlap in the
                # narrow direction (perpendicular to tile boundary). This
                # avoids internal 2π transitions that may occur at different
                # positions in the two tiles, which would split the k votes.
                # Use most of the extent in the long direction for statistics.
                strip_half = 10  # ±10 pixels around center
                margin_frac = 0.1  # skip 10% margins in long direction

                if oh < ow:
                    # Vertical neighbor: overlap is short (rows) and wide (cols)
                    center_r = overlap_row_start + oh // 2
                    cr0 = max(overlap_row_start, center_r - strip_half)
                    cr1 = min(overlap_row_end, center_r + strip_half)
                    margin_w = int(ow * margin_frac)
                    cc0 = overlap_col_start + margin_w
                    cc1 = overlap_col_end - margin_w
                else:
                    # Horizontal neighbor: overlap is tall (rows) and narrow (cols)
                    margin_h = int(oh * margin_frac)
                    cr0 = overlap_row_start + margin_h
                    cr1 = overlap_row_end - margin_h
                    center_c = overlap_col_start + ow // 2
                    cc0 = max(overlap_col_start, center_c - strip_half)
                    cc1 = min(overlap_col_end, center_c + strip_half)

                if cr0 >= cr1:
                    cr0, cr1 = overlap_row_start, overlap_row_end
                if cc0 >= cc1:
                    cc0, cc1 = overlap_col_start, overlap_col_end

                # Subsample for speed on large strips
                n_pixels = (cr1 - cr0) * (cc1 - cc0)
                stride = max(1, int(np.sqrt(n_pixels / 50000))) if n_pixels > 50000 else 1

                curr_overlap = curr_data[
                    cr0 - curr_tile.row_start:cr1 - curr_tile.row_start:stride,
                    cc0 - curr_tile.col_start:cc1 - curr_tile.col_start:stride,
                ]
                neighbor_overlap = neighbor_data[
                    cr0 - neighbor_tile.row_start:cr1 - neighbor_tile.row_start:stride,
                    cc0 - neighbor_tile.col_start:cc1 - neighbor_tile.col_start:stride,
                ]

                diff = (curr_overlap + curr_offset) - neighbor_overlap
                valid_mask = ~np.isnan(diff)

                if valid_mask.any():
                    # Compute integer 2π multiple for each pixel, then take the
                    # mode (most common value). This ensures the offset is always
                    # an exact multiple of 2π, preserving congruence.
                    two_pi = 2 * np.pi
                    k_per_pixel = np.round(diff[valid_mask] / two_pi).astype(np.int64)
                    k_min = int(k_per_pixel.min())
                    k_shifted = k_per_pixel - k_min
                    counts = np.bincount(k_shifted)
                    k_mode = int(np.argmax(counts)) + k_min
                    offset = k_mode * two_pi

                    if verbose:
                        n_valid = int(valid_mask.sum())
                        mode_count = int(counts[k_mode - k_min])
                        pct = 100.0 * mode_count / n_valid
                        print(
                            f"  Align ({curr_idx[0]},{curr_idx[1]})->({neighbor_idx[0]},{neighbor_idx[1]}): "
                            f"k_mode={k_mode}, votes={mode_count}/{n_valid} ({pct:.0f}%), "
                            f"offset={offset:.2f} rad"
                        )
                else:
                    offset = 0.0

                offsets[neighbor_idx] = offset
                aligned.add(neighbor_idx)
                queue.append(neighbor_idx)

        # Apply offsets
        aligned_tiles = []
        for tile, data in tiles_data:
            idx = (tile.row_idx, tile.col_idx)
            offset = offsets.get(idx, 0.0)
            aligned_tiles.append((tile, data + offset))

        return aligned_tiles

    def _merge_tiles_numpy(
        self,
        tiles_data: list[tuple[TileInfo, np.ndarray]],
        shape: tuple[int, int],
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Merge processed tiles back into a full image (numpy/CPU version).

        Aligns phase offsets using integer 2π voting in the overlap center
        strip, then hard-stitches each tile's DATA region (no blending).
        """
        H, W = shape

        # Align phase offsets between tiles
        if len(tiles_data) > 1:
            if verbose:
                print("Aligning tile phases...")
            tiles_data = self._align_tile_phases_numpy(tiles_data, verbose=verbose)

        # Hard-stitch: each tile contributes only its DATA region.
        # This avoids blending-induced non-congruence at boundaries.
        result = np.full((H, W), np.nan, dtype=np.float64)

        for tile, data in tiles_data:
            # Extract the DATA region (non-overlap portion) from the tile
            dr0 = tile.data_row_start - tile.row_start
            dr1 = tile.data_row_end - tile.row_start
            dc0 = tile.data_col_start - tile.col_start
            dc1 = tile.data_col_end - tile.col_start

            result[
                tile.data_row_start:tile.data_row_end,
                tile.data_col_start:tile.data_col_end,
            ] = data[dr0:dr1, dc0:dc1]

        return result

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
