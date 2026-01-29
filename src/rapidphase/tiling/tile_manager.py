"""
Tile manager for processing large images.

Splits large interferograms into overlapping tiles for memory-efficient
GPU processing, then merges results with phase alignment and feathered blending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

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

        Each tile from phase unwrapping has an arbitrary constant offset.
        This method computes and applies corrections so that overlapping
        regions have consistent phase values.

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

        n_rows, n_cols = self.ntiles

        # Create a dictionary for easy lookup by (row, col) index
        tile_dict = {}
        for tile, data in tiles_data:
            tile_dict[(tile.row_idx, tile.col_idx)] = (tile, data.clone())

        # Track which tiles have been aligned
        aligned = set()
        # Store offsets to apply
        offsets = {(0, 0): 0.0}  # First tile is reference
        aligned.add((0, 0))

        # BFS to propagate alignment from tile (0,0)
        queue = [(0, 0)]

        while queue:
            curr_idx = queue.pop(0)
            curr_tile, curr_data = tile_dict[curr_idx]
            curr_offset = offsets[curr_idx]

            # Check all neighbors (right and down primarily, but also left and up)
            neighbors = [
                (curr_idx[0], curr_idx[1] + 1),  # right
                (curr_idx[0] + 1, curr_idx[1]),  # down
                (curr_idx[0], curr_idx[1] - 1),  # left
                (curr_idx[0] - 1, curr_idx[1]),  # up
            ]

            for neighbor_idx in neighbors:
                if neighbor_idx in aligned:
                    continue
                if neighbor_idx not in tile_dict:
                    continue

                neighbor_tile, neighbor_data = tile_dict[neighbor_idx]

                # Find overlapping region in global coordinates
                overlap_row_start = max(curr_tile.row_start, neighbor_tile.row_start)
                overlap_row_end = min(curr_tile.row_end, neighbor_tile.row_end)
                overlap_col_start = max(curr_tile.col_start, neighbor_tile.col_start)
                overlap_col_end = min(curr_tile.col_end, neighbor_tile.col_end)

                # Check if there's actual overlap
                if overlap_row_start >= overlap_row_end or overlap_col_start >= overlap_col_end:
                    continue

                # Extract overlap regions from both tiles (in local coordinates)
                curr_local_row_start = overlap_row_start - curr_tile.row_start
                curr_local_row_end = overlap_row_end - curr_tile.row_start
                curr_local_col_start = overlap_col_start - curr_tile.col_start
                curr_local_col_end = overlap_col_end - curr_tile.col_start

                neighbor_local_row_start = overlap_row_start - neighbor_tile.row_start
                neighbor_local_row_end = overlap_row_end - neighbor_tile.row_start
                neighbor_local_col_start = overlap_col_start - neighbor_tile.col_start
                neighbor_local_col_end = overlap_col_end - neighbor_tile.col_start

                curr_overlap = curr_data[
                    curr_local_row_start:curr_local_row_end,
                    curr_local_col_start:curr_local_col_end
                ]
                neighbor_overlap = neighbor_data[
                    neighbor_local_row_start:neighbor_local_row_end,
                    neighbor_local_col_start:neighbor_local_col_end
                ]

                # Compute offset: neighbor should match curr (after curr's offset is applied)
                # Use median for robustness to outliers
                diff = (curr_overlap + curr_offset) - neighbor_overlap

                # Handle NaN values by masking them out before computing median
                valid_mask = ~(torch.isnan(diff))
                if valid_mask.any():
                    valid_diff = diff[valid_mask]
                    offset = torch.median(valid_diff).item()
                else:
                    # No valid overlap, use zero offset
                    offset = 0.0

                offsets[neighbor_idx] = offset
                aligned.add(neighbor_idx)
                queue.append(neighbor_idx)

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
