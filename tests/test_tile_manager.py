"""Tests for the TileManager."""

import numpy as np
import pytest
import torch

from rapidphase.tiling.tile_manager import TileManager, TileInfo
from rapidphase.device.manager import DeviceManager


class TestTileManager:
    """Tests for TileManager class."""

    @pytest.fixture
    def tile_manager(self, device_manager):
        """Create a TileManager instance."""
        return TileManager(device_manager, ntiles=(2, 2), overlap=16)

    def test_compute_tiles_basic(self, tile_manager):
        """Test basic tile computation."""
        tiles = tile_manager.compute_tiles((100, 100))

        assert len(tiles) == 4  # 2x2 tiles

        # Check all tiles are TileInfo
        for tile in tiles:
            assert isinstance(tile, TileInfo)

    def test_compute_tiles_covers_image(self, tile_manager):
        """Test that tiles cover the entire image."""
        H, W = 100, 100
        tiles = tile_manager.compute_tiles((H, W))

        # Check that data regions cover entire image
        covered = np.zeros((H, W), dtype=bool)
        for tile in tiles:
            covered[
                tile.data_row_start:tile.data_row_end,
                tile.data_col_start:tile.data_col_end
            ] = True

        assert np.all(covered)

    def test_compute_tiles_overlap(self, tile_manager):
        """Test that tiles have proper overlap."""
        tiles = tile_manager.compute_tiles((100, 100))

        # Adjacent tiles should overlap
        tile_00 = tiles[0]  # Top-left
        tile_01 = tiles[1]  # Top-right

        # Horizontal overlap
        assert tile_00.col_end > tile_01.col_start

    def test_extract_tile(self, tile_manager):
        """Test extracting a tile from an image."""
        image = torch.arange(100 * 100).reshape(100, 100).float()
        image = image.to(tile_manager.dm.device)

        tiles = tile_manager.compute_tiles((100, 100))
        tile_data = tile_manager.extract_tile(image, tiles[0])

        expected_h = tiles[0].row_end - tiles[0].row_start
        expected_w = tiles[0].col_end - tiles[0].col_start
        assert tile_data.shape == (expected_h, expected_w)

    def test_create_blend_weights_interior(self, tile_manager):
        """Test blend weights for interior tile."""
        H, W = 100, 100
        tiles = tile_manager.compute_tiles((H, W))

        # Pick a middle tile if exists (in 2x2, all tiles touch edges)
        tile = tiles[0]
        tile_shape = (tile.row_end - tile.row_start, tile.col_end - tile.col_start)

        weights = tile_manager.create_blend_weights(tile_shape, tile, (H, W))

        assert weights.shape == tile_shape
        # Weights should be non-negative (may have very small values at corners)
        assert torch.all(weights >= 0)
        # Weights should be <= 1
        assert torch.all(weights <= 1)
        # Most weights should be positive
        assert (weights > 0).sum() > 0.9 * weights.numel()

    def test_merge_tiles_identity(self, tile_manager):
        """Test that merging tiles of constant value returns constant."""
        H, W = 64, 64
        tiles = tile_manager.compute_tiles((H, W))

        # All tiles have constant value with correct dtype
        tiles_data = []
        for tile in tiles:
            th = tile.row_end - tile.row_start
            tw = tile.col_end - tile.col_start
            data = torch.ones(
                (th, tw),
                device=tile_manager.dm.device,
                dtype=tile_manager.dm.dtype,
            )
            tiles_data.append((tile, data))

        merged = tile_manager.merge_tiles(tiles_data, (H, W))

        # Should be close to all ones (may have small variations at boundaries)
        expected = torch.ones(
            H, W,
            device=tile_manager.dm.device,
            dtype=tile_manager.dm.dtype,
        )
        torch.testing.assert_close(merged, expected, atol=0.1, rtol=0.05)

    def test_merge_tiles_shape(self, tile_manager):
        """Test that merged output has correct shape."""
        H, W = 64, 64
        tiles = tile_manager.compute_tiles((H, W))

        tiles_data = []
        for tile in tiles:
            th = tile.row_end - tile.row_start
            tw = tile.col_end - tile.col_start
            data = torch.randn((th, tw), device=tile_manager.dm.device)
            tiles_data.append((tile, data))

        merged = tile_manager.merge_tiles(tiles_data, (H, W))

        assert merged.shape == (H, W)

    def test_process_tiled_identity(self, tile_manager):
        """Test process_tiled with identity function."""
        image = torch.randn(64, 64, device=tile_manager.dm.device)

        # Identity processing
        result = tile_manager.process_tiled(image, lambda x: x)

        # Should be close to original (may have blending artifacts at tile boundaries)
        # Check that the overall structure is preserved
        assert result.shape == image.shape
        # Correlation should be high
        corr = torch.corrcoef(torch.stack([image.flatten(), result.flatten()]))[0, 1]
        assert corr > 0.95

    def test_process_tiled_with_coherence(self, tile_manager):
        """Test process_tiled passes coherence to processing function."""
        image = torch.randn(64, 64, device=tile_manager.dm.device)
        coherence = torch.rand(64, 64, device=tile_manager.dm.device)

        # Function that uses coherence
        def process(tile_img, tile_coh=None):
            assert tile_coh is not None
            return tile_img * tile_coh

        result = tile_manager.process_tiled(image, process, coherence)

        assert result.shape == image.shape

    def test_different_tile_counts(self, device_manager):
        """Test with different tile configurations."""
        for ntiles in [(1, 1), (2, 2), (3, 3), (2, 4)]:
            tm = TileManager(device_manager, ntiles=ntiles, overlap=8)
            tiles = tm.compute_tiles((100, 100))

            expected_count = ntiles[0] * ntiles[1]
            assert len(tiles) == expected_count

    def test_small_overlap(self, device_manager):
        """Test with small overlap."""
        tm = TileManager(device_manager, ntiles=(2, 2), overlap=4)
        tiles = tm.compute_tiles((64, 64))

        assert len(tiles) == 4

    def test_large_overlap(self, device_manager):
        """Test with large overlap."""
        tm = TileManager(device_manager, ntiles=(2, 2), overlap=32)
        tiles = tm.compute_tiles((100, 100))

        # Tiles should have significant overlap
        for tile in tiles:
            assert tile.row_end - tile.row_start > 50
            assert tile.col_end - tile.col_start > 50


class TestTileInfo:
    """Tests for TileInfo dataclass."""

    def test_tile_info_creation(self):
        """Test creating TileInfo."""
        tile = TileInfo(
            row_idx=0,
            col_idx=1,
            row_start=0,
            row_end=60,
            col_start=40,
            col_end=100,
            data_row_start=0,
            data_row_end=50,
            data_col_start=50,
            data_col_end=100,
        )

        assert tile.row_idx == 0
        assert tile.col_idx == 1
        assert tile.row_start == 0
        assert tile.row_end == 60
