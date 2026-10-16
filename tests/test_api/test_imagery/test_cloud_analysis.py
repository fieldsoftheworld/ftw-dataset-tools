"""Tests for chip-level SCL statistics."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from ftw_dataset_tools.api.imagery.cloud_analysis import (
    calculate_pixel_cloud_cover,
    read_scl_chip_stats,
)

# A 10x10 SCL raster in EPSG:32633 whose top-left 100 m square is the chip.
PIXEL = 20.0
ORIGIN_X, ORIGIN_Y = 500000.0, 5100000.0


def _write_scl(path, values):
    data = np.array(values, dtype="uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="uint8",
        crs="EPSG:32633",
        transform=from_origin(ORIGIN_X, ORIGIN_Y, PIXEL, PIXEL),
    ) as dst:
        dst.write(data, 1)
    return str(path)


def _bbox_for(cols, rows):
    """Lon/lat bbox covering the first ``cols`` x ``rows`` pixels."""
    from rasterio.warp import transform_bounds

    return transform_bounds(
        "EPSG:32633",
        "EPSG:4326",
        ORIGIN_X,
        ORIGIN_Y - rows * PIXEL,
        ORIGIN_X + cols * PIXEL,
        ORIGIN_Y,
    )


@pytest.fixture
def scl_path(tmp_path):
    # Row 0: all vegetation. Row 1: half cloud-high, half vegetation.
    # Row 2: all no-data. Rows 3+: vegetation.
    grid = [[4] * 4 for _ in range(4)]
    grid[1] = [9, 9, 4, 4]
    grid[2] = [0, 0, 0, 0]
    return _write_scl(tmp_path / "scl.tif", grid)


class TestReadSCLChipStats:
    def test_clean_window_is_all_zero(self, scl_path):
        stats = read_scl_chip_stats(scl_path, _bbox_for(4, 1))
        assert stats.nodata_pct == 0.0
        assert stats.cloud_pct == 0.0

    def test_nodata_comes_from_class_zero(self, scl_path):
        # Rows 0-2: 12 pixels, 4 of them no-data.
        stats = read_scl_chip_stats(scl_path, _bbox_for(4, 3))
        assert stats.nodata_pct == pytest.approx(100 * 4 / 12)

    def test_cloud_is_a_share_of_valid_pixels_only(self, scl_path):
        # Rows 0-2: 12 pixels, 8 valid, 2 of those cloud.
        stats = read_scl_chip_stats(scl_path, _bbox_for(4, 3))
        assert stats.cloud_pct == pytest.approx(100 * 2 / 8)

    def test_cloud_matches_the_standalone_cloud_function(self, scl_path):
        bbox = _bbox_for(4, 3)
        assert read_scl_chip_stats(scl_path, bbox).cloud_pct == pytest.approx(
            calculate_pixel_cloud_cover(cloud_href=scl_path, bbox=bbox, cloud_type="scl")
        )

    def test_shadow_and_cirrus_count_as_cloud(self, tmp_path):
        path = _write_scl(tmp_path / "mixed.tif", [[3, 8, 9, 10], [4, 4, 4, 4]])
        stats = read_scl_chip_stats(path, _bbox_for(4, 2))
        assert stats.cloud_pct == pytest.approx(50.0)
        assert stats.nodata_pct == 0.0

    def test_window_outside_the_scene_is_fully_nodata(self, scl_path):
        from rasterio.warp import transform_bounds

        far = transform_bounds(
            "EPSG:32633",
            "EPSG:4326",
            ORIGIN_X + 1e5,
            ORIGIN_Y + 1e5,
            ORIGIN_X + 1.1e5,
            ORIGIN_Y + 1.1e5,
        )
        stats = read_scl_chip_stats(scl_path, far)
        assert stats.nodata_pct == 100.0
        assert stats.cloud_pct == 0.0

    def test_all_nodata_window_reports_no_cloud(self, tmp_path):
        path = _write_scl(tmp_path / "empty.tif", [[0, 0], [0, 0]])
        stats = read_scl_chip_stats(path, _bbox_for(2, 2))
        assert stats.nodata_pct == 100.0
        assert stats.cloud_pct == 0.0
