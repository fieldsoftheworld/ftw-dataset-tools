"""Tests for chip-window nodata percentages read from a band COG."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from ftw_dataset_tools.api.imagery.nodata_analysis import (
    calculate_nodata_percentage,
    window_nodata_pct,
)

# A 4x4 band in EPSG:32633 at 20 m: left half valid, right half nodata (0).
PIXEL = 20.0
ORIGIN_X, ORIGIN_Y = 500000.0, 5100000.0


@pytest.fixture
def band_path(tmp_path):
    data = np.array([[5, 5, 0, 0]] * 4, dtype="uint16")
    path = tmp_path / "nir.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="uint16",
        crs="EPSG:32633",
        transform=from_origin(ORIGIN_X, ORIGIN_Y, PIXEL, PIXEL),
    ) as dst:
        dst.write(data, 1)
    return str(path)


def _bbox(col0, col1, rows=4):
    """Lon/lat bbox over columns [col0, col1) of the first ``rows`` rows."""
    return transform_bounds(
        "EPSG:32633",
        "EPSG:4326",
        ORIGIN_X + col0 * PIXEL,
        ORIGIN_Y - rows * PIXEL,
        ORIGIN_X + col1 * PIXEL,
        ORIGIN_Y,
    )


class TestCalculateNodataPercentage:
    def test_valid_half(self, band_path):
        assert calculate_nodata_percentage(band_path, _bbox(0, 2)) == pytest.approx(0.0, abs=2.0)

    def test_nodata_half(self, band_path):
        assert calculate_nodata_percentage(band_path, _bbox(2, 4)) == pytest.approx(
            100.0, abs=2.0
        )

    def test_window_outside_the_scene_is_fully_nodata(self, band_path):
        """Used to raise WindowError, which selection then treated as a clean chip."""
        assert calculate_nodata_percentage(band_path, _bbox(40, 44)) == 100.0

    def test_window_hanging_off_the_scene_counts_the_missing_part(self, band_path):
        # Columns -2..2: two outside the raster, two valid.
        assert calculate_nodata_percentage(band_path, _bbox(-2, 2)) == pytest.approx(
            50.0, abs=3.0
        )


class TestWindowNodataPct:
    def test_fully_inside(self):
        w = Window(0, 0, 10, 10)
        assert window_nodata_pct(w, w, 0.25) == pytest.approx(25.0)

    def test_outside_part_counts_as_nodata(self):
        requested = Window(-5, 0, 10, 10)
        read = Window(0, 0, 5, 10)
        assert window_nodata_pct(requested, read, 0.0) == pytest.approx(50.0)
        assert window_nodata_pct(requested, read, 0.5) == pytest.approx(75.0)

    def test_empty_request_is_fully_nodata(self):
        assert window_nodata_pct(Window(0, 0, 0, 0), Window(0, 0, 0, 0), 0.0) == 100.0
