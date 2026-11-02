"""Tests for chip-window nodata percentages read from a band COG."""

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from ftw_dataset_tools.api.imagery.nodata_analysis import (
    calculate_nodata_percentage,
    chip_nodata_pct,
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
        # Production rejects anything above 0, so a clean window must read exactly 0.
        assert calculate_nodata_percentage(band_path, _bbox(0, 2)) == 0.0

    def test_nodata_half(self, band_path):
        assert calculate_nodata_percentage(band_path, _bbox(2, 4)) == pytest.approx(100.0, abs=2.0)

    def test_window_outside_the_scene_is_fully_nodata(self, band_path):
        """Used to raise WindowError, which selection then treated as a clean chip."""
        assert calculate_nodata_percentage(band_path, _bbox(40, 44)) == 100.0

    def test_window_hanging_off_the_scene_counts_the_missing_part(self, band_path):
        # Columns -2..2: two outside the raster, two valid.
        assert calculate_nodata_percentage(band_path, _bbox(-2, 2)) == pytest.approx(50.0, abs=3.0)


class TestFullResolution:
    def test_a_few_pixel_hole_an_overview_averages_away_is_counted(self, tmp_path):
        """A quarterly mosaic's cloud hole is a few pixels; overviews blur it out."""
        from rasterio.enums import Resampling

        size = 256
        data = np.full((size, size), 5, dtype="int16")
        data[100:102, 100:102] = -32768  # 4 pixels
        path = tmp_path / "mosaic.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=size,
            width=size,
            count=1,
            dtype="int16",
            nodata=-32768,
            crs="EPSG:32633",
            transform=from_origin(ORIGIN_X, ORIGIN_Y, PIXEL, PIXEL),
        ) as dst:
            dst.write(data, 1)
            dst.build_overviews([2, 4, 8], Resampling.average)
        with rasterio.open(path, overview_level=2) as src:
            assert not (src.read(1) == -32768).any()  # the hole is gone at this level

        # A window around the hole, inside the raster: 4 of ~100x100 pixels.
        bbox = transform_bounds(
            "EPSG:32633",
            "EPSG:4326",
            ORIGIN_X + 50 * PIXEL,
            ORIGIN_Y - 150 * PIXEL,
            ORIGIN_X + 150 * PIXEL,
            ORIGIN_Y - 50 * PIXEL,
        )
        assert 0.03 < calculate_nodata_percentage(str(path), bbox) < 0.05


class TestChipNodataPct:
    def test_fully_inside(self):
        assert chip_nodata_pct(0.0, 0.25) == pytest.approx(25.0)

    def test_outside_part_counts_as_nodata(self):
        assert chip_nodata_pct(0.5, 0.0) == pytest.approx(50.0)
        assert chip_nodata_pct(0.5, 0.5) == pytest.approx(75.0)


class TestFootprintNearGranuleCorner:
    """A chip 20 m inside a granule's north-west corner, ~2.5 deg from the zone's
    central meridian at 58 N, the top-left chip of an MGRS square.

    Its lon/lat bbox, taken back onto the UTM grid, is an envelope ~70 m wider per
    side than the chip, so measured by bbox it hangs off the raster; measured by its
    footprint it is fully covered.
    """

    CRS = "EPSG:32634"
    LEFT, TOP = 360000.0, 6440000.0  # the granule's top-left corner

    @pytest.fixture
    def granule(self, tmp_path):
        size = 300  # 6 km at 20 m
        path = tmp_path / "granule.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=size,
            width=size,
            count=1,
            dtype="uint16",
            crs=self.CRS,
            transform=from_origin(self.LEFT, self.TOP, 20.0, 20.0),
        ) as dst:
            dst.write(np.full((size, size), 5, dtype="uint16"), 1)
        return str(path)

    def _chip(self):
        x0, y1 = self.LEFT + 20, self.TOP - 20
        corners = [(x0, y1 - 2000), (x0 + 2000, y1 - 2000), (x0 + 2000, y1), (x0, y1)]
        to_lonlat = Transformer.from_crs(self.CRS, "EPSG:4326", always_xy=True)
        ring = [list(to_lonlat.transform(x, y)) for x, y in corners]
        footprint = {"type": "Polygon", "coordinates": [[*ring, ring[0]]]}
        lons, lats = zip(*ring, strict=True)
        return footprint, (min(lons), min(lats), max(lons), max(lats))

    def test_footprint_is_fully_covered(self, granule):
        footprint, bbox = self._chip()
        assert calculate_nodata_percentage(granule, bbox, footprint) == 0.0

    def test_bbox_envelope_hangs_off_the_edge(self, granule):
        """Why selection passes the footprint: the bbox alone reads as partly missing."""
        _, bbox = self._chip()
        assert calculate_nodata_percentage(granule, bbox) > 1.0
