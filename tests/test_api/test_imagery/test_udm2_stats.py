"""Tests for reading a chip's clear and nodata shares from a PlanetScope UDM2 file."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from ftw_dataset_tools.api.imagery.cloud_analysis import read_udm2_chip_stats

if TYPE_CHECKING:
    from pathlib import Path

# A 3 m UTM 32N raster of 100 x 100 pixels; the chip is its western half.
X0, Y0, RES, SIZE = 300_000.0, 5_500_000.0, 3.0, 100
TO_LONLAT = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)


def _chip_bbox(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
    # Shrink a touch so the reprojected box stays inside the intended pixels.
    pad = 1.0
    west, south = TO_LONLAT.transform(x0 + pad, y0 + pad)
    east, north = TO_LONLAT.transform(x1 - pad, y1 - pad)
    return (west, south, east, north)


def _write_udm2(path: Path, clear: np.ndarray, unusable: np.ndarray) -> Path:
    bands = np.zeros((8, SIZE, SIZE), dtype=np.uint8)
    bands[0] = clear
    bands[7] = unusable
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=SIZE,
        height=SIZE,
        count=8,
        dtype="uint8",
        crs="EPSG:32632",
        transform=from_origin(X0, Y0, RES, RES),
    ) as dst:
        dst.write(bands)
    return path


WEST_HALF = _chip_bbox(X0, Y0 - SIZE * RES, X0 + SIZE * RES / 2, Y0)


def test_all_clear(tmp_path: Path) -> None:
    path = _write_udm2(tmp_path / "udm2.tif", np.ones((SIZE, SIZE)), np.zeros((SIZE, SIZE)))

    stats = read_udm2_chip_stats(str(path), WEST_HALF)

    assert stats.nodata_pct == pytest.approx(0.0, abs=1.5)
    assert stats.cloud_pct == pytest.approx(0.0)


def test_cloud_is_a_share_of_valid_pixels(tmp_path: Path) -> None:
    clear = np.ones((SIZE, SIZE))
    clear[:, :25] = 0  # the western quarter, half of the chip, is not clear
    path = _write_udm2(tmp_path / "udm2.tif", clear, np.zeros((SIZE, SIZE)))

    stats = read_udm2_chip_stats(str(path), WEST_HALF)

    assert stats.cloud_pct == pytest.approx(50.0, abs=3)


def test_blackfill_counts_as_nodata_not_cloud(tmp_path: Path) -> None:
    clear = np.ones((SIZE, SIZE))
    unusable = np.zeros((SIZE, SIZE))
    unusable[:50, :] = 1  # blackfill (bit 0) over the northern half
    clear[:50, :] = 0
    path = _write_udm2(tmp_path / "udm2.tif", clear, unusable)

    stats = read_udm2_chip_stats(str(path), WEST_HALF)

    assert stats.nodata_pct == pytest.approx(50.0, abs=3)
    assert stats.cloud_pct == pytest.approx(0.0)


def test_chip_beyond_the_file_is_all_nodata(tmp_path: Path) -> None:
    path = _write_udm2(tmp_path / "udm2.tif", np.ones((SIZE, SIZE)), np.zeros((SIZE, SIZE)))
    elsewhere = _chip_bbox(X0 + 10_000, Y0 - 300, X0 + 10_300, Y0)

    assert read_udm2_chip_stats(str(path), elsewhere).nodata_pct == 100.0


def test_chip_half_past_the_edge_counts_the_missing_part(tmp_path: Path) -> None:
    path = _write_udm2(tmp_path / "udm2.tif", np.ones((SIZE, SIZE)), np.zeros((SIZE, SIZE)))
    straddling = _chip_bbox(X0 + SIZE * RES / 2, Y0 - SIZE * RES, X0 + SIZE * RES * 1.5, Y0)

    assert read_udm2_chip_stats(str(path), straddling).nodata_pct == pytest.approx(50.0, abs=3)
