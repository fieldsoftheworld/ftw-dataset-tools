"""Tests for raster statistics computation and embedding."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds


class TestComputeBandStats:
    def test_exact_stats_without_nodata(self) -> None:
        from ftw_dataset_tools.api.raster_stats import compute_band_stats

        data = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        stats = compute_band_stats(data)

        assert stats.minimum == 0
        assert stats.maximum == 3
        assert stats.mean == 1.5
        assert abs(stats.stddev - 1.118033988749895) < 1e-9
        assert stats.valid_percent is None

    def test_nodata_excluded_and_valid_percent_reported(self) -> None:
        from ftw_dataset_tools.api.raster_stats import compute_band_stats

        data = np.array([[255, 1], [2, 255]], dtype=np.uint8)
        stats = compute_band_stats(data, nodata=255)

        assert stats.minimum == 1
        assert stats.maximum == 2
        assert stats.mean == 1.5
        assert stats.valid_percent == 50.0

    def test_all_nodata_yields_zero_valid_percent(self) -> None:
        from ftw_dataset_tools.api.raster_stats import compute_band_stats

        data = np.full((2, 2), 9, dtype=np.uint8)
        stats = compute_band_stats(data, nodata=9)

        assert stats.valid_percent == 0.0
        assert stats.minimum == 0.0
        assert stats.maximum == 0.0
        assert stats.mean == 0.0
        assert stats.stddev == 0.0


class TestEmbedAndRead:
    def _write(self, path: Path, data: np.ndarray, nodata: int | None = None) -> None:
        from ftw_dataset_tools.api.raster_stats import compute_band_stats, embed_band_stats

        profile = {
            "driver": "COG",
            "width": data.shape[1],
            "height": data.shape[0],
            "count": 1,
            "dtype": data.dtype,
            "crs": "EPSG:4326",
            "transform": from_bounds(0, 0, 1, 1, data.shape[1], data.shape[0]),
            "compress": "deflate",
        }
        if nodata is not None:
            profile["nodata"] = nodata
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)
            embed_band_stats(dst, 1, compute_band_stats(data, nodata=nodata))

    def test_roundtrip_in_cog_without_sidecar(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.raster_stats import read_band_stats

        path = tmp_path / "m.tif"
        self._write(path, np.array([[0, 1], [2, 0]], dtype=np.uint8))

        stats = read_band_stats(path, 1)
        assert stats is not None
        assert stats.minimum == 0
        assert stats.maximum == 2
        assert stats.mean == 0.75
        assert stats.valid_percent is None
        assert not (tmp_path / "m.tif.aux.xml").exists()

    def test_valid_percent_written_when_nodata(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.raster_stats import read_band_stats

        path = tmp_path / "n.tif"
        self._write(path, np.array([[7, 1], [2, 7]], dtype=np.uint8), nodata=7)

        stats = read_band_stats(path, 1)
        assert stats is not None
        assert stats.valid_percent == 50.0

    def test_read_returns_none_when_absent(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.raster_stats import read_band_stats

        path = tmp_path / "plain.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=1,
            height=1,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_bounds(0, 0, 1, 1, 1, 1),
        ) as dst:
            dst.write(np.zeros((1, 1), dtype=np.uint8), 1)

        assert read_band_stats(path, 1) is None
