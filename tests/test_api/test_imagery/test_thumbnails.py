"""Tests for satellite imagery thumbnail generation."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from ftw_dataset_tools.api.imagery.thumbnails import ThumbnailError, generate_thumbnail


def _write_rgb_tif(path: Path, width: int, height: int) -> None:
    """Write a small three-band GeoTIFF for thumbnail tests."""
    data = np.arange(width * height, dtype=np.uint16).reshape(height, width)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 3,
        "dtype": "uint16",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 1, 0.01, 0.01),
    }

    with rasterio.open(path, "w", **profile) as dataset:
        for band_index in range(1, 4):
            dataset.write(data, band_index)


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (200, 200),
        (1000, 500),
    ],
)
def test_generate_thumbnail_matches_tif_dimensions(
    tmp_path: Path,
    width: int,
    height: int,
) -> None:
    """JPG previews retain the source TIF's native dimensions."""
    tif_path = tmp_path / f"sample_{width}x{height}.tif"
    jpg_path = tmp_path / f"sample_{width}x{height}.jpg"
    _write_rgb_tif(tif_path, width, height)

    result = generate_thumbnail(tif_path, jpg_path)

    assert result == jpg_path
    with Image.open(jpg_path) as preview:
        assert preview.size == (width, height)


def test_generate_thumbnail_raises_for_missing_tif(tmp_path: Path) -> None:
    """Missing input files raise a clear thumbnail error."""
    tif_path = tmp_path / "missing.tif"
    jpg_path = tmp_path / "preview.jpg"

    with pytest.raises(ThumbnailError, match="Input file does not exist"):
        generate_thumbnail(tif_path, jpg_path)
