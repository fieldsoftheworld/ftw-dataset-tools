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


class TestSceneThumbnail:
    """Previews read straight from a scene COG, for datasets that keep it remote."""

    def _scene(self, path: Path, crs: str = "EPSG:32633") -> Path:
        """A 'scene' COG far larger than one chip, with a recognisable gradient."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        width = height = 400
        # Scene spans 4000 m; a chip will take the middle 10% of it.
        transform = from_bounds(500000, 5000000, 504000, 5004000, width, height)
        data = np.zeros((3, height, width), dtype="uint8")
        ramp = np.linspace(0, 255, width, dtype="uint8")
        data[0, :, :] = ramp[None, :]
        data[1, :, :] = ramp[::-1][None, :]
        data[2, :, :] = 128
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=3,
            dtype="uint8",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data)
        return path

    def _chip_mask(self, path: Path, crs: str = "EPSG:32633") -> Path:
        """A mask on the chip's own grid, inside the scene's footprint."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        size = 64
        transform = from_bounds(501000, 5001000, 501640, 5001640, size, size)
        mask = np.zeros((size, size), dtype="uint8")
        mask[10:40, 10:40] = 1  # field interior
        mask[40:45, 10:40] = 2  # boundary
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=size,
            height=size,
            count=1,
            dtype="uint8",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(mask, 1)
        return path

    def test_reads_the_chip_window_not_the_whole_scene(self, tmp_path: Path) -> None:
        from PIL import Image

        from ftw_dataset_tools.api.imagery.thumbnails import generate_scene_thumbnail

        scene = self._scene(tmp_path / "scene.tif")
        mask = self._chip_mask(tmp_path / "chip_semantic_3_class.tif")
        out = generate_scene_thumbnail(str(scene), mask, tmp_path / "preview.jpg")

        assert out.exists()
        with Image.open(out) as img:
            assert img.mode == "RGB"
            # Square chip grid -> square preview, capped by max_size.
            assert img.width == img.height
            assert max(img.size) <= 512

    def test_preview_matches_the_mask_aspect_so_the_overlay_registers(self, tmp_path: Path) -> None:
        """A non-square chip must not be previewed square, or the overlay shears."""
        import numpy as np
        import rasterio
        from PIL import Image
        from rasterio.transform import from_bounds

        from ftw_dataset_tools.api.imagery.thumbnails import generate_scene_thumbnail

        scene = self._scene(tmp_path / "scene.tif")
        wide = tmp_path / "wide_mask.tif"
        with rasterio.open(
            wide,
            "w",
            driver="GTiff",
            width=120,
            height=60,
            count=1,
            dtype="uint8",
            crs="EPSG:32633",
            transform=from_bounds(501000, 5001000, 502200, 5001600, 120, 60),
        ) as dst:
            dst.write(np.zeros((60, 120), dtype="uint8"), 1)

        out = generate_scene_thumbnail(str(scene), wide, tmp_path / "wide.jpg")
        with Image.open(out) as img:
            assert img.width == 2 * img.height

    def test_overlay_composites_onto_a_remote_sourced_preview(self, tmp_path: Path) -> None:
        """The mask overlay works on a preview that never came from a local clip."""
        from PIL import Image

        from ftw_dataset_tools.api.imagery.thumbnails import (
            generate_overlay_thumbnail,
            generate_scene_thumbnail,
        )

        scene = self._scene(tmp_path / "scene.tif")
        mask = self._chip_mask(tmp_path / "chip_semantic_3_class.tif")
        base = generate_scene_thumbnail(str(scene), mask, tmp_path / "base.jpg")
        overlay = generate_overlay_thumbnail(base, mask, tmp_path / "overlay.jpg")

        with Image.open(base) as b, Image.open(overlay) as o:
            assert o.size == b.size
            assert o.tobytes() != b.tobytes(), "overlay changed nothing"

    def test_missing_reference_raises_thumbnail_error(self, tmp_path: Path) -> None:
        import pytest

        from ftw_dataset_tools.api.imagery.thumbnails import (
            ThumbnailError,
            generate_scene_thumbnail,
        )

        scene = self._scene(tmp_path / "scene.tif")
        with pytest.raises(ThumbnailError, match="Reference raster does not exist"):
            generate_scene_thumbnail(str(scene), tmp_path / "nope.tif", tmp_path / "x.jpg")

    def test_unreadable_scene_raises_thumbnail_error(self, tmp_path: Path) -> None:
        import pytest

        from ftw_dataset_tools.api.imagery.thumbnails import (
            ThumbnailError,
            generate_scene_thumbnail,
        )

        mask = self._chip_mask(tmp_path / "chip_semantic_3_class.tif")
        broken = tmp_path / "broken.tif"
        broken.write_bytes(b"not a raster")
        with pytest.raises(ThumbnailError, match="Failed to read"):
            generate_scene_thumbnail(str(broken), mask, tmp_path / "x.jpg")
