"""Tests for image_download module helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pystac
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely import segmentize
from shapely.geometry import box, mapping
from shapely.ops import transform as shapely_transform

from ftw_dataset_tools.api.chip_grid import utm_grid_for_cell
from ftw_dataset_tools.api.imagery.image_download import (
    _missing_bands_error,
    compute_target_grid,
    download_and_clip_scene,
    find_reference_mask_for_output,
    on_reference_grid,
)
from ftw_dataset_tools.api.imagery.settings import BANDS_OF_INTEREST

if TYPE_CHECKING:
    from pathlib import Path

    from ftw_dataset_tools.api.imagery.scene_selection import SelectedScene


class TestFindReferenceMaskForOutput:
    """Tests for reference mask auto-detection."""

    def test_prefers_semantic_3_class(self, tmp_path: Path) -> None:
        """Picks semantic_3_class when multiple masks are available."""
        output_path = tmp_path / "chip_001_2024_planting_image_s2.tif"
        (tmp_path / "chip_001_2024_instance.tif").touch()
        expected = tmp_path / "chip_001_2024_semantic_3_class.tif"
        expected.touch()

        found = find_reference_mask_for_output(output_path)

        assert found == expected

    def test_falls_back_to_semantic_2_class(self, tmp_path: Path) -> None:
        """Falls back to semantic_2_class when semantic_3_class is missing."""
        output_path = tmp_path / "chip_001_2024_harvest_image_s2.tif"
        expected = tmp_path / "chip_001_2024_semantic_2_class.tif"
        expected.touch()

        found = find_reference_mask_for_output(output_path)

        assert found == expected

    def test_falls_back_to_instance(self, tmp_path: Path) -> None:
        """Falls back to instance mask when semantic masks are missing."""
        output_path = tmp_path / "chip_001_2024_planting_image_s2.tif"
        expected = tmp_path / "chip_001_2024_instance.tif"
        expected.touch()

        found = find_reference_mask_for_output(output_path)

        assert found == expected

    def test_returns_none_when_not_matching_pattern(self, tmp_path: Path) -> None:
        """Returns None for filenames that do not match imagery naming pattern."""
        output_path = tmp_path / "custom_output.tif"

        found = find_reference_mask_for_output(output_path)

        assert found is None


def _ftw_chip(minx: float = 500_000, miny: float = 5_540_000):
    """A 2 km FTW chip in zone 32N as the grid stores it: in lon/lat, edge midpoints included."""
    to_lonlat = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True).transform
    return shapely_transform(to_lonlat, segmentize(box(minx, miny, minx + 2000, miny + 2000), 1000))


def _write_raster(path: Path, crs: str, transform, size: tuple[int, int]) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "width": size[0],
        "height": size[1],
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.zeros((1, size[1], size[0]), dtype="uint8"))


class TestComputeTargetGrid:
    """The output grid when no mask sits beside the image."""

    def test_fallback_is_the_chips_mask_grid(self, tmp_path: Path) -> None:
        chip_dir = tmp_path / "ftw-32UMA0101_2024"
        chip_dir.mkdir()
        chip = _ftw_chip()

        grid, error = compute_target_grid(
            bbox=chip.bounds,
            output_path=chip_dir / "ftw-32UMA0101_2024_planting_image_s2.tif",
            resolution=10.0,
            reference_raster=None,
            cell=chip,
        )

        assert error is None
        crs, transform, width, height, reference = grid
        mask_grid = utm_grid_for_cell("ftw-32UMA0101", chip.bounds, "EPSG:4326", 10.0, chip)
        assert (crs, transform, width, height) == mask_grid
        assert crs.to_string() == "EPSG:32632"
        assert (width, height) == (200, 200)
        assert (transform.c, transform.f) == pytest.approx((500_000, 5_542_000))
        assert reference is None


class TestOnReferenceGrid:
    """A resumed download keeps an image only when it is on its mask's grid."""

    def _chip(self, tmp_path: Path) -> Path:
        chip_dir = tmp_path / "ftw-32UMA0101_2024"
        chip_dir.mkdir()
        _write_raster(
            chip_dir / "ftw-32UMA0101_2024_semantic_3_class.tif",
            "EPSG:32632",
            from_origin(500_000, 5_542_000, 10, 10),
            (200, 200),
        )
        return chip_dir / "ftw-32UMA0101_2024_planting_image_s2.tif"

    def test_matching_image_is_kept(self, tmp_path: Path) -> None:
        image = self._chip(tmp_path)
        _write_raster(image, "EPSG:32632", from_origin(500_000, 5_542_000, 10, 10), (200, 200))

        assert on_reference_grid(image)

    def test_lonlat_image_beside_a_utm_mask_is_not(self, tmp_path: Path) -> None:
        image = self._chip(tmp_path)
        _write_raster(image, "EPSG:4326", from_origin(9.0, 50.0, 0.00009, 0.00009), (310, 200))

        assert not on_reference_grid(image)

    def test_image_without_a_mask_is_kept(self, tmp_path: Path) -> None:
        image = tmp_path / "ftw-32UMA0101_2024_planting_image_s2.tif"
        _write_raster(image, "EPSG:4326", from_origin(9.0, 50.0, 0.00009, 0.00009), (310, 200))

        assert on_reference_grid(image)

    def test_stale_mask_beside_rebuilt_ones_is_caught(self, tmp_path: Path) -> None:
        image = self._chip(tmp_path)
        _write_raster(image, "EPSG:32632", from_origin(500_000, 5_542_000, 10, 10), (200, 200))
        # A rerun rebuilt the instance mask on UTM but left a lon/lat 2-class mask behind.
        _write_raster(
            image.parent / "ftw-32UMA0101_2024_semantic_2_class.tif",
            "EPSG:4326",
            from_origin(9.0, 50.0, 0.00009, 0.00009),
            (310, 200),
        )

        assert not on_reference_grid(image)

    def test_resume_flags_a_misregistered_image(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.imagery.download_workflow import skip_download_reason

        image = self._chip(tmp_path)
        _write_raster(image, "EPSG:4326", from_origin(9.0, 50.0, 0.00009, 0.00009), (310, 200))
        item = pystac.Item(
            "ftw-32UMA0101_2024_planting_s2",
            mapping(_ftw_chip()),
            list(_ftw_chip().bounds),
            datetime(2024, 5, 1, tzinfo=UTC),
            {},
        )
        item.add_asset("image", pystac.Asset(href=f"./{image.name}"))
        item_path = image.parent / f"{item.id}.json"

        # Its band links are gone, so it is reported for select-images --force, not queued.
        reason = skip_download_reason(item, item_path, resume=True)
        assert reason is not None
        assert "select-images --force" in reason
        _write_raster(image, "EPSG:32632", from_origin(500_000, 5_542_000, 10, 10), (200, 200))
        assert skip_download_reason(item, item_path, resume=True) == "Already downloaded"


class TestDownloadAndClipScene:
    """Tests for download_and_clip_scene validation behavior."""

    @staticmethod
    def _write_single_band_raster(
        path: Path,
        data: np.ndarray,
        transform,
        crs: str = "EPSG:4326",
    ) -> None:
        """Write a single-band test raster."""
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=data.shape[1],
            height=data.shape[0],
            count=1,
            dtype=str(data.dtype),
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data, 1)

    def test_uses_explicit_reference_raster_override_happy_path(
        self,
        tmp_path: Path,
        mock_selected_scene: SelectedScene,
    ) -> None:
        """Uses explicit reference_raster override instead of auto-detected mask."""
        output_path = tmp_path / "chip_001_2024_planting_image_s2.tif"

        # Auto-detected mask candidate (should be ignored due to explicit override)
        auto_mask = tmp_path / "chip_001_2024_semantic_3_class.tif"
        self._write_single_band_raster(
            auto_mask,
            np.zeros((3, 4), dtype=np.uint8),
            from_origin(10.0, 50.01, 0.0025, 0.0025),
        )

        explicit_reference = tmp_path / "explicit_reference.tif"
        self._write_single_band_raster(
            explicit_reference,
            np.zeros((6, 7), dtype=np.uint8),
            from_origin(10.0, 50.01, 0.0014285714285714286, 0.0016666666666666668),
        )

        source_path = tmp_path / "source_red.tif"
        source_data = np.arange(100, dtype=np.uint16).reshape(10, 10)
        self._write_single_band_raster(
            source_path,
            source_data,
            from_origin(10.0, 50.01, 0.001, 0.001),
        )

        mock_selected_scene.item.assets["red"].href = str(source_path)

        result = download_and_clip_scene(
            scene=mock_selected_scene,
            bbox=(10.0, 50.0, 10.01, 50.01),
            output_path=output_path,
            bands=["red"],
            reference_raster=explicit_reference,
            resolution=10.0,
        )

        assert result.success is True
        assert result.width == 7
        assert result.height == 6
        assert result.crs == "EPSG:4326"

        with rasterio.open(output_path) as output_ds:
            assert output_ds.width == 7
            assert output_ds.height == 6

    def test_returns_structured_failure_for_nonexistent_reference_raster(
        self,
        tmp_path: Path,
        mock_selected_scene: SelectedScene,
    ) -> None:
        """Returns DownloadResult failure when explicit reference_raster does not exist."""
        output_path = tmp_path / "chip_001_2024_planting_image_s2.tif"
        missing_reference = tmp_path / "missing_reference.tif"

        result = download_and_clip_scene(
            scene=mock_selected_scene,
            bbox=(10.0, 50.0, 10.01, 50.01),
            output_path=output_path,
            bands=["red"],
            reference_raster=missing_reference,
            resolution=10.0,
        )

        assert result.success is False
        assert result.error is not None
        assert "Failed to use reference raster" in result.error

    def test_validation_failure_cleans_up_output_file(
        self,
        tmp_path: Path,
        mock_selected_scene: SelectedScene,
    ) -> None:
        """Removes output file and returns error when alignment validation fails."""
        output_path = tmp_path / "chip_001_2024_planting_image_s2.tif"
        reference_mask = tmp_path / "chip_001_2024_semantic_3_class.tif"
        source_path = tmp_path / "source_red.tif"

        self._write_single_band_raster(
            reference_mask,
            np.zeros((8, 8), dtype=np.uint8),
            from_origin(10.0, 50.01, 0.00125, 0.00125),
        )
        self._write_single_band_raster(
            source_path,
            np.arange(100, dtype=np.uint16).reshape(10, 10),
            from_origin(10.0, 50.01, 0.001, 0.001),
        )

        mock_selected_scene.item.assets["red"].href = str(source_path)

        real_rasterio_open = rasterio.open
        reference_open_calls = {"count": 0}

        def open_with_validation_failure(path, *args, **kwargs):
            mode = kwargs.get("mode") if "mode" in kwargs else (args[0] if args else "r")
            path_str = str(path)
            if path_str == str(reference_mask) and mode == "r":
                reference_open_calls["count"] += 1
                if reference_open_calls["count"] >= 2:
                    raise RuntimeError("forced validation failure")
            return real_rasterio_open(path, *args, **kwargs)

        with patch("ftw_dataset_tools.api.imagery.image_download.rasterio.open") as mock_open:
            mock_open.side_effect = open_with_validation_failure

            result = download_and_clip_scene(
                scene=mock_selected_scene,
                bbox=(10.0, 50.0, 10.01, 50.01),
                output_path=output_path,
                bands=["red"],
                resolution=10.0,
            )

        assert result.success is False
        assert result.error is not None
        assert "Failed to validate output alignment" in result.error
        assert output_path.exists() is False

    def test_fails_for_non_positive_resolution_without_reference_mask(
        self,
        tmp_path: Path,
        mock_selected_scene: SelectedScene,
    ) -> None:
        """Returns structured failure before any raster read for invalid fallback resolution."""
        output_path = tmp_path / "chip_001_2024_planting_image_s2.tif"

        with patch("ftw_dataset_tools.api.imagery.image_download.rasterio.open") as mock_open:
            result = download_and_clip_scene(
                scene=mock_selected_scene,
                bbox=(10.0, 50.0, 10.01, 50.01),
                output_path=output_path,
                resolution=0.0,
            )

        assert result.success is False
        assert result.error is not None
        assert "Resolution must be > 0" in result.error
        assert mock_open.call_count == 0

    def test_uses_reference_mask_even_with_non_positive_resolution(
        self,
        tmp_path: Path,
        mock_selected_scene: SelectedScene,
    ) -> None:
        """Does not fail on non-positive resolution when reference grid is provided."""
        output_path = tmp_path / "chip_001_2024_planting_image_s2.tif"
        reference_mask = tmp_path / "chip_001_2024_semantic_3_class.tif"

        mask_data = np.zeros((8, 8), dtype=np.uint8)
        with rasterio.open(
            reference_mask,
            "w",
            driver="GTiff",
            width=8,
            height=8,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_origin(10.0, 50.01, 0.00125, 0.00125),
        ) as dst:
            dst.write(mask_data, 1)

        # We only validate that resolution guard is bypassed. Source reads may still fail
        # due to remote mock assets, but not because of non-positive resolution.
        result = download_and_clip_scene(
            scene=mock_selected_scene,
            bbox=(10.0, 50.0, 10.01, 50.01),
            output_path=output_path,
            resolution=0.0,
        )

        assert "Resolution must be > 0" not in (result.error or "")


class TestMissingBandsError:
    """An item with no band assets gets an explanation, not just the symptom."""

    @staticmethod
    def _item(assets: dict[str, str]) -> pystac.Item:
        item = pystac.Item(
            id="chip_000_planting_s2",
            geometry=None,
            bbox=[0.0, 0.0, 1.0, 1.0],
            datetime=datetime(2024, 6, 1, tzinfo=UTC),
            properties={},
        )
        for key, href in assets.items():
            item.assets[key] = pystac.Asset(href=href)
        return item

    def test_local_image_asset_is_reported_as_already_downloaded(self) -> None:
        message = _missing_bands_error(
            self._item({"image": "./chip_000_planting_image_s2.tif"}), BANDS_OF_INTEREST
        )

        assert "already downloaded" in message
        assert "'image'" in message
        # Plain `select-images` skips chips that already have a selection, so only
        # --force actually puts the remote band refs back.
        assert "select-images --force" in message

    def test_keep_remote_refs_clipped_asset_is_reported_too(self) -> None:
        message = _missing_bands_error(
            self._item({"clipped": "./chip_000_planting_image_s2.tif"}), BANDS_OF_INTEREST
        )

        assert "already downloaded" in message
        assert "'clipped'" in message

    def test_an_item_with_no_local_imagery_keeps_the_original_message(self) -> None:
        """Nothing downloaded and no bands either is a genuinely malformed item."""
        message = _missing_bands_error(
            self._item({"scl": "https://example.com/scl.tif"}), BANDS_OF_INTEREST
        )

        assert "No matching band assets found in scene" in message
        assert "scl" in message

    def test_a_band_the_scene_never_carried_lists_what_is_available(self) -> None:
        """--keep-remote-refs keeps the band refs, so this is not an already-downloaded item.

        Child items only ever carry red/green/blue/nir/scl/visual (+ cloud
        probability), so asking for swir16 can never succeed - and saying the
        bands "were replaced by the local 'clipped' asset" is simply false while
        they are sitting right there.
        """
        item = self._item(
            {
                "red": "https://example.com/red.tif",
                "green": "https://example.com/green.tif",
                "blue": "https://example.com/blue.tif",
                "nir": "https://example.com/nir.tif",
                "clipped": "./chip_000_planting_image_s2.tif",
            }
        )

        message = _missing_bands_error(item, ["swir16"])

        assert "already downloaded" not in message
        assert "No matching band assets found in scene" in message
        assert "red" in message

    def test_a_stripped_item_that_kept_ancillary_refs_is_still_already_downloaded(self) -> None:
        """A default download strips red/green/blue/nir but leaves scl and visual behind."""
        item = self._item(
            {
                "scl": "https://example.com/scl.tif",
                "visual": "https://example.com/visual.tif",
                "image": "./chip_000_planting_image_s2.tif",
                "thumbnail": "./chip_000_planting_image_s2.jpg",
            }
        )

        message = _missing_bands_error(item, BANDS_OF_INTEREST)

        assert "already downloaded" in message
        assert "select-images --force" in message


class TestWriteCogStats:
    def test_write_cog_embeds_per_band_stats(self, tmp_path: Path) -> None:
        import numpy as np
        from rasterio.transform import from_bounds

        from ftw_dataset_tools.api.imagery.image_download import write_cog
        from ftw_dataset_tools.api.raster_stats import read_band_stats

        stacked = np.stack(
            [
                np.array([[1, 2], [3, 4]], dtype=np.uint16),
                np.array([[0, 0], [0, 8]], dtype=np.uint16),
            ]
        )
        profile = {
            "driver": "COG",
            "dtype": "uint16",
            "width": 2,
            "height": 2,
            "count": 2,
            "crs": "EPSG:4326",
            "transform": from_bounds(0, 0, 1, 1, 2, 2),
            "compress": "deflate",
        }
        out = tmp_path / "img.tif"

        assert write_cog(out, stacked, ["red", "nir"], profile) is None

        red = read_band_stats(out, 1)
        nir = read_band_stats(out, 2)
        assert red is not None and red.maximum == 4
        assert nir is not None and nir.maximum == 8


class TestProcessDownloadedSceneAssets:
    def test_image_asset_has_size_and_bands(self, tmp_path: Path) -> None:
        import numpy as np
        import pystac
        from rasterio.transform import from_bounds

        from ftw_dataset_tools.api.imagery.image_download import (
            process_downloaded_scene,
            write_cog,
        )

        chip_dir = tmp_path / "chip"
        chip_dir.mkdir()
        parent = pystac.Item(
            id="chip",
            geometry={"type": "Point", "coordinates": [0.5, 0.5]},
            bbox=[0, 0, 1, 1],
            datetime=None,
            properties={
                "start_datetime": "2024-01-01T00:00:00Z",
                "end_datetime": "2024-12-31T00:00:00Z",
            },
        )
        parent_path = chip_dir / "chip.json"
        parent.save_object(dest_href=str(parent_path))

        child = pystac.Item(
            id="chip_planting_s2",
            geometry=parent.geometry,
            bbox=parent.bbox,
            datetime=datetime(2024, 3, 1, tzinfo=UTC),
            properties={},
        )
        child_path = chip_dir / "chip_planting_s2.json"
        child.set_self_href(str(child_path))

        stacked = np.zeros((4, 2, 2), dtype=np.uint16)
        profile = {
            "driver": "COG",
            "dtype": "uint16",
            "width": 2,
            "height": 2,
            "count": 4,
            "crs": "EPSG:4326",
            "transform": from_bounds(0, 0, 1, 1, 2, 2),
            "compress": "deflate",
        }
        image_path = chip_dir / "chip_planting_image_s2.tif"
        write_cog(image_path, stacked, ["red", "green", "blue", "nir"], profile)

        process_downloaded_scene(
            item=child,
            item_path=child_path,
            output_path=image_path,
            output_filename=image_path.name,
            band_list=["red", "green", "blue", "nir"],
            season="planting",
            base_id="chip",
            generate_thumbnails=False,
        )

        saved_child = pystac.Item.from_file(str(child_path))
        image = saved_child.assets["image"]
        assert image.extra_fields["file:size"] == image_path.stat().st_size
        bands = image.extra_fields["raster:bands"]
        assert [b["description"] for b in bands] == ["red", "green", "blue", "nir"]
        assert bands[0]["data_type"] == "uint16"

        saved_parent = pystac.Item.from_file(str(parent_path))
        parent_image = saved_parent.assets["planting_image"]
        assert parent_image.extra_fields["file:size"] == image_path.stat().st_size
        assert len(parent_image.extra_fields["raster:bands"]) == 4


class TestImageryNodata:
    def test_reflectance_only_stack_declares_zero_nodata(self) -> None:
        from ftw_dataset_tools.api.imagery.image_download import stack_nodata

        assert stack_nodata(["red", "green", "blue", "nir"]) == 0
        assert stack_nodata(["coastal", "swir16", "nir08"]) == 0

    def test_mixed_reflectance_and_cloud_stack_declares_no_nodata(self) -> None:
        from ftw_dataset_tools.api.imagery.image_download import stack_nodata

        # GeoTIFF nodata is per-dataset: 0 is a real cloud probability, so
        # declaring it would drop those pixels from the band statistics.
        assert stack_nodata(["red", "green", "blue", "nir", "cloud"]) is None
        assert stack_nodata(["red", "snow"]) is None
        assert stack_nodata(["red", "aot"]) is None
        assert stack_nodata(["red", "wvp"]) is None
        assert stack_nodata(["scl"]) is None
        assert stack_nodata([]) is None

    def test_mixed_stack_keeps_zero_valued_cloud_pixels_in_statistics(self, tmp_path: Path) -> None:
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        from ftw_dataset_tools.api.imagery.image_download import stack_nodata, write_cog
        from ftw_dataset_tools.api.raster_stats import read_band_stats

        found_bands = ["red", "cloud"]
        stacked = np.array(
            [[[0, 5], [7, 0]], [[0, 0], [0, 100]]],
            dtype=np.uint16,
        )
        nodata = stack_nodata(found_bands)
        profile = {
            "driver": "COG",
            "dtype": "uint16",
            "width": 2,
            "height": 2,
            "count": 2,
            "crs": "EPSG:4326",
            "transform": from_bounds(0, 0, 1, 1, 2, 2),
            "compress": "deflate",
            "nodata": nodata,
        }
        out = tmp_path / "mixed.tif"

        assert write_cog(out, stacked, found_bands, profile, nodata=nodata) is None

        cloud_stats = read_band_stats(out, 2)
        assert cloud_stats is not None
        assert cloud_stats.minimum == 0
        assert cloud_stats.maximum == 100
        # No nodata declared, so nothing is excluded and valid_percent is undefined.
        assert cloud_stats.valid_percent is None
        with rasterio.open(out) as src:
            assert src.nodata is None

    def test_write_cog_with_zero_nodata_reports_valid_percent(self, tmp_path: Path) -> None:
        import numpy as np
        from rasterio.transform import from_bounds

        from ftw_dataset_tools.api.imagery.image_download import write_cog
        from ftw_dataset_tools.api.raster_stats import read_band_stats

        stacked = np.array([[[0, 5], [7, 0]]], dtype=np.uint16)
        profile = {
            "driver": "COG",
            "dtype": "uint16",
            "width": 2,
            "height": 2,
            "count": 1,
            "crs": "EPSG:4326",
            "transform": from_bounds(0, 0, 1, 1, 2, 2),
            "compress": "deflate",
            "nodata": 0,
        }
        out = tmp_path / "img.tif"

        assert write_cog(out, stacked, ["red"], profile, nodata=profile["nodata"]) is None

        stats = read_band_stats(out, 1)
        assert stats is not None
        assert stats.minimum == 5
        assert stats.valid_percent == 50.0


class TestProcessDownloadedSceneWithoutAResolvableRoot:
    """A staged child item whose `rel: root` target is missing must still save."""

    def test_writes_the_child_item(self, tmp_path: Path) -> None:
        import json

        import numpy as np
        import pystac
        from rasterio.transform import from_bounds

        from ftw_dataset_tools.api.imagery.image_download import (
            process_downloaded_scene,
            write_cog,
        )

        chip_dir = tmp_path / "chips" / "33UXP" / "chip"
        chip_dir.mkdir(parents=True)
        child_path = chip_dir / "chip_planting_s2.json"
        child = pystac.Item(
            id="chip_planting_s2",
            geometry={"type": "Point", "coordinates": [0.5, 0.5]},
            bbox=[0, 0, 1, 1],
            datetime=datetime(2024, 3, 1, tzinfo=UTC),
            properties={},
        )
        child.add_link(pystac.Link(rel="root", target="../../../../catalog.json"))
        child_path.write_text(
            json.dumps(child.to_dict(include_self_link=False, transform_hrefs=False), indent=2)
        )

        profile = {
            "driver": "COG",
            "dtype": "uint16",
            "width": 2,
            "height": 2,
            "count": 4,
            "crs": "EPSG:4326",
            "transform": from_bounds(0, 0, 1, 1, 2, 2),
            "compress": "deflate",
        }
        image_path = chip_dir / "chip_planting_image_s2.tif"
        write_cog(
            image_path,
            np.zeros((4, 2, 2), dtype=np.uint16),
            ["red", "green", "blue", "nir"],
            profile,
        )

        staged = pystac.Item.from_file(str(child_path))
        process_downloaded_scene(
            item=staged,
            item_path=child_path,
            output_path=image_path,
            output_filename=image_path.name,
            band_list=["red", "green", "blue", "nir"],
            season="planting",
            base_id="chip",
            generate_thumbnails=False,
        )

        written = json.loads(child_path.read_text())
        rels = {link["rel"]: link["href"] for link in written["links"]}
        assert rels["root"] == "../../../../catalog.json"
        assert written["assets"]["image"]["href"] == "./chip_planting_image_s2.tif"


class TestProcessDownloadedSceneParentFailure:
    """A parent-item write failure is surfaced, not swallowed.

    The parent update used to run inside ``contextlib.suppress(Exception)``, so a
    read-only or full destination left the chip without its season image and
    thumbnail assets while the run still counted the scene as downloaded.
    """

    @staticmethod
    def _staged_chip(tmp_path: Path) -> tuple[Path, Path, Path]:
        """Stage parent + child items and a written image; return their paths."""
        import json

        import numpy as np
        import pystac
        from rasterio.transform import from_bounds

        from ftw_dataset_tools.api.imagery.image_download import write_cog

        chip_dir = tmp_path / "chips" / "33UXP" / "chip"
        chip_dir.mkdir(parents=True)

        geometry = {"type": "Point", "coordinates": [0.5, 0.5]}
        parent = pystac.Item(
            id="chip",
            geometry=geometry,
            bbox=[0, 0, 1, 1],
            datetime=datetime(2024, 1, 1, tzinfo=UTC),
            properties={},
        )
        parent_path = chip_dir / "chip.json"
        parent_path.write_text(
            json.dumps(parent.to_dict(include_self_link=False, transform_hrefs=False), indent=2)
        )

        child = pystac.Item(
            id="chip_planting_s2",
            geometry=geometry,
            bbox=[0, 0, 1, 1],
            datetime=datetime(2024, 3, 1, tzinfo=UTC),
            properties={},
        )
        child_path = chip_dir / "chip_planting_s2.json"
        child_path.write_text(
            json.dumps(child.to_dict(include_self_link=False, transform_hrefs=False), indent=2)
        )

        image_path = chip_dir / "chip_planting_image_s2.tif"
        write_cog(
            image_path,
            np.zeros((4, 2, 2), dtype=np.uint16),
            ["red", "green", "blue", "nir"],
            {
                "driver": "COG",
                "dtype": "uint16",
                "width": 2,
                "height": 2,
                "count": 4,
                "crs": "EPSG:4326",
                "transform": from_bounds(0, 0, 1, 1, 2, 2),
                "compress": "deflate",
            },
        )
        return parent_path, child_path, image_path

    def test_raises_when_the_parent_cannot_be_written(self, tmp_path: Path) -> None:
        import pystac
        import pytest

        from ftw_dataset_tools.api.imagery.image_download import process_downloaded_scene
        from ftw_dataset_tools.api.stac_items import STACSaveError

        _parent_path, child_path, image_path = self._staged_chip(tmp_path)

        with (
            patch(
                "ftw_dataset_tools.api.imagery.image_download.update_parent_item",
                side_effect=STACSaveError("destination is read-only"),
            ),
            pytest.raises(STACSaveError, match="read-only"),
        ):
            process_downloaded_scene(
                item=pystac.Item.from_file(str(child_path)),
                item_path=child_path,
                output_path=image_path,
                output_filename=image_path.name,
                band_list=["red", "green", "blue", "nir"],
                season="planting",
                base_id="chip",
                generate_thumbnails=False,
            )

    def test_a_read_only_parent_is_reported_as_a_failed_scene(self, tmp_path: Path) -> None:
        """End to end: the catalog workflow attributes the failure to the scene."""
        import stat

        from ftw_dataset_tools.api.imagery.download_workflow import download_imagery_for_catalog

        parent_path, _child_path, _image_path = self._staged_chip(tmp_path)
        parent_path.chmod(stat.S_IRUSR)

        try:
            with patch(
                "ftw_dataset_tools.api.imagery.download_workflow.download_and_clip_scene"
            ) as mock_download:
                mock_download.return_value = MagicMock(success=True)

                result = download_imagery_for_catalog(
                    catalog_dir=tmp_path,
                    show_progress_bar=False,
                )
        finally:
            parent_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

        assert result.successful == 0
        assert result.failed == 1
        assert result.failed_details[0]["item"] == "chip_planting_s2"
        assert "parent item" in result.failed_details[0]["error"]

    def test_child_item_keeps_its_band_refs_when_the_parent_fails(self, tmp_path: Path) -> None:
        import pystac
        import pytest

        from ftw_dataset_tools.api.imagery.image_download import process_downloaded_scene
        from ftw_dataset_tools.api.stac_items import STACSaveError, write_item

        _parent_path, child_path, image_path = self._staged_chip(tmp_path)
        staged = pystac.Item.from_file(str(child_path))
        staged.add_asset("red", pystac.Asset(href="https://example.com/B04.tif"))
        write_item(staged, child_path)

        with (
            patch(
                "ftw_dataset_tools.api.imagery.image_download.update_parent_item",
                side_effect=STACSaveError("destination is read-only"),
            ),
            pytest.raises(STACSaveError),
        ):
            process_downloaded_scene(
                item=pystac.Item.from_file(str(child_path)),
                item_path=child_path,
                output_path=image_path,
                output_filename=image_path.name,
                band_list=["red", "green", "blue", "nir"],
                season="planting",
                base_id="chip",
                generate_thumbnails=False,
            )

        on_disk = pystac.Item.from_file(str(child_path))
        assert "image" not in on_disk.assets
        assert on_disk.assets["red"].href == "https://example.com/B04.tif"

    def test_a_scene_whose_parent_failed_is_retried_on_resume(self, tmp_path: Path) -> None:
        import stat

        import pystac

        from ftw_dataset_tools.api.imagery.download_workflow import download_imagery_for_catalog

        parent_path, child_path, _image_path = self._staged_chip(tmp_path)

        with patch(
            "ftw_dataset_tools.api.imagery.download_workflow.download_and_clip_scene"
        ) as mock_download:
            mock_download.return_value = MagicMock(success=True)

            parent_path.chmod(stat.S_IRUSR)
            try:
                first = download_imagery_for_catalog(catalog_dir=tmp_path, show_progress_bar=False)
            finally:
                parent_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

            second = download_imagery_for_catalog(catalog_dir=tmp_path, show_progress_bar=False)

        assert first.failed == 1
        assert second.skipped == 0
        assert second.successful == 1
        assert "image" in pystac.Item.from_file(str(child_path)).assets
        assert "planting_image" in pystac.Item.from_file(str(parent_path)).assets
