"""Tests for image_download module helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

from ftw_dataset_tools.api.imagery.image_download import (
    download_and_clip_scene,
    find_reference_mask_for_output,
)

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
