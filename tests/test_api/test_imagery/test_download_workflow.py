"""Tests for download_workflow module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pystac

from ftw_dataset_tools.api.imagery.download_workflow import (
    DownloadWorkflowResult,
    download_imagery_for_catalog,
    find_s2_child_items,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestFindS2ChildItems:
    """Tests for find_s2_child_items function."""

    def test_finds_planting_items(self, mock_catalog_with_s2_children: Path) -> None:
        """Test that planting S2 child items are found."""
        items = find_s2_child_items(mock_catalog_with_s2_children)

        planting_ids = {item.id for item, _ in items if "_planting_s2" in item.id}
        assert len(planting_ids) == 2

    def test_finds_harvest_items(self, mock_catalog_with_s2_children: Path) -> None:
        """Test that harvest S2 child items are found."""
        items = find_s2_child_items(mock_catalog_with_s2_children)

        harvest_ids = {item.id for item, _ in items if "_harvest_s2" in item.id}
        assert len(harvest_ids) == 2

    def test_skips_parent_items(self, mock_catalog_with_s2_children: Path) -> None:
        """Test that parent items are not included."""
        items = find_s2_child_items(mock_catalog_with_s2_children)

        # All items should be child items
        for item, _ in items:
            assert item.id.endswith("_planting_s2") or item.id.endswith("_harvest_s2")

    def test_returns_empty_for_empty_dir(self, mock_catalog_empty: Path) -> None:
        """Test that empty list is returned for empty catalog."""
        items = find_s2_child_items(mock_catalog_empty)

        assert items == []

    def test_skips_hidden_dirs(self, tmp_path: Path) -> None:
        """Test that hidden directories are skipped."""
        from .conftest import create_mock_s2_assets, create_mock_stac_item

        # Create hidden directory with S2 child item
        hidden_dir = tmp_path / ".hidden"
        hidden_dir.mkdir()
        hidden_item = create_mock_stac_item(
            item_id="hidden_planting_s2",
            assets=create_mock_s2_assets(),
        )
        hidden_item.set_self_href(str(hidden_dir / "hidden_planting_s2.json"))
        hidden_item.save_object(dest_href=str(hidden_dir / "hidden_planting_s2.json"))

        # Create normal directory with S2 child item
        normal_dir = tmp_path / "chip_001"
        normal_dir.mkdir()
        normal_item = create_mock_stac_item(
            item_id="chip_001_planting_s2",
            bbox=(10.0, 50.0, 10.01, 50.01),
            properties={"eo:cloud_cover": 1.5},
            assets=create_mock_s2_assets(),
        )
        normal_item.set_self_href(str(normal_dir / "chip_001_planting_s2.json"))
        normal_item.save_object(dest_href=str(normal_dir / "chip_001_planting_s2.json"))

        items = find_s2_child_items(tmp_path)

        assert len(items) == 1
        assert items[0][0].id == "chip_001_planting_s2"

    def test_skips_invalid_json(self, tmp_path: Path) -> None:
        """Test that invalid JSON files are skipped."""
        from .conftest import create_mock_s2_assets, create_mock_stac_item

        # Create directory with invalid JSON
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()
        invalid_json = invalid_dir / "invalid_planting_s2.json"
        invalid_json.write_text("{ invalid json }")

        # Create directory with valid item
        valid_dir = tmp_path / "chip_001"
        valid_dir.mkdir()
        valid_item = create_mock_stac_item(
            item_id="chip_001_planting_s2",
            bbox=(10.0, 50.0, 10.01, 50.01),
            properties={"eo:cloud_cover": 1.5},
            assets=create_mock_s2_assets(),
        )
        valid_item.set_self_href(str(valid_dir / "chip_001_planting_s2.json"))
        valid_item.save_object(dest_href=str(valid_dir / "chip_001_planting_s2.json"))

        items = find_s2_child_items(tmp_path)

        assert len(items) == 1
        assert items[0][0].id == "chip_001_planting_s2"

    def test_returns_item_paths(self, mock_catalog_with_s2_children: Path) -> None:
        """Test that correct paths are returned with items."""
        items = find_s2_child_items(mock_catalog_with_s2_children)

        for item, item_path in items:
            assert item_path.exists()
            assert item_path.name == f"{item.id}.json"


class TestDownloadImageryForCatalog:
    """Tests for download_imagery_for_catalog function."""

    def test_returns_empty_for_no_items(self, mock_catalog_empty: Path) -> None:
        """Test returns empty result when no S2 items found."""
        result = download_imagery_for_catalog(
            catalog_dir=mock_catalog_empty,
        )

        assert result.successful == 0
        assert result.skipped == 0
        assert result.failed == 0

    def test_uses_default_bands(self, mock_catalog_with_s2_children: Path) -> None:
        """Test that default bands are used when not specified."""
        with (
            patch(
                "ftw_dataset_tools.api.imagery.download_workflow.download_and_clip_scene"
            ) as mock_download,
            patch("ftw_dataset_tools.api.imagery.download_workflow.process_downloaded_scene"),
            patch("ftw_dataset_tools.api.imagery.download_workflow.has_rgb_bands") as mock_has_rgb,
        ):
            mock_download.return_value = MagicMock(success=True)
            mock_has_rgb.return_value = True

            download_imagery_for_catalog(
                catalog_dir=mock_catalog_with_s2_children,
                show_progress_bar=False,
            )

            # Check that default bands were passed to download function
            call_kwargs = mock_download.call_args_list[0][1]
            assert call_kwargs["bands"] == ["red", "green", "blue", "nir"]

    def test_skips_items_without_bbox(self, tmp_path: Path) -> None:
        """Test that items without bbox are skipped."""
        import json

        # Create child item without bbox in JSON
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        # Write a minimal STAC item JSON without bbox
        item_json = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "chip_001_planting_s2",
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            "properties": {"datetime": "2024-01-01T00:00:00Z", "eo:cloud_cover": 1.5},
            "links": [],
            "assets": {
                "red": {"href": "https://example.com/red.tif"},
                "green": {"href": "https://example.com/green.tif"},
                "blue": {"href": "https://example.com/blue.tif"},
                "nir": {"href": "https://example.com/nir.tif"},
            },
        }
        # Explicitly omit bbox from JSON
        item_path = chip_dir / "chip_001_planting_s2.json"
        item_path.write_text(json.dumps(item_json))

        result = download_imagery_for_catalog(
            catalog_dir=tmp_path,
            show_progress_bar=False,
        )

        assert result.skipped == 1
        assert result.skipped_details[0]["reason"] == "No bbox in item"

    def test_resume_skips_existing(self, tmp_path: Path) -> None:
        """Test that resume mode skips already downloaded items."""
        from .conftest import create_mock_s2_assets, create_mock_stac_item

        # Create child item with local image asset
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        item = create_mock_stac_item(
            item_id="chip_001_planting_s2",
            bbox=(10.0, 50.0, 10.01, 50.01),
            properties={"eo:cloud_cover": 1.5},
            assets=create_mock_s2_assets(),
        )

        # Add local image asset
        item.assets["image"] = pystac.Asset(
            href="./chip_001_planting_image_s2.tif",
            media_type="image/tiff",
        )

        item.set_self_href(str(chip_dir / "chip_001_planting_s2.json"))
        item.save_object(dest_href=str(chip_dir / "chip_001_planting_s2.json"))

        # Create the local file
        local_file = chip_dir / "chip_001_planting_image_s2.tif"
        local_file.write_bytes(b"fake image data")

        result = download_imagery_for_catalog(
            catalog_dir=tmp_path,
            resume=True,
            show_progress_bar=False,
        )

        assert result.skipped == 1
        assert "Already downloaded" in result.skipped_details[0]["reason"]

    def test_resume_downloads_missing(self, tmp_path: Path) -> None:
        """Test that resume mode downloads when local file doesn't exist."""
        from .conftest import create_mock_s2_assets, create_mock_stac_item

        # Create child item with image asset but no local file
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        item = create_mock_stac_item(
            item_id="chip_001_planting_s2",
            bbox=(10.0, 50.0, 10.01, 50.01),
            properties={"eo:cloud_cover": 1.5},
            assets=create_mock_s2_assets(),
        )

        # Add image asset but don't create the file
        item.assets["image"] = pystac.Asset(
            href="./chip_001_planting_image_s2.tif",
            media_type="image/tiff",
        )

        item.set_self_href(str(chip_dir / "chip_001_planting_s2.json"))
        item.save_object(dest_href=str(chip_dir / "chip_001_planting_s2.json"))

        with (
            patch(
                "ftw_dataset_tools.api.imagery.download_workflow.download_and_clip_scene"
            ) as mock_download,
            patch("ftw_dataset_tools.api.imagery.download_workflow.process_downloaded_scene"),
            patch("ftw_dataset_tools.api.imagery.download_workflow.has_rgb_bands") as mock_has_rgb,
        ):
            mock_download.return_value = MagicMock(success=True)
            mock_has_rgb.return_value = True

            result = download_imagery_for_catalog(
                catalog_dir=tmp_path,
                resume=True,
                show_progress_bar=False,
            )

            assert result.successful == 1
            assert mock_download.called

    def test_determines_season_planting(self, mock_catalog_with_s2_children: Path) -> None:
        """Test that planting season is determined from item ID."""
        with (
            patch(
                "ftw_dataset_tools.api.imagery.download_workflow.download_and_clip_scene"
            ) as mock_download,
            patch(
                "ftw_dataset_tools.api.imagery.download_workflow.process_downloaded_scene"
            ) as mock_process,
            patch("ftw_dataset_tools.api.imagery.download_workflow.has_rgb_bands") as mock_has_rgb,
        ):
            mock_download.return_value = MagicMock(success=True)
            mock_has_rgb.return_value = True

            download_imagery_for_catalog(
                catalog_dir=mock_catalog_with_s2_children,
                show_progress_bar=False,
            )

            # Check that some calls have "planting" season
            planting_calls = [
                call for call in mock_process.call_args_list if call[1].get("season") == "planting"
            ]
            assert len(planting_calls) == 2

    def test_determines_season_harvest(self, mock_catalog_with_s2_children: Path) -> None:
        """Test that harvest season is determined from item ID."""
        with (
            patch(
                "ftw_dataset_tools.api.imagery.download_workflow.download_and_clip_scene"
            ) as mock_download,
            patch(
                "ftw_dataset_tools.api.imagery.download_workflow.process_downloaded_scene"
            ) as mock_process,
            patch("ftw_dataset_tools.api.imagery.download_workflow.has_rgb_bands") as mock_has_rgb,
        ):
            mock_download.return_value = MagicMock(success=True)
            mock_has_rgb.return_value = True

            download_imagery_for_catalog(
                catalog_dir=mock_catalog_with_s2_children,
                show_progress_bar=False,
            )

            # Check that some calls have "harvest" season
            harvest_calls = [
                call for call in mock_process.call_args_list if call[1].get("season") == "harvest"
            ]
            assert len(harvest_calls) == 2

    def test_successful_download(self, mock_catalog_with_s2_children: Path) -> None:
        """Test successful download processing."""
        with (
            patch(
                "ftw_dataset_tools.api.imagery.download_workflow.download_and_clip_scene"
            ) as mock_download,
            patch(
                "ftw_dataset_tools.api.imagery.download_workflow.process_downloaded_scene"
            ) as mock_process,
            patch("ftw_dataset_tools.api.imagery.download_workflow.has_rgb_bands") as mock_has_rgb,
        ):
            mock_download.return_value = MagicMock(success=True)
            mock_has_rgb.return_value = True

            result = download_imagery_for_catalog(
                catalog_dir=mock_catalog_with_s2_children,
                show_progress_bar=False,
            )

            assert result.successful == 4  # 2 chips x 2 seasons
            assert result.failed == 0
            assert mock_process.call_count == 4

    def test_failed_download(self, mock_catalog_with_s2_children: Path) -> None:
        """Test handling of failed downloads."""
        with (
            patch(
                "ftw_dataset_tools.api.imagery.download_workflow.download_and_clip_scene"
            ) as mock_download,
            patch("ftw_dataset_tools.api.imagery.download_workflow.has_rgb_bands") as mock_has_rgb,
        ):
            mock_download.return_value = MagicMock(success=False, error="Download failed")
            mock_has_rgb.return_value = True

            result = download_imagery_for_catalog(
                catalog_dir=mock_catalog_with_s2_children,
                show_progress_bar=False,
            )

            assert result.successful == 0
            assert result.failed == 4
            assert "Download failed" in result.failed_details[0]["error"]

    def test_handles_exceptions(self, mock_catalog_with_s2_children: Path) -> None:
        """Test that exceptions are caught and recorded."""
        with (
            patch(
                "ftw_dataset_tools.api.imagery.download_workflow.download_and_clip_scene"
            ) as mock_download,
            patch("ftw_dataset_tools.api.imagery.download_workflow.has_rgb_bands") as mock_has_rgb,
        ):
            mock_download.side_effect = RuntimeError("Network error")
            mock_has_rgb.return_value = True

            result = download_imagery_for_catalog(
                catalog_dir=mock_catalog_with_s2_children,
                show_progress_bar=False,
            )

            assert result.failed == 4
            assert "Network error" in result.failed_details[0]["error"]

    def test_progress_callbacks(self, mock_catalog_with_s2_children: Path) -> None:
        """Test that progress callbacks are called."""
        with (
            patch(
                "ftw_dataset_tools.api.imagery.download_workflow.download_and_clip_scene"
            ) as mock_download,
            patch("ftw_dataset_tools.api.imagery.download_workflow.process_downloaded_scene"),
            patch("ftw_dataset_tools.api.imagery.download_workflow.has_rgb_bands") as mock_has_rgb,
        ):
            mock_download.return_value = MagicMock(success=True)
            mock_has_rgb.return_value = True

            progress_calls = []

            def on_progress(current: int, total: int) -> None:
                progress_calls.append((current, total))

            download_imagery_for_catalog(
                catalog_dir=mock_catalog_with_s2_children,
                on_progress=on_progress,
                show_progress_bar=False,
            )

            # Should have 4 progress calls (2 chips x 2 seasons)
            assert len(progress_calls) == 4
            assert progress_calls[-1] == (4, 4)

    def test_progress_bar_integration(self, mock_catalog_with_s2_children: Path) -> None:
        """Test that tqdm progress bar is used when show_progress_bar=True."""
        with (
            patch(
                "ftw_dataset_tools.api.imagery.download_workflow.download_and_clip_scene"
            ) as mock_download,
            patch("ftw_dataset_tools.api.imagery.download_workflow.process_downloaded_scene"),
            patch("ftw_dataset_tools.api.imagery.download_workflow.has_rgb_bands") as mock_has_rgb,
            patch("ftw_dataset_tools.api.imagery.download_workflow.tqdm") as mock_tqdm,
        ):
            mock_download.return_value = MagicMock(success=True)
            mock_has_rgb.return_value = True
            mock_progress = MagicMock()
            mock_tqdm.return_value = mock_progress

            download_imagery_for_catalog(
                catalog_dir=mock_catalog_with_s2_children,
                show_progress_bar=True,
            )

            # Progress bar should be created
            assert mock_tqdm.called
            # Progress bar should be updated
            assert mock_progress.update.call_count == 4
            # Progress bar should be closed
            assert mock_progress.close.called

    def test_thumbnail_generation(self, mock_catalog_with_s2_children: Path) -> None:
        """Test that thumbnail generation flag is passed through."""
        with (
            patch(
                "ftw_dataset_tools.api.imagery.download_workflow.download_and_clip_scene"
            ) as mock_download,
            patch(
                "ftw_dataset_tools.api.imagery.download_workflow.process_downloaded_scene"
            ) as mock_process,
            patch("ftw_dataset_tools.api.imagery.download_workflow.has_rgb_bands") as mock_has_rgb,
        ):
            mock_download.return_value = MagicMock(success=True)
            mock_has_rgb.return_value = True

            download_imagery_for_catalog(
                catalog_dir=mock_catalog_with_s2_children,
                generate_thumbnails=True,
                show_progress_bar=False,
            )

            # Check that generate_thumbnails was passed to process function
            call_kwargs = mock_process.call_args_list[0][1]
            assert call_kwargs["generate_thumbnails"] is True

    def test_no_thumbnails_without_rgb(self, mock_catalog_with_s2_children: Path) -> None:
        """Test that thumbnails are not generated without RGB bands."""
        with (
            patch(
                "ftw_dataset_tools.api.imagery.download_workflow.download_and_clip_scene"
            ) as mock_download,
            patch(
                "ftw_dataset_tools.api.imagery.download_workflow.process_downloaded_scene"
            ) as mock_process,
            patch("ftw_dataset_tools.api.imagery.download_workflow.has_rgb_bands") as mock_has_rgb,
        ):
            mock_download.return_value = MagicMock(success=True)
            mock_has_rgb.return_value = False  # No RGB bands

            download_imagery_for_catalog(
                catalog_dir=mock_catalog_with_s2_children,
                generate_thumbnails=True,  # Requested but can't do
                show_progress_bar=False,
            )

            # Check that generate_thumbnails is False when no RGB bands
            call_kwargs = mock_process.call_args_list[0][1]
            assert call_kwargs["generate_thumbnails"] is False

    def test_custom_bands(self, mock_catalog_with_s2_children: Path) -> None:
        """Test that custom bands are passed through."""
        with (
            patch(
                "ftw_dataset_tools.api.imagery.download_workflow.download_and_clip_scene"
            ) as mock_download,
            patch("ftw_dataset_tools.api.imagery.download_workflow.process_downloaded_scene"),
            patch("ftw_dataset_tools.api.imagery.download_workflow.has_rgb_bands") as mock_has_rgb,
        ):
            mock_download.return_value = MagicMock(success=True)
            mock_has_rgb.return_value = True

            custom_bands = ["red", "nir", "swir16"]

            download_imagery_for_catalog(
                catalog_dir=mock_catalog_with_s2_children,
                bands=custom_bands,
                show_progress_bar=False,
            )

            call_kwargs = mock_download.call_args_list[0][1]
            assert call_kwargs["bands"] == custom_bands

    def test_custom_resolution(self, mock_catalog_with_s2_children: Path) -> None:
        """Test that custom resolution is passed through."""
        with (
            patch(
                "ftw_dataset_tools.api.imagery.download_workflow.download_and_clip_scene"
            ) as mock_download,
            patch("ftw_dataset_tools.api.imagery.download_workflow.process_downloaded_scene"),
            patch("ftw_dataset_tools.api.imagery.download_workflow.has_rgb_bands") as mock_has_rgb,
        ):
            mock_download.return_value = MagicMock(success=True)
            mock_has_rgb.return_value = True

            download_imagery_for_catalog(
                catalog_dir=mock_catalog_with_s2_children,
                resolution=20.0,
                show_progress_bar=False,
            )

            call_kwargs = mock_download.call_args_list[0][1]
            assert call_kwargs["resolution"] == 20.0


class TestDownloadWorkflowResult:
    """Tests for DownloadWorkflowResult dataclass."""

    def test_default_values(self) -> None:
        """Test that default values are initialized correctly."""
        result = DownloadWorkflowResult()

        assert result.successful == 0
        assert result.skipped == 0
        assert result.failed == 0
        assert result.skipped_details == []
        assert result.failed_details == []

    def test_mutable_defaults_are_independent(self) -> None:
        """Test that list defaults are independent between instances."""
        result1 = DownloadWorkflowResult()
        result2 = DownloadWorkflowResult()

        result1.failed_details.append({"item": "test"})

        assert len(result1.failed_details) == 1
        assert len(result2.failed_details) == 0
