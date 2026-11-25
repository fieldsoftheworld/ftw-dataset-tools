"""Tests for stac_child_items module."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pystac

from ftw_dataset_tools.api.imagery.catalog_ops import (
    QUARTER_NODATA_PROPERTIES,
    clear_chip_selections,
    preserve_imagery_selection,
)
from ftw_dataset_tools.api.imagery.download_workflow import find_s2_child_items
from ftw_dataset_tools.api.imagery.mosaic_selection import MosaicSelectionResult, SelectedMosaic
from ftw_dataset_tools.api.imagery.stac_child_items import (
    _create_season_child_item,
    attach_existing_seasons,
    create_child_items_from_selection,
    create_mosaic_child_items,
    record_quarter_nodata,
)
from ftw_dataset_tools.api.stac import PORTOLAN_SCHEMA_URI

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult


class TestCreateChildItemsFromSelection:
    """Tests for create_child_items_from_selection function."""

    def test_updates_parent_properties(
        self,
        tmp_path: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """Test that parent item is updated with FTW properties."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        create_child_items_from_selection(
            chip_dir=chip_dir,
            parent_item=parent_item,
            result=mock_selection_result,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
            num_buffer_expansions=3,
            buffer_expansion_size=14,
        )

        # Verify FTW properties are set
        assert parent_item.properties["ftw:calendar_year"] == 2024
        assert parent_item.properties["ftw:planting_day"] == 150
        assert parent_item.properties["ftw:harvest_day"] == 250
        assert parent_item.properties["ftw:stac_host"] == "earthsearch"
        assert parent_item.properties["ftw:cloud_cover_chip_threshold"] == 2.0
        assert parent_item.properties["ftw:buffer_days"] == 14
        assert parent_item.properties["ftw:num_buffer_expansions"] == 3
        assert parent_item.properties["ftw:buffer_expansion_size"] == 14
        assert parent_item.properties["ftw:planting_buffer_used"] == 14
        assert parent_item.properties["ftw:harvest_buffer_used"] == 14
        assert parent_item.properties["ftw:expansions_performed"] == 0

    def test_stac_host_reflects_selection_backend(
        self,
        tmp_path: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """ftw:stac_host records the backend the selection actually used."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()
        mock_selection_result.selection_params = {"stac_host": "parquet-mirror"}

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        create_child_items_from_selection(
            chip_dir=chip_dir,
            parent_item=parent_item,
            result=mock_selection_result,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
            num_buffer_expansions=3,
            buffer_expansion_size=14,
        )

        assert parent_item.properties["ftw:stac_host"] == "parquet-mirror"

    def test_sets_temporal_extent_from_scene_dates(
        self,
        tmp_path: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """Test that temporal extent is set from actual scene acquisition dates.

        Regression test: Previously this was incorrectly set to the full calendar year
        (Jan 1 - Dec 31). It should reflect the actual planting and harvest dates.
        """
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        create_child_items_from_selection(
            chip_dir=chip_dir,
            parent_item=parent_item,
            result=mock_selection_result,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )

        # Mock scenes have planting=June 15 and harvest=Sept 15
        # start_datetime should be the planting date, end_datetime should be harvest
        assert parent_item.properties["start_datetime"] == "2024-06-15T10:00:00+00:00"
        assert parent_item.properties["end_datetime"] == "2024-09-15T10:00:00+00:00"

    def test_temporal_extent_not_full_year_regression(
        self,
        tmp_path: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """Regression test: temporal extent must NOT be the full calendar year.

        This test explicitly verifies that a previous bug (setting datetime to full year)
        does not resurface. The datetime range should come from actual scene dates.
        """
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        create_child_items_from_selection(
            chip_dir=chip_dir,
            parent_item=parent_item,
            result=mock_selection_result,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )

        # Explicitly verify these are NOT the old buggy values (full year range)
        assert parent_item.properties["start_datetime"] != "2024-01-01T00:00:00+00:00"
        assert parent_item.properties["end_datetime"] != "2024-12-31T23:59:59+00:00"

    def test_temporal_extent_with_planting_only(
        self,
        tmp_path: Path,
        mock_selection_result_planting_only: SceneSelectionResult,
    ) -> None:
        """Test temporal extent when only planting scene is present."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        create_child_items_from_selection(
            chip_dir=chip_dir,
            parent_item=parent_item,
            result=mock_selection_result_planting_only,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )

        # With only planting scene (June 15), both start and end should be same date
        assert parent_item.properties["start_datetime"] == "2024-06-15T10:00:00+00:00"
        assert parent_item.properties["end_datetime"] == "2024-06-15T10:00:00+00:00"

    def test_temporal_extent_with_harvest_only(
        self,
        tmp_path: Path,
        mock_selection_result_harvest_only: SceneSelectionResult,
    ) -> None:
        """Test temporal extent when only harvest scene is present."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        create_child_items_from_selection(
            chip_dir=chip_dir,
            parent_item=parent_item,
            result=mock_selection_result_harvest_only,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )

        # With only harvest scene (Sept 15), both start and end should be same date
        assert parent_item.properties["start_datetime"] == "2024-09-15T10:00:00+00:00"
        assert parent_item.properties["end_datetime"] == "2024-09-15T10:00:00+00:00"

    def test_sets_cloud_cover_from_scenes(
        self,
        tmp_path: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """Test that cloud cover from child scenes is added to parent."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        create_child_items_from_selection(
            chip_dir=chip_dir,
            parent_item=parent_item,
            result=mock_selection_result,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )

        assert parent_item.properties["ftw:planting_cloud_cover"] == 1.5
        assert parent_item.properties["ftw:harvest_cloud_cover"] == 2.0

    def test_removes_existing_links(
        self,
        tmp_path: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """Test that existing planting/harvest/derived links are removed."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        # Add existing links that should be removed
        parent_item.add_link(pystac.Link(rel="ftw:planting", target="./old_planting.json"))
        parent_item.add_link(pystac.Link(rel="ftw:harvest", target="./old_harvest.json"))
        parent_item.add_link(pystac.Link(rel="derived", target="./derived.json"))
        # Use an absolute URL for 'self' to avoid resolution issues with relative links
        parent_item.add_link(
            pystac.Link(rel="self", target="https://example.com/catalog/chip_001.json")
        )

        create_child_items_from_selection(
            chip_dir=chip_dir,
            parent_item=parent_item,
            result=mock_selection_result,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )

        # Verify old links are removed and new ones added
        link_rels = [link.rel for link in parent_item.links]
        assert link_rels.count("ftw:planting") == 1
        assert link_rels.count("ftw:harvest") == 1
        assert "derived" not in link_rels
        assert "self" in link_rels  # Should be preserved

    def test_removes_stale_imagery_assets(
        self,
        tmp_path: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """Test that imagery assets from a replaced selection are dropped.

        A --force-image-selection re-run picks a different scene, so the
        planting/harvest GeoTIFFs and the thumbnail the parent advertises no
        longer describe the selection. Mask assets are untouched.
        """
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )
        parent_item.add_asset(
            "planting_image", pystac.Asset(href="./chip_001_planting_image_s2.tif")
        )
        parent_item.add_asset("harvest_image", pystac.Asset(href="./chip_001_harvest_image_s2.tif"))
        parent_item.add_asset("thumbnail", pystac.Asset(href="./chip_001_thumbnail.jpg"))
        parent_item.add_asset("instance_mask", pystac.Asset(href="./chip_001_instance.tif"))

        create_child_items_from_selection(
            chip_dir=chip_dir,
            parent_item=parent_item,
            result=mock_selection_result,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )

        assert "planting_image" not in parent_item.assets
        assert "harvest_image" not in parent_item.assets
        assert "thumbnail" not in parent_item.assets
        assert "instance_mask" in parent_item.assets

    def test_adds_planting_link(
        self,
        tmp_path: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """Test that planting link is added to parent item."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        create_child_items_from_selection(
            chip_dir=chip_dir,
            parent_item=parent_item,
            result=mock_selection_result,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )

        planting_links = [link for link in parent_item.links if link.rel == "ftw:planting"]
        assert len(planting_links) == 1
        assert planting_links[0].target == "./chip_001_planting_s2.json"
        assert planting_links[0].media_type == "application/json"
        assert planting_links[0].title == "Planting season Sentinel-2 imagery"

    def test_adds_harvest_link(
        self,
        tmp_path: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """Test that harvest link is added to parent item."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        create_child_items_from_selection(
            chip_dir=chip_dir,
            parent_item=parent_item,
            result=mock_selection_result,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )

        harvest_links = [link for link in parent_item.links if link.rel == "ftw:harvest"]
        assert len(harvest_links) == 1
        assert harvest_links[0].target == "./chip_001_harvest_s2.json"
        assert harvest_links[0].media_type == "application/json"
        assert harvest_links[0].title == "Harvest season Sentinel-2 imagery"

    def test_saves_parent_item(
        self,
        tmp_path: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """Test that parent item is saved to disk."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        create_child_items_from_selection(
            chip_dir=chip_dir,
            parent_item=parent_item,
            result=mock_selection_result,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )

        parent_path = chip_dir / "chip_001.json"
        assert parent_path.exists()

        # Verify saved content
        saved_item = pystac.Item.from_file(str(parent_path))
        assert saved_item.properties["ftw:calendar_year"] == 2024

    def test_creates_planting_child(
        self,
        tmp_path: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """Test that planting child item is created."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        create_child_items_from_selection(
            chip_dir=chip_dir,
            parent_item=parent_item,
            result=mock_selection_result,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )

        planting_path = chip_dir / "chip_001_planting_s2.json"
        assert planting_path.exists()

        child_item = pystac.Item.from_file(str(planting_path))
        assert child_item.id == "chip_001_planting_s2"
        assert child_item.properties["ftw:season"] == "planting"

    def test_creates_harvest_child(
        self,
        tmp_path: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """Test that harvest child item is created."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        create_child_items_from_selection(
            chip_dir=chip_dir,
            parent_item=parent_item,
            result=mock_selection_result,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )

        harvest_path = chip_dir / "chip_001_harvest_s2.json"
        assert harvest_path.exists()

        child_item = pystac.Item.from_file(str(harvest_path))
        assert child_item.id == "chip_001_harvest_s2"
        assert child_item.properties["ftw:season"] == "harvest"

    def test_handles_missing_planting(
        self,
        tmp_path: Path,
        mock_selection_result_harvest_only: SceneSelectionResult,
    ) -> None:
        """Test handling when planting scene is missing."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        create_child_items_from_selection(
            chip_dir=chip_dir,
            parent_item=parent_item,
            result=mock_selection_result_harvest_only,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )

        # Planting child should not be created
        planting_path = chip_dir / "chip_001_planting_s2.json"
        assert not planting_path.exists()

        # Harvest child should be created
        harvest_path = chip_dir / "chip_001_harvest_s2.json"
        assert harvest_path.exists()

        # Parent should not have planting link
        planting_links = [link for link in parent_item.links if link.rel == "ftw:planting"]
        assert len(planting_links) == 0

        # Parent should not have planting cloud cover property
        assert "ftw:planting_cloud_cover" not in parent_item.properties

    def test_handles_missing_harvest(
        self,
        tmp_path: Path,
        mock_selection_result_planting_only: SceneSelectionResult,
    ) -> None:
        """Test handling when harvest scene is missing."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        create_child_items_from_selection(
            chip_dir=chip_dir,
            parent_item=parent_item,
            result=mock_selection_result_planting_only,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )

        # Planting child should be created
        planting_path = chip_dir / "chip_001_planting_s2.json"
        assert planting_path.exists()

        # Harvest child should not be created
        harvest_path = chip_dir / "chip_001_harvest_s2.json"
        assert not harvest_path.exists()

        # Parent should not have harvest link
        harvest_links = [link for link in parent_item.links if link.rel == "ftw:harvest"]
        assert len(harvest_links) == 0

        # Parent should not have harvest cloud cover property
        assert "ftw:harvest_cloud_cover" not in parent_item.properties

    def test_sets_self_href_when_missing(
        self,
        tmp_path: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """Test that self_href is set on parent item when missing."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        # Ensure no self_href
        assert parent_item.get_self_href() is None

        create_child_items_from_selection(
            chip_dir=chip_dir,
            parent_item=parent_item,
            result=mock_selection_result,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )

        assert parent_item.get_self_href() is not None
        assert parent_item.get_self_href() == str(chip_dir / "chip_001.json")


class TestCreateSeasonChildItem:
    """Tests for _create_season_child_item function."""

    def test_creates_child_with_correct_id(
        self,
        tmp_path: Path,
        mock_selected_scene: MagicMock,
    ) -> None:
        """Test that child item has correct ID format."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )
        parent_item.set_self_href(str(chip_dir / "chip_001.json"))

        _create_season_child_item(
            chip_dir=chip_dir,
            parent_item=parent_item,
            scene=mock_selected_scene,
            season="planting",
            year=2024,
        )

        child_path = chip_dir / "chip_001_planting_s2.json"
        assert child_path.exists()

        child_item = pystac.Item.from_file(str(child_path))
        assert child_item.id == "chip_001_planting_s2"

    def test_sets_child_properties(
        self,
        tmp_path: Path,
        mock_selected_scene: MagicMock,
    ) -> None:
        """Test that child item has correct properties."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )
        parent_item.set_self_href(str(chip_dir / "chip_001.json"))

        _create_season_child_item(
            chip_dir=chip_dir,
            parent_item=parent_item,
            scene=mock_selected_scene,
            season="planting",
            year=2024,
        )

        child_path = chip_dir / "chip_001_planting_s2.json"
        child_item = pystac.Item.from_file(str(child_path))

        assert child_item.properties["ftw:season"] == "planting"
        assert child_item.properties["ftw:source"] == "sentinel-2"
        assert child_item.properties["ftw:calendar_year"] == 2024

    def test_copies_band_assets(
        self,
        tmp_path: Path,
        mock_selected_scene: MagicMock,
    ) -> None:
        """Test that band assets are copied from source scene."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )
        parent_item.set_self_href(str(chip_dir / "chip_001.json"))

        _create_season_child_item(
            chip_dir=chip_dir,
            parent_item=parent_item,
            scene=mock_selected_scene,
            season="planting",
            year=2024,
        )

        child_path = chip_dir / "chip_001_planting_s2.json"
        child_item = pystac.Item.from_file(str(child_path))

        # Check that band assets are copied
        expected_bands = ["red", "green", "blue", "nir", "scl", "visual"]
        for band in expected_bands:
            assert band in child_item.assets
            assert "example.com" in child_item.assets[band].href

    def test_copies_cloud_probability(
        self,
        tmp_path: Path,
        mock_selected_scene: MagicMock,
    ) -> None:
        """Test that cloud probability asset is copied."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )
        parent_item.set_self_href(str(chip_dir / "chip_001.json"))

        _create_season_child_item(
            chip_dir=chip_dir,
            parent_item=parent_item,
            scene=mock_selected_scene,
            season="planting",
            year=2024,
        )

        child_path = chip_dir / "chip_001_planting_s2.json"
        child_item = pystac.Item.from_file(str(child_path))

        # Check that cloud_probability asset is copied (from "cloud" in source)
        assert "cloud_probability" in child_item.assets

    def test_adds_parent_link(
        self,
        tmp_path: Path,
        mock_selected_scene: MagicMock,
    ) -> None:
        """Test that parent chip link is added."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )
        parent_item.set_self_href(str(chip_dir / "chip_001.json"))

        _create_season_child_item(
            chip_dir=chip_dir,
            parent_item=parent_item,
            scene=mock_selected_scene,
            season="planting",
            year=2024,
        )

        child_path = chip_dir / "chip_001_planting_s2.json"
        child_item = pystac.Item.from_file(str(child_path))

        parent_links = [link for link in child_item.links if link.rel == "ftw:parent_chip"]
        assert len(parent_links) == 1
        assert parent_links[0].target == "./chip_001.json"
        assert parent_links[0].media_type == "application/json"

    def test_adds_via_link(
        self,
        tmp_path: Path,
        mock_selected_scene: MagicMock,
    ) -> None:
        """Test that via link to source STAC is added."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )
        parent_item.set_self_href(str(chip_dir / "chip_001.json"))

        _create_season_child_item(
            chip_dir=chip_dir,
            parent_item=parent_item,
            scene=mock_selected_scene,
            season="planting",
            year=2024,
        )

        child_path = chip_dir / "chip_001_planting_s2.json"
        child_item = pystac.Item.from_file(str(child_path))

        via_links = [link for link in child_item.links if link.rel == "via"]
        assert len(via_links) == 1
        assert "example.com/stac" in via_links[0].target
        assert via_links[0].media_type == "application/json"

    def test_sets_cloud_cover_property(
        self,
        tmp_path: Path,
        mock_selected_scene: MagicMock,
    ) -> None:
        """Test that eo:cloud_cover property is set."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )
        parent_item.set_self_href(str(chip_dir / "chip_001.json"))

        _create_season_child_item(
            chip_dir=chip_dir,
            parent_item=parent_item,
            scene=mock_selected_scene,
            season="planting",
            year=2024,
        )

        child_path = chip_dir / "chip_001_planting_s2.json"
        child_item = pystac.Item.from_file(str(child_path))

        assert child_item.properties["eo:cloud_cover"] == 1.5

    def test_saves_child_item(
        self,
        tmp_path: Path,
        mock_selected_scene: MagicMock,
    ) -> None:
        """Test that child item is saved to disk."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )
        parent_item.set_self_href(str(chip_dir / "chip_001.json"))

        _create_season_child_item(
            chip_dir=chip_dir,
            parent_item=parent_item,
            scene=mock_selected_scene,
            season="planting",
            year=2024,
        )

        child_path = chip_dir / "chip_001_planting_s2.json"
        assert child_path.exists()

    def test_child_item_carries_portolan_schema_uri_exactly_once(
        self,
        tmp_path: Path,
        mock_selected_scene: MagicMock,
    ) -> None:
        """Test that the child item declares the Portolan schema URI exactly once."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )
        parent_item.set_self_href(str(chip_dir / "chip_001.json"))

        _create_season_child_item(
            chip_dir=chip_dir,
            parent_item=parent_item,
            scene=mock_selected_scene,
            season="planting",
            year=2024,
        )

        child_path = chip_dir / "chip_001_planting_s2.json"
        child_item = pystac.Item.from_file(str(child_path))

        assert child_item.stac_extensions.count(PORTOLAN_SCHEMA_URI) == 1

    def test_creates_harvest_season(
        self,
        tmp_path: Path,
        mock_bbox: tuple[float, float, float, float],
    ) -> None:
        """Test creation of harvest season child item."""
        from ftw_dataset_tools.api.imagery.scene_selection import SelectedScene

        from .conftest import create_mock_s2_assets, create_mock_stac_item

        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=mock_bbox,
            datetime=datetime.now(UTC),
            properties={},
        )
        parent_item.set_self_href(str(chip_dir / "chip_001.json"))

        harvest_item = create_mock_stac_item(
            item_id="S2A_TILE_20240915_test",
            bbox=mock_bbox,
            properties={"eo:cloud_cover": 2.0},
            assets=create_mock_s2_assets(),
        )

        harvest_scene = SelectedScene(
            item=harvest_item,
            season="harvest",
            cloud_cover=2.0,
            datetime=datetime(2024, 9, 15, 10, 0, 0, tzinfo=UTC),
            stac_url="https://example.com/stac/S2A_TILE_20240915_test",
        )

        _create_season_child_item(
            chip_dir=chip_dir,
            parent_item=parent_item,
            scene=harvest_scene,
            season="harvest",
            year=2024,
        )

        child_path = chip_dir / "chip_001_harvest_s2.json"
        assert child_path.exists()

        child_item = pystac.Item.from_file(str(child_path))
        assert child_item.id == "chip_001_harvest_s2"
        assert child_item.properties["ftw:season"] == "harvest"

    def test_handles_scene_without_stac_url(
        self,
        tmp_path: Path,
        mock_bbox: tuple[float, float, float, float],
    ) -> None:
        """Test handling when scene has empty stac_url."""
        from ftw_dataset_tools.api.imagery.scene_selection import SelectedScene

        from .conftest import create_mock_s2_assets, create_mock_stac_item

        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=mock_bbox,
            datetime=datetime.now(UTC),
            properties={},
        )
        parent_item.set_self_href(str(chip_dir / "chip_001.json"))

        scene_item = create_mock_stac_item(
            item_id="S2A_TILE_20240615_test",
            bbox=mock_bbox,
            properties={"eo:cloud_cover": 1.5},
            assets=create_mock_s2_assets(),
        )

        scene = SelectedScene(
            item=scene_item,
            season="planting",
            cloud_cover=1.5,
            datetime=datetime(2024, 6, 15, 10, 0, 0, tzinfo=UTC),
            stac_url="",  # Empty URL
        )

        _create_season_child_item(
            chip_dir=chip_dir,
            parent_item=parent_item,
            scene=scene,
            season="planting",
            year=2024,
        )

        child_path = chip_dir / "chip_001_planting_s2.json"
        child_item = pystac.Item.from_file(str(child_path))

        # Via link should not be added for empty URL
        via_links = [link for link in child_item.links if link.rel == "via"]
        assert len(via_links) == 0

    def test_copies_geometry_from_parent(
        self,
        tmp_path: Path,
        mock_selected_scene: MagicMock,
    ) -> None:
        """Test that child item copies geometry from parent."""
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        parent_geometry = {
            "type": "Polygon",
            "coordinates": [[[10.0, 50.0], [10.01, 50.0], [10.01, 50.01], [10.0, 50.0]]],
        }
        parent_bbox = (10.0, 50.0, 10.01, 50.01)

        parent_item = pystac.Item(
            id="chip_001",
            geometry=parent_geometry,
            bbox=parent_bbox,
            datetime=datetime.now(UTC),
            properties={},
        )
        parent_item.set_self_href(str(chip_dir / "chip_001.json"))

        _create_season_child_item(
            chip_dir=chip_dir,
            parent_item=parent_item,
            scene=mock_selected_scene,
            season="planting",
            year=2024,
        )

        child_path = chip_dir / "chip_001_planting_s2.json"
        child_item = pystac.Item.from_file(str(child_path))

        assert child_item.geometry == parent_geometry
        assert child_item.bbox == list(parent_bbox)


HIERARCHICAL_HREFS = {
    "root": "../../../../catalog.json",
    "parent": "../catalog.json",
    "collection": "../../../collection.json",
}


def _staged_parent(tmp_path: Path) -> tuple[pystac.Item, Path]:
    """A chip item on disk whose hierarchical links all point at missing files.

    This is the shape a staging tree has once its items carry the *published*
    root href: nothing the links name exists next to the item.
    """
    import json

    chip_dir = tmp_path / "chips" / "31UFR" / "chip_001"
    chip_dir.mkdir(parents=True)
    path = chip_dir / "chip_001.json"

    parent = pystac.Item(
        id="chip_001",
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        bbox=(0.0, 0.0, 1.0, 1.0),
        datetime=datetime.now(UTC),
        properties={},
        collection="lu",
    )
    for rel, href in HIERARCHICAL_HREFS.items():
        parent.add_link(pystac.Link(rel=rel, target=href, media_type="application/json"))
    path.write_text(
        json.dumps(parent.to_dict(include_self_link=False, transform_hrefs=False), indent=2)
    )

    return pystac.Item.from_file(str(path)), path


def _hrefs(path: Path) -> dict[str, str]:
    import json

    return {link["rel"]: link["href"] for link in json.loads(path.read_text())["links"]}


class TestWritesWithoutResolvingTheRoot:
    """Regression: an unresolvable `rel: root` link must not fail every chip."""

    def test_succeeds_when_the_root_link_target_is_missing(
        self,
        tmp_path: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        parent_item, parent_path = _staged_parent(tmp_path)

        create_child_items_from_selection(
            chip_dir=parent_path.parent,
            parent_item=parent_item,
            result=mock_selection_result,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
            num_buffer_expansions=3,
            buffer_expansion_size=14,
        )

        assert (parent_path.parent / "chip_001_planting_s2.json").exists()
        assert (parent_path.parent / "chip_001_harvest_s2.json").exists()

    def test_parent_keeps_its_relative_hierarchical_links(
        self,
        tmp_path: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        parent_item, parent_path = _staged_parent(tmp_path)

        create_child_items_from_selection(
            chip_dir=parent_path.parent,
            parent_item=parent_item,
            result=mock_selection_result,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
            num_buffer_expansions=3,
            buffer_expansion_size=14,
        )

        written = _hrefs(parent_path)
        for rel, href in HIERARCHICAL_HREFS.items():
            assert written[rel] == href
        assert written["ftw:planting"] == "./chip_001_planting_s2.json"

    def test_children_inherit_the_parents_hierarchical_links(
        self,
        tmp_path: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        parent_item, parent_path = _staged_parent(tmp_path)

        create_child_items_from_selection(
            chip_dir=parent_path.parent,
            parent_item=parent_item,
            result=mock_selection_result,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
            num_buffer_expansions=3,
            buffer_expansion_size=14,
        )

        for season in ("planting", "harvest"):
            child = _hrefs(parent_path.parent / f"chip_001_{season}_s2.json")
            for rel, href in HIERARCHICAL_HREFS.items():
                assert child[rel] == href, f"{season} child link {rel}"
            assert child["ftw:parent_chip"] == "./chip_001.json"


MEDIA_TYPE_COG = "image/tiff; application=geotiff; profile=cloud-optimized"


class TestVisualSeasonAssets:
    """The parent chip item carries the true-colour scene COG for each season."""

    def _parent(self) -> pystac.Item:
        return pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

    def _run(self, chip_dir: Path, parent_item: pystac.Item, result) -> None:
        chip_dir.mkdir(exist_ok=True)
        create_child_items_from_selection(
            chip_dir=chip_dir,
            parent_item=parent_item,
            result=result,
            year=2024,
            cloud_cover_chip=2.0,
            buffer_days=14,
        )

    def test_both_seasons_get_a_visual_asset(
        self, tmp_path: Path, mock_selection_result: SceneSelectionResult
    ) -> None:
        parent_item = self._parent()

        self._run(tmp_path / "chip_001", parent_item, mock_selection_result)

        for season in ("planting", "harvest"):
            asset = parent_item.assets[f"{season}_visual"]
            assert asset.href == "https://example.com/data/visual.tif"
            assert asset.media_type == MEDIA_TYPE_COG
            assert asset.roles == ["visual"]
            assert asset.title == f"{season.capitalize()} season scene (true colour)"

    def test_visual_asset_records_scene_and_datetime(
        self, tmp_path: Path, mock_selection_result: SceneSelectionResult
    ) -> None:
        parent_item = self._parent()

        self._run(tmp_path / "chip_001", parent_item, mock_selection_result)

        planting = parent_item.assets["planting_visual"].extra_fields
        assert planting["ftw:scene"] == "S2A_TILE_20240615_test"
        assert planting["datetime"].startswith("2024-06-15T10:00:00")
        harvest = parent_item.assets["harvest_visual"].extra_fields
        assert harvest["ftw:scene"] == "S2A_TILE_20240915_test"

    def test_visual_assets_are_written_to_disk(
        self, tmp_path: Path, mock_selection_result: SceneSelectionResult
    ) -> None:
        import json

        chip_dir = tmp_path / "chip_001"
        parent_item = self._parent()

        self._run(chip_dir, parent_item, mock_selection_result)

        saved = json.loads((chip_dir / "chip_001.json").read_text())
        assert saved["assets"]["planting_visual"]["href"] == "https://example.com/data/visual.tif"
        assert saved["assets"]["harvest_visual"]["roles"] == ["visual"]

    def test_missing_season_gets_no_visual_asset(
        self, tmp_path: Path, mock_selection_result_planting_only: SceneSelectionResult
    ) -> None:
        parent_item = self._parent()

        self._run(tmp_path / "chip_001", parent_item, mock_selection_result_planting_only)

        assert "planting_visual" in parent_item.assets
        assert "harvest_visual" not in parent_item.assets

    def test_rerun_replaces_rather_than_duplicates(
        self,
        tmp_path: Path,
        mock_selection_result: SceneSelectionResult,
        mock_selection_result_planting_only: SceneSelectionResult,
    ) -> None:
        chip_dir = tmp_path / "chip_001"
        parent_item = self._parent()

        self._run(chip_dir, parent_item, mock_selection_result)
        self._run(chip_dir, parent_item, mock_selection_result_planting_only)

        assert "harvest_visual" not in parent_item.assets
        assert parent_item.assets["planting_visual"].href == "https://example.com/data/visual.tif"
        rels = [link.rel for link in parent_item.links]
        assert rels.count("ftw:planting") == 1

    def test_reselection_drops_the_replaced_scenes_local_image(
        self, tmp_path: Path, mock_selection_result: SceneSelectionResult
    ) -> None:
        """A new selection stops advertising the previous scene's clipped image.

        The GeoTIFF on disk was clipped from the scene that has just been
        replaced, so the parent must not keep describing it as this chip's
        imagery. The file itself is left alone: the next download overwrites it
        and ``attach_season_to_parent`` re-derives the asset from it.
        """
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()
        image_path = chip_dir / "chip_001_planting_image_s2.tif"
        image_path.write_bytes(b"not really a geotiff")

        parent_item = self._parent()
        parent_item.add_asset(
            "planting_image",
            pystac.Asset(href="./chip_001_planting_image_s2.tif", roles=["data"]),
        )

        self._run(chip_dir, parent_item, mock_selection_result)

        assert "planting_image" not in parent_item.assets
        assert image_path.exists()


class TestAttachSeasonToParent:
    """The shared helper used by selection and by a STAC rebuild."""

    def test_reattaches_link_visual_and_local_image(self, tmp_path: Path) -> None:
        import json

        from ftw_dataset_tools.api.imagery.stac_child_items import attach_existing_seasons

        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()
        _write_child(chip_dir, "chip_001", "planting")

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        attached = attach_existing_seasons(parent_item, chip_dir)

        assert attached == ["planting"]
        assert parent_item.get_single_link("ftw:planting").href == "./chip_001_planting_s2.json"
        assert parent_item.assets["planting_visual"].href.startswith("https://")
        assert "harvest_visual" not in parent_item.assets
        assert json.loads(json.dumps(parent_item.to_dict(include_self_link=False)))

    def test_no_children_on_disk_is_a_no_op(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.imagery.stac_child_items import attach_existing_seasons

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        assert attach_existing_seasons(parent_item, tmp_path) == []
        assert parent_item.assets == {}


def _write_child(chip_dir: Path, chip_id: str, season: str, image: bool = False) -> None:
    """Write a minimal season child item next to its parent chip."""
    import json

    scene_id = "S2B_T31UFR_20250406T104557_L2A"
    child = {
        "type": "Feature",
        "stac_version": "1.1.0",
        "id": f"{chip_id}_{season}_s2",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "properties": {
            "ftw:season": season,
            "datetime": "2025-04-06T10:47:09.925000Z",
        },
        "links": [
            {
                "rel": "via",
                "href": f"https://earth-search.aws.element84.com/v1/collections/s2/items/{scene_id}",
                "type": "application/json",
            }
        ],
        "assets": {
            "visual": {
                "href": f"https://example.com/{scene_id}/TCI.tif",
                "type": MEDIA_TYPE_COG,
                "roles": ["visual"],
            }
        },
    }
    if image:
        child["assets"]["image"] = {
            "href": f"./{chip_id}_{season}_image_s2.tif",
            "type": MEDIA_TYPE_COG,
            "roles": ["data"],
        }
    (chip_dir / f"{chip_id}_{season}_s2.json").write_text(json.dumps(child))


class TestSeasonImageTitleFallback:
    """A GeoTIFF without band descriptions must not lose the band list in the title."""

    def test_falls_back_to_the_childs_own_image_title(self, tmp_path: Path) -> None:
        import json

        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        from ftw_dataset_tools.api.imagery.stac_child_items import attach_existing_seasons

        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()
        _write_child(chip_dir, "chip_001", "planting", image=True)

        # A one-band COG with no band descriptions at all.
        image_path = chip_dir / "chip_001_planting_image_s2.tif"
        data = np.array([[1, 2], [3, 4]], dtype="uint16")
        with rasterio.open(
            image_path,
            "w",
            driver="COG",
            width=2,
            height=2,
            count=1,
            dtype="uint16",
            crs="EPSG:4326",
            transform=from_bounds(0, 0, 1, 1, 2, 2),
        ) as dst:
            dst.write(data, 1)

        child = json.loads((chip_dir / "chip_001_planting_s2.json").read_text())
        child["assets"]["image"]["title"] = "Clipped 4-band image (red,green,blue,nir)"
        (chip_dir / "chip_001_planting_s2.json").write_text(json.dumps(child))

        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        attach_existing_seasons(parent_item, chip_dir)

        assert parent_item.assets["planting_image"].title == (
            "Planting season imagery (red,green,blue,nir)"
        )


class TestParentSurvivesAChildFailure:
    """A raise inside the season loop must not lose the selection properties."""

    def test_parent_is_written_even_when_a_child_fails(
        self, tmp_path: Path, mock_selection_result: SceneSelectionResult
    ) -> None:
        import json
        from unittest.mock import patch

        import pytest

        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()
        parent_item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        with (
            patch(
                "ftw_dataset_tools.api.imagery.stac_child_items._create_season_child_item",
                side_effect=OSError("disk full"),
            ),
            pytest.raises(OSError, match="disk full"),
        ):
            create_child_items_from_selection(
                chip_dir=chip_dir,
                parent_item=parent_item,
                result=mock_selection_result,
                year=2024,
                cloud_cover_chip=2.0,
                buffer_days=14,
            )

        saved = json.loads((chip_dir / "chip_001.json").read_text())
        assert saved["properties"]["ftw:calendar_year"] == 2024
        assert saved["properties"]["ftw:planting_day"] == 150


_QUARTER_NODATA = {"q1": 0.0, "q2": 2.2874, "q3": 13.0, "q4": 100.0}
_CHIP_BBOX = (10.0, 50.0, 10.01, 50.01)


def _mosaic_selection(nodata: dict[str, float]) -> MosaicSelectionResult:
    def tile(number: int) -> pystac.Item:
        item = pystac.Item(
            id=f"Sentinel-2_mosaic_2024_Q{number}_32UNA_0_0",
            geometry=None,
            bbox=None,
            datetime=None,
            start_datetime=datetime(2024, 3 * number - 2, 1, tzinfo=UTC),
            end_datetime=datetime(2024, 3 * number, 28, tzinfo=UTC),
            properties={"ftw:mosaic_tile": "32UNA_0_0"},
        )
        item.add_asset("red", pystac.Asset(href=f"https://example.com/Q{number}/red.tif"))
        return item

    quarters = {
        slot: SelectedMosaic(item=tile(int(slot[1])), quarter=slot, year=2024, nodata_pct=pct)
        for slot, pct in nodata.items()
    }
    return MosaicSelectionResult(
        chip_id="chip_001",
        bbox=_CHIP_BBOX,
        requested_year=2024,
        imagery_year=2024,
        quarters=quarters,
    )


def _bare_chip(chip_id: str = "chip_001") -> pystac.Item:
    return pystac.Item(
        id=chip_id,
        geometry=None,
        bbox=_CHIP_BBOX,
        datetime=datetime(2024, 1, 1, tzinfo=UTC),
        properties={},
    )


def _mosaic_chip(tmp_path: Path) -> Path:
    """A chip item with all four quarters selected; returns the chip item's path."""
    chip_dir = tmp_path / "chips" / "32UNA" / "chip_001"
    chip_dir.mkdir(parents=True)
    path = chip_dir / "chip_001.json"
    parent = _bare_chip()
    parent.set_self_href(str(path))
    create_mosaic_child_items(chip_dir, parent, _mosaic_selection(_QUARTER_NODATA))
    return path


def _strip_quarter_nodata(path: Path) -> None:
    """Make the chip item look like one written before it carried the quarters' nodata."""
    data = json.loads(path.read_text())
    for key in QUARTER_NODATA_PROPERTIES:
        data["properties"].pop(key)
    path.write_text(json.dumps(data))


class TestQuarterNodataOnChip:
    """Each quarter child's ftw:nodata_pct is mirrored onto the chip item."""

    def test_selection_records_every_quarter(self, tmp_path: Path) -> None:
        props = pystac.Item.from_file(str(_mosaic_chip(tmp_path))).properties

        assert {key: props.get(key) for key in QUARTER_NODATA_PROPERTIES} == {
            "ftw:q1_nodata_pct": 0.0,
            "ftw:q2_nodata_pct": 2.29,
            "ftw:q3_nodata_pct": 13.0,
            "ftw:q4_nodata_pct": 100.0,
        }

    def test_matches_the_childs_nodata(self, tmp_path: Path) -> None:
        path = _mosaic_chip(tmp_path)
        child = pystac.Item.from_file(str(path.parent / "chip_001_q2_s2.json"))

        parent = pystac.Item.from_file(str(path))
        assert parent.properties.get("ftw:q2_nodata_pct") == child.properties["ftw:nodata_pct"]

    def test_scene_children_record_no_nodata(
        self, tmp_path: Path, mock_selection_result: SceneSelectionResult
    ) -> None:
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()
        parent = _bare_chip()
        create_child_items_from_selection(chip_dir, parent, mock_selection_result, 2024, 2.0, 14)

        assert not any(key.endswith("_nodata_pct") for key in parent.properties)

    def test_rebuild_reads_them_from_the_children_on_disk(self, tmp_path: Path) -> None:
        path = _mosaic_chip(tmp_path)
        fresh = _bare_chip()
        attach_existing_seasons(fresh, path.parent)

        assert fresh.properties.get("ftw:q3_nodata_pct") == 13.0
        assert all(key in fresh.properties for key in QUARTER_NODATA_PROPERTIES)

    def test_a_missing_quarter_leaves_its_property_absent(self, tmp_path: Path) -> None:
        path = _mosaic_chip(tmp_path)
        (path.parent / "chip_001_q2_s2.json").unlink()
        fresh = _bare_chip()
        attach_existing_seasons(fresh, path.parent)
        # The previous chip item still holds the deleted quarter's value.
        preserve_imagery_selection(fresh, path)

        assert "ftw:q2_nodata_pct" not in fresh.properties
        assert fresh.properties.get("ftw:q1_nodata_pct") == 0.0

    def test_a_chip_without_mosaics_has_none(self, tmp_path: Path) -> None:
        path = _mosaic_chip(tmp_path)
        clear_chip_selections(pystac.Item.from_file(str(path)), modes=["mosaics"])

        props = pystac.Item.from_file(str(path)).properties
        assert not any(key in props for key in QUARTER_NODATA_PROPERTIES)


class TestRecordQuarterNodata:
    """Backfill: chip items written before they carried the quarters' nodata."""

    def test_fills_chip_items_from_the_children_on_disk(self, tmp_path: Path) -> None:
        path = _mosaic_chip(tmp_path)
        _strip_quarter_nodata(path)

        assert record_quarter_nodata(find_s2_child_items(tmp_path)) == 1

        props = pystac.Item.from_file(str(path)).properties
        assert props.get("ftw:q2_nodata_pct") == 2.29
        assert all(key in props for key in QUARTER_NODATA_PROPERTIES)

    def test_leaves_up_to_date_chip_items_alone(self, tmp_path: Path) -> None:
        path = _mosaic_chip(tmp_path)
        before = path.read_bytes()

        assert record_quarter_nodata(find_s2_child_items(tmp_path)) == 0
        assert path.read_bytes() == before

    def test_a_child_without_nodata_records_none(self, tmp_path: Path) -> None:
        path = _mosaic_chip(tmp_path)
        _strip_quarter_nodata(path)
        q1 = path.parent / "chip_001_q1_s2.json"
        data = json.loads(q1.read_text())
        del data["properties"]["ftw:nodata_pct"]
        q1.write_text(json.dumps(data))

        record_quarter_nodata(find_s2_child_items(tmp_path))

        props = pystac.Item.from_file(str(path)).properties
        assert "ftw:q1_nodata_pct" not in props
        assert props.get("ftw:q4_nodata_pct") == 100.0

    def test_an_unlinked_leftover_child_records_nothing(self, tmp_path: Path) -> None:
        path = _mosaic_chip(tmp_path)
        data = json.loads(path.read_text())
        data["links"] = [link for link in data["links"] if link["rel"] != "ftw:q3"]
        path.write_text(json.dumps(data))

        record_quarter_nodata(find_s2_child_items(tmp_path))

        props = pystac.Item.from_file(str(path)).properties
        assert "ftw:q3_nodata_pct" not in props
        assert props.get("ftw:q2_nodata_pct") == 2.29

    def test_a_linked_quarter_with_an_unreadable_child_loses_its_value(
        self, tmp_path: Path
    ) -> None:
        path = _mosaic_chip(tmp_path)
        (path.parent / "chip_001_q4_s2.json").write_text("{not json")

        assert record_quarter_nodata(find_s2_child_items(tmp_path)) == 1
        assert "ftw:q4_nodata_pct" not in pystac.Item.from_file(str(path)).properties

    def test_an_unreadable_chip_item_is_skipped(self, tmp_path: Path) -> None:
        path = _mosaic_chip(tmp_path)
        path.write_text("{not json")

        assert record_quarter_nodata(find_s2_child_items(tmp_path)) == 0
        assert path.read_text() == "{not json"
