"""Tests for stac_child_items module."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pystac

from ftw_dataset_tools.api.imagery.stac_child_items import (
    _create_season_child_item,
    create_child_items_from_selection,
)

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
