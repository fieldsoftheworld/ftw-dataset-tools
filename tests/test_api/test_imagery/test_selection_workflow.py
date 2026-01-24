"""Tests for selection_workflow module."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pystac
import pytest

from ftw_dataset_tools.api.imagery.catalog_ops import has_existing_scenes
from ftw_dataset_tools.api.imagery.selection_workflow import (
    SelectionWorkflowResult,
    find_chip_items,
    select_imagery_for_catalog,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult


class TestFindChipItems:
    """Tests for find_chip_items function."""

    def test_finds_parent_items(self, mock_catalog_with_chips: Path) -> None:
        """Test that parent chip items are found."""
        items = find_chip_items(mock_catalog_with_chips)

        assert len(items) == 2
        item_ids = {item.id for item, _ in items}
        assert item_ids == {"chip_001", "chip_002"}

    def test_skips_child_items(self, mock_catalog_with_s2_children: Path) -> None:
        """Test that S2 child items are excluded."""
        items = find_chip_items(mock_catalog_with_s2_children)

        # Should only find parent items, not child items
        assert len(items) == 2
        item_ids = {item.id for item, _ in items}
        assert item_ids == {"chip_001", "chip_002"}

        # Verify no child items in results
        for item, _ in items:
            assert "_planting_s2" not in item.id
            assert "_harvest_s2" not in item.id

    def test_skips_hidden_dirs(self, mock_catalog_with_hidden_dir: Path) -> None:
        """Test that hidden directories are skipped."""
        items = find_chip_items(mock_catalog_with_hidden_dir)

        assert len(items) == 1
        assert items[0][0].id == "chip_001"

    def test_skips_invalid_json(self, mock_catalog_with_invalid_json: Path) -> None:
        """Test that invalid JSON files are skipped."""
        items = find_chip_items(mock_catalog_with_invalid_json)

        # Should only find the valid item
        assert len(items) == 1
        assert items[0][0].id == "chip_001"

    def test_returns_empty_for_empty_dir(self, mock_catalog_empty: Path) -> None:
        """Test that empty list is returned for empty catalog."""
        items = find_chip_items(mock_catalog_empty)

        assert items == []

    def test_returns_item_paths(self, mock_catalog_with_chips: Path) -> None:
        """Test that correct paths are returned with items."""
        items = find_chip_items(mock_catalog_with_chips)

        for item, item_path in items:
            assert item_path.exists()
            assert item_path.name == f"{item.id}.json"
            assert item_path.parent.name == item.id


class TestHasExistingScenes:
    """Tests for has_existing_scenes function."""

    def test_returns_true_when_both_links(self) -> None:
        """Test returns True when both planting and harvest links exist."""
        item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )
        item.add_link(pystac.Link(rel="ftw:planting", target="./planting.json"))
        item.add_link(pystac.Link(rel="ftw:harvest", target="./harvest.json"))

        assert has_existing_scenes(item) is True

    def test_returns_false_when_planting_only(self) -> None:
        """Test returns False when only planting link exists."""
        item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )
        item.add_link(pystac.Link(rel="ftw:planting", target="./planting.json"))

        assert has_existing_scenes(item) is False

    def test_returns_false_when_harvest_only(self) -> None:
        """Test returns False when only harvest link exists."""
        item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )
        item.add_link(pystac.Link(rel="ftw:harvest", target="./harvest.json"))

        assert has_existing_scenes(item) is False

    def test_returns_false_when_no_links(self) -> None:
        """Test returns False when no scene links exist."""
        item = pystac.Item(
            id="chip_001",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            bbox=(0.0, 0.0, 1.0, 1.0),
            datetime=datetime.now(UTC),
            properties={},
        )

        assert has_existing_scenes(item) is False


class TestSelectImageryForCatalog:
    """Tests for select_imagery_for_catalog function."""

    def test_returns_empty_for_no_chips(self, mock_catalog_empty: Path) -> None:
        """Test returns empty result when no chips found."""
        result = select_imagery_for_catalog(
            catalog_dir=mock_catalog_empty,
            year=2024,
        )

        assert result.successful == 0
        assert result.skipped == 0
        assert result.failed == 0

    def test_skips_chips_without_bbox(self, tmp_path: Path) -> None:
        """Test that chips without bbox are skipped."""
        # Create a mock item with bbox=None that we can patch
        chip_dir = tmp_path / "chip_001"
        chip_dir.mkdir()

        # Write a minimal STAC item JSON without bbox
        import json

        item_json = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "chip_001",
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            "properties": {"datetime": "2024-01-01T00:00:00Z"},
            "links": [],
            "assets": {},
        }
        # Explicitly omit bbox from JSON
        item_path = chip_dir / "chip_001.json"
        item_path.write_text(json.dumps(item_json))

        result = select_imagery_for_catalog(
            catalog_dir=tmp_path,
            year=2024,
        )

        assert result.skipped == 1
        assert result.skipped_details[0]["reason"] == "No bbox in item"

    def test_skips_existing_scenes(self, mock_catalog_with_existing_scenes: Path) -> None:
        """Test that chips with existing scenes are skipped by default."""
        result = select_imagery_for_catalog(
            catalog_dir=mock_catalog_with_existing_scenes,
            year=2024,
        )

        assert result.skipped == 1
        assert "Already has imagery selections" in result.skipped_details[0]["reason"]

    @patch("ftw_dataset_tools.api.imagery.selection_workflow.select_scenes_for_chip")
    @patch("ftw_dataset_tools.api.imagery.selection_workflow.create_child_items_from_selection")
    @patch("ftw_dataset_tools.api.imagery.selection_workflow.ImageryProgressBar")
    def test_force_overrides_existing(
        self,
        mock_progress: MagicMock,
        mock_create_child: MagicMock,
        mock_select: MagicMock,
        mock_catalog_with_existing_scenes: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """Test that force=True processes chips with existing scenes."""
        mock_select.return_value = mock_selection_result
        mock_progress.return_value.__enter__ = MagicMock()
        mock_progress.return_value.__exit__ = MagicMock()

        result = select_imagery_for_catalog(
            catalog_dir=mock_catalog_with_existing_scenes,
            year=2024,
            force=True,
        )

        assert result.successful == 1
        assert mock_select.called
        assert mock_create_child.called

    @patch("ftw_dataset_tools.api.imagery.selection_workflow.select_scenes_for_chip")
    @patch("ftw_dataset_tools.api.imagery.selection_workflow.create_child_items_from_selection")
    @patch("ftw_dataset_tools.api.imagery.selection_workflow.ImageryProgressBar")
    def test_successful_selection(
        self,
        mock_progress: MagicMock,
        mock_create_child: MagicMock,
        mock_select: MagicMock,
        mock_catalog_with_chips: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """Test successful scene selection for chips."""
        mock_select.return_value = mock_selection_result
        mock_progress.return_value.__enter__ = MagicMock()
        mock_progress.return_value.__exit__ = MagicMock()

        result = select_imagery_for_catalog(
            catalog_dir=mock_catalog_with_chips,
            year=2024,
        )

        assert result.successful == 2
        assert result.skipped == 0
        assert result.failed == 0
        assert mock_create_child.call_count == 2

    @patch("ftw_dataset_tools.api.imagery.selection_workflow.select_scenes_for_chip")
    @patch("ftw_dataset_tools.api.imagery.selection_workflow.ImageryProgressBar")
    def test_handles_failed_selection(
        self,
        mock_progress: MagicMock,
        mock_select: MagicMock,
        mock_catalog_with_chips: Path,
        mock_crop_calendar: MagicMock,
    ) -> None:
        """Test handling of failed scene selection."""
        from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult

        # Return a failed selection result
        mock_select.return_value = SceneSelectionResult(
            chip_id="chip_001",
            bbox=(10.0, 50.0, 10.01, 50.01),
            year=2024,
            crop_calendar=mock_crop_calendar,
            planting_scene=None,
            harvest_scene=None,
            skipped_reason="No cloud-free scenes found",
            candidates_checked=5,
        )
        mock_progress.return_value.__enter__ = MagicMock()
        mock_progress.return_value.__exit__ = MagicMock()

        result = select_imagery_for_catalog(
            catalog_dir=mock_catalog_with_chips,
            year=2024,
            on_missing="skip",
        )

        assert result.successful == 0
        assert result.skipped == 2
        assert "No cloud-free scenes found" in result.skipped_details[0]["reason"]

    @patch("ftw_dataset_tools.api.imagery.selection_workflow.select_scenes_for_chip")
    @patch("ftw_dataset_tools.api.imagery.selection_workflow.ImageryProgressBar")
    def test_on_missing_fail_raises(
        self,
        mock_progress_class: MagicMock,
        mock_select: MagicMock,
        mock_catalog_with_chips: Path,
        mock_crop_calendar: MagicMock,
    ) -> None:
        """Test that on_missing='fail' raises exception."""
        from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult

        mock_select.return_value = SceneSelectionResult(
            chip_id="chip_001",
            bbox=(10.0, 50.0, 10.01, 50.01),
            year=2024,
            crop_calendar=mock_crop_calendar,
            planting_scene=None,
            harvest_scene=None,
            skipped_reason="No cloud-free scenes found",
        )
        # Properly set up the context manager mock
        mock_progress = MagicMock()
        mock_progress_class.return_value.__enter__.return_value = mock_progress
        mock_progress_class.return_value.__exit__.return_value = None

        with pytest.raises(ValueError, match="No cloud-free scenes"):
            select_imagery_for_catalog(
                catalog_dir=mock_catalog_with_chips,
                year=2024,
                on_missing="fail",
            )

    @patch("ftw_dataset_tools.api.imagery.selection_workflow.select_scenes_for_chip")
    @patch("ftw_dataset_tools.api.imagery.selection_workflow.ImageryProgressBar")
    def test_on_missing_skip(
        self,
        mock_progress: MagicMock,
        mock_select: MagicMock,
        mock_catalog_with_chips: Path,
        mock_crop_calendar: MagicMock,
    ) -> None:
        """Test that on_missing='skip' records skipped details."""
        from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult

        mock_select.return_value = SceneSelectionResult(
            chip_id="chip_001",
            bbox=(10.0, 50.0, 10.01, 50.01),
            year=2024,
            crop_calendar=mock_crop_calendar,
            planting_scene=None,
            harvest_scene=None,
            skipped_reason="No cloud-free planting scene found",
            candidates_checked=10,
        )
        mock_progress.return_value.__enter__ = MagicMock()
        mock_progress.return_value.__exit__ = MagicMock()

        result = select_imagery_for_catalog(
            catalog_dir=mock_catalog_with_chips,
            year=2024,
            on_missing="skip",
        )

        assert result.skipped == 2
        assert result.skipped_details[0]["candidates_checked"] == 10

    @patch("ftw_dataset_tools.api.imagery.selection_workflow.select_scenes_for_chip")
    @patch("ftw_dataset_tools.api.imagery.selection_workflow.ImageryProgressBar")
    def test_handles_exceptions(
        self,
        mock_progress: MagicMock,
        mock_select: MagicMock,
        mock_catalog_with_chips: Path,
    ) -> None:
        """Test that exceptions are caught and recorded."""
        mock_select.side_effect = RuntimeError("STAC API error")
        mock_progress.return_value.__enter__ = MagicMock()
        mock_progress.return_value.__exit__ = MagicMock()

        result = select_imagery_for_catalog(
            catalog_dir=mock_catalog_with_chips,
            year=2024,
            on_missing="skip",
        )

        assert result.failed == 2
        assert "STAC API error" in result.failed_details[0]["error"]

    @patch("ftw_dataset_tools.api.imagery.selection_workflow.select_scenes_for_chip")
    @patch("ftw_dataset_tools.api.imagery.selection_workflow.ImageryProgressBar")
    def test_exceptions_raise_when_on_missing_fail(
        self,
        mock_progress_class: MagicMock,
        mock_select: MagicMock,
        mock_catalog_with_chips: Path,
    ) -> None:
        """Test that exceptions are re-raised when on_missing='fail'."""
        mock_select.side_effect = RuntimeError("STAC API error")
        # Properly set up the context manager mock
        mock_progress = MagicMock()
        mock_progress_class.return_value.__enter__.return_value = mock_progress
        mock_progress_class.return_value.__exit__.return_value = None

        with pytest.raises(RuntimeError, match="STAC API error"):
            select_imagery_for_catalog(
                catalog_dir=mock_catalog_with_chips,
                year=2024,
                on_missing="fail",
            )

    @patch("ftw_dataset_tools.api.imagery.selection_workflow.select_scenes_for_chip")
    @patch("ftw_dataset_tools.api.imagery.selection_workflow.create_child_items_from_selection")
    @patch("ftw_dataset_tools.api.imagery.selection_workflow.ImageryProgressBar")
    def test_progress_callbacks(
        self,
        mock_progress_class: MagicMock,
        _mock_create_child: MagicMock,
        mock_select: MagicMock,
        mock_catalog_with_chips: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """Test that progress callbacks are called."""
        mock_select.return_value = mock_selection_result

        mock_progress = MagicMock()
        mock_progress_class.return_value.__enter__ = MagicMock(return_value=mock_progress)
        mock_progress_class.return_value.__exit__ = MagicMock()

        select_imagery_for_catalog(
            catalog_dir=mock_catalog_with_chips,
            year=2024,
        )

        # Verify progress methods were called
        assert mock_progress.start_chip.call_count == 2
        assert mock_progress.mark_success.call_count == 2

    @patch("ftw_dataset_tools.api.imagery.selection_workflow.select_scenes_for_chip")
    @patch("ftw_dataset_tools.api.imagery.selection_workflow.create_child_items_from_selection")
    @patch("ftw_dataset_tools.api.imagery.selection_workflow.ImageryProgressBar")
    def test_passes_parameters_to_select_function(
        self,
        mock_progress: MagicMock,
        _mock_create_child: MagicMock,
        mock_select: MagicMock,
        mock_catalog_with_chips: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """Test that parameters are passed to select_scenes_for_chip."""
        mock_select.return_value = mock_selection_result
        mock_progress.return_value.__enter__ = MagicMock()
        mock_progress.return_value.__exit__ = MagicMock()

        select_imagery_for_catalog(
            catalog_dir=mock_catalog_with_chips,
            year=2024,
            cloud_cover_chip=5.0,
            nodata_max=1.0,
            buffer_days=21,
            num_buffer_expansions=5,
            buffer_expansion_size=7,
        )

        # Check the first call's arguments
        call_kwargs = mock_select.call_args_list[0][1]
        assert call_kwargs["year"] == 2024
        assert call_kwargs["cloud_cover_chip"] == 5.0
        assert call_kwargs["nodata_max"] == 1.0
        assert call_kwargs["buffer_days"] == 21
        assert call_kwargs["num_buffer_expansions"] == 5
        assert call_kwargs["buffer_expansion_size"] == 7

    @patch("ftw_dataset_tools.api.imagery.selection_workflow.select_scenes_for_chip")
    @patch("ftw_dataset_tools.api.imagery.selection_workflow.create_child_items_from_selection")
    @patch("ftw_dataset_tools.api.imagery.selection_workflow.ImageryProgressBar")
    def test_passes_parameters_to_create_child(
        self,
        mock_progress: MagicMock,
        mock_create_child: MagicMock,
        mock_select: MagicMock,
        mock_catalog_with_chips: Path,
        mock_selection_result: SceneSelectionResult,
    ) -> None:
        """Test that parameters are passed to create_child_items_from_selection."""
        mock_select.return_value = mock_selection_result
        mock_progress.return_value.__enter__ = MagicMock()
        mock_progress.return_value.__exit__ = MagicMock()

        select_imagery_for_catalog(
            catalog_dir=mock_catalog_with_chips,
            year=2024,
            cloud_cover_chip=5.0,
            buffer_days=21,
            num_buffer_expansions=5,
            buffer_expansion_size=7,
        )

        # Check the first call's arguments
        call_kwargs = mock_create_child.call_args_list[0][1]
        assert call_kwargs["year"] == 2024
        assert call_kwargs["cloud_cover_chip"] == 5.0
        assert call_kwargs["buffer_days"] == 21
        assert call_kwargs["num_buffer_expansions"] == 5
        assert call_kwargs["buffer_expansion_size"] == 7


class TestSelectionWorkflowResult:
    """Tests for SelectionWorkflowResult dataclass."""

    def test_default_values(self) -> None:
        """Test that default values are initialized correctly."""
        result = SelectionWorkflowResult()

        assert result.successful == 0
        assert result.skipped == 0
        assert result.failed == 0
        assert result.skipped_details == []
        assert result.failed_details == []

    def test_mutable_defaults_are_independent(self) -> None:
        """Test that list defaults are independent between instances."""
        result1 = SelectionWorkflowResult()
        result2 = SelectionWorkflowResult()

        result1.skipped_details.append({"chip": "test"})

        assert len(result1.skipped_details) == 1
        assert len(result2.skipped_details) == 0
