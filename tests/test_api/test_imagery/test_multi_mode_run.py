"""How a both-modes selection run resolves years and fails fast, before any selection work."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from ftw_dataset_tools.api.imagery import download_workflow, selection_workflow
from ftw_dataset_tools.api.imagery.catalog_ops import SelectionConflictError
from ftw_dataset_tools.api.imagery.selection_workflow import select_imagery_modes_for_catalog
from ftw_dataset_tools.api.imagery.slots import MOSAIC_SLOTS
from ftw_dataset_tools.api.imagery.summary import selection_summary_lines
from tests.test_api.test_imagery.test_mosaics import _chip
from tests.test_api.test_imagery.test_multi_mode import (
    _YEAR_AVAILABLE,
    _add_mosaics,
    _fake_select,
)

if TYPE_CHECKING:
    from pathlib import Path


def _run(tmp_path: Path, available: object) -> tuple:
    success = MagicMock(success=True, skipped_reason=None, candidates_checked=1)
    with (
        patch(_YEAR_AVAILABLE, side_effect=available),
        patch.object(selection_workflow, "run_chip_selection", return_value=success) as run,
        patch.object(selection_workflow, "ensure_crop_calendar_exists"),
        patch.object(selection_workflow, "_build_scene_index", return_value=None),
    ):
        result = select_imagery_modes_for_catalog(tmp_path, ("scenes", "mosaics"), year=None)
    return result, run


class TestOneYearForBothModes:
    """Without a run year both modes use each chip's own year."""

    def test_mosaics_follow_the_chip_year_not_the_fallback(self, tmp_path: Path) -> None:
        _chip(tmp_path)  # dated 2024
        result, run = _run(tmp_path, lambda _year: True)
        assert [(c.kwargs["imagery_mode"], c.args[0].year) for c in run.call_args_list] == [
            ("scenes", 2024),
            ("mosaics", 2024),
        ]
        assert result.successful == 2

    def test_chips_whose_year_has_no_mosaics_skip_only_them(self, tmp_path: Path) -> None:
        _chip(tmp_path)
        result, run = _run(tmp_path, lambda year: year != 2024)
        assert [c.kwargs["imagery_mode"] for c in run.call_args_list] == ["scenes"]
        assert result.skipped_modes == {}
        assert result.by_mode["mosaics"].skipped_details == [
            {
                "chip": "chip_001",
                "reason": "Mosaics for 2024 are not available (its quarterly index is incomplete)",
            }
        ]

    def test_a_chip_with_no_year_falls_back_for_mosaics_only(self) -> None:
        item = SimpleNamespace(id="chip", properties={}, datetime=None)
        assert selection_workflow.mode_year(item, "mosaics", None) == 2025
        assert selection_workflow.mode_year(item, "scenes", None) is None
        assert selection_workflow.mode_year(item, "mosaics", 2021) == 2021


class TestFailFast:
    def test_a_failing_availability_lookup_stops_before_the_scenes(self, tmp_path: Path) -> None:
        _chip(tmp_path)
        with pytest.raises(OSError, match="offline"):
            _run(tmp_path, OSError("offline"))

    def test_a_mosaic_conflict_stops_before_the_scenes(self, tmp_path: Path) -> None:
        _parent, path = _chip(tmp_path)
        _add_mosaics(path, requested=2022)
        calls: list[dict] = []
        with (
            patch(_YEAR_AVAILABLE, return_value=True),
            patch.object(selection_workflow, "select_imagery_for_catalog", _fake_select(calls)),
            pytest.raises(SelectionConflictError, match="has mosaics for 2022"),
        ):
            select_imagery_modes_for_catalog(tmp_path, ("scenes", "mosaics"), year=2024)
        assert calls == []
        assert (path.parent / "chip_001_q1_s2.json").exists()  # nothing was cleared


class TestChipItemsReadOnce:
    def test_a_bare_string_is_one_mode(self, tmp_path: Path) -> None:
        calls: list[dict] = []
        with patch.object(selection_workflow, "select_imagery_for_catalog", _fake_select(calls)):
            select_imagery_modes_for_catalog(tmp_path, "scenes", year=2024)
        assert [c["imagery_mode"] for c in calls] == ["scenes"]

    def test_unreadable_chips_are_counted_once(self, tmp_path: Path) -> None:
        _parent, path = _chip(tmp_path)
        broken = path.parent.parent / "chip_002"
        broken.mkdir()
        (broken / "chip_002.json").write_text("{not json")
        calls: list[dict] = []
        with (
            patch(_YEAR_AVAILABLE, return_value=True),
            patch.object(selection_workflow, "select_imagery_for_catalog", _fake_select(calls)),
        ):
            result = select_imagery_modes_for_catalog(tmp_path, ("scenes", "mosaics"), year=2024)
        assert calls[0]["chip_items"] is calls[1]["chip_items"]
        assert [item.id for item, _p in calls[0]["chip_items"]] == ["chip_001"]
        assert result.failed == 1
        assert result.failed_details[0]["chip"] == "chip_002"
        assert selection_summary_lines(result)[-1].startswith("  Unreadable chips: 1")


class TestDownloadSlotFilter:
    def test_an_excluded_slot_is_never_read(self, tmp_path: Path) -> None:
        _parent, path = _chip(tmp_path)
        _add_mosaics(path)
        (path.parent / "chip_001_planting_s2.json").write_text("{not json")
        unreadable: list[dict] = []
        found = download_workflow.find_s2_child_items(
            tmp_path, unreadable=unreadable, slots=MOSAIC_SLOTS
        )
        assert unreadable == []
        assert sorted(item.id for item, _p in found) == [f"chip_001_q{n}_s2" for n in (1, 2, 3, 4)]
