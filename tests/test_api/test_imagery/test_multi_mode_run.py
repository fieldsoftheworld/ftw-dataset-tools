"""How a both-modes selection run resolves years and fails fast, before any selection work."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from ftw_dataset_tools.api.imagery import download_workflow, selection_workflow
from ftw_dataset_tools.api.imagery.catalog_ops import SelectionConflictError
from ftw_dataset_tools.api.imagery.mosaic_selection import MosaicYearError
from ftw_dataset_tools.api.imagery.selection_workflow import select_imagery_modes_for_catalog
from ftw_dataset_tools.api.imagery.slots import MOSAIC_SLOTS
from ftw_dataset_tools.api.imagery.summary import selection_summary_lines
from tests.test_api.test_imagery.test_mosaics import _chip
from tests.test_api.test_imagery.test_multi_mode import (
    _YEAR_AVAILABLE,
    _add_mosaics,
    _add_scenes,
    _fake_select,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult


def _run(
    tmp_path: Path, available: object, modes: tuple = ("scenes", "mosaics"), **kwargs: object
) -> tuple:
    def select(job: selection_workflow.ChipSelectionJob, **_options: object) -> MagicMock:
        # Like a real selection: record the year and rewrite the temporal range.
        job.item.properties["ftw:calendar_year"] = job.year
        job.item.properties["start_datetime"] = "2019-06-01T00:00:00+00:00"
        return MagicMock(success=True, skipped_reason=None, candidates_checked=1)

    with (
        patch(_YEAR_AVAILABLE, side_effect=available),
        patch.object(selection_workflow, "run_chip_selection", side_effect=select) as run,
        patch.object(selection_workflow, "ensure_crop_calendar_exists"),
        patch.object(selection_workflow, "_build_scene_index", return_value=None),
    ):
        result = select_imagery_modes_for_catalog(tmp_path, modes, **{"year": None, **kwargs})
    return result, run


def _years(run: MagicMock) -> list[tuple[str, int]]:
    return [(c.kwargs["imagery_mode"], c.args[0].year) for c in run.call_args_list]


class TestOneYearForBothModes:
    """Without a run year both modes use the chip's one stable year."""

    def test_both_modes_use_the_chip_id_year_whatever_selection_rewrites(
        self, tmp_path: Path
    ) -> None:
        _chip(tmp_path, "chip_001_2024")
        result, run = _run(tmp_path, lambda _year: True)
        assert _years(run) == [("scenes", 2024), ("mosaics", 2024)]
        assert result.successful == 2

    def test_a_recorded_mosaic_year_drives_only_the_mosaics(self, tmp_path: Path) -> None:
        # Built before multi-mode support: mosaics selected without a year got 2025.
        _parent, path = _chip(tmp_path, "chip_001_2024")
        _add_mosaics(path, year=2025, requested=2025)
        result, run = _run(tmp_path, lambda _year: True)
        assert _years(run) == [("scenes", 2024)]
        assert result.by_mode["mosaics"].skipped_details[0]["code"] == "already_selected"

    def test_mosaics_from_a_nearby_year_stop_a_rerun(self, tmp_path: Path) -> None:
        # Older ftwd filled a 2024 request with 2023 mosaics.
        _parent, path = _chip(tmp_path, "chip_001_2024")
        _add_mosaics(path, year=2023, requested=2024)
        with pytest.raises(SelectionConflictError, match="has mosaics for 2023"):
            _run(tmp_path, lambda _year: True)

    def test_explicit_years_per_mode_then_a_plain_run_changes_nothing(
        self, tmp_path: Path, mock_selection_result: SceneSelectionResult
    ) -> None:
        _parent, path = _chip(tmp_path, "chip_001")
        _add_scenes(path, mock_selection_result)  # scenes for 2024
        _add_mosaics(path, year=2022, requested=2022)
        result, run = _run(tmp_path, lambda _year: True)
        assert run.call_count == 0
        assert result.skipped == 2

    def test_a_chip_with_no_year_skips_both_modes(self, tmp_path: Path) -> None:
        _chip(tmp_path, "chip_001")
        result, run = _run(tmp_path, lambda _year: True)
        assert run.call_count == 0
        for mode in ("scenes", "mosaics"):
            assert result.by_mode[mode].skipped_details[0]["code"] == "no_year"

    def test_chips_whose_year_has_no_mosaics_skip_only_them(self, tmp_path: Path) -> None:
        _chip(tmp_path, "chip_001_2023")
        result, run = _run(tmp_path, lambda year: year != 2023)
        assert _years(run) == [("scenes", 2023)]
        assert result.by_mode["mosaics"].skipped_details == [
            {
                "chip": "chip_001_2023",
                "code": "no_mosaics_for_year",
                "reason": "Mosaics for 2023 are not available (its quarterly index is incomplete)",
            }
        ]

    def test_an_explicit_year_without_mosaics_skips_them_the_same_way(self, tmp_path: Path) -> None:
        _chip(tmp_path, "chip_001_2024")
        result, run = _run(tmp_path, lambda year: year != 2026, year=2026)
        assert _years(run) == [("scenes", 2026)]
        assert result.by_mode["mosaics"].skipped == 1


class TestFailFast:
    def test_mosaics_alone_without_mosaics_for_a_chip_year_fail_before_work(
        self, tmp_path: Path
    ) -> None:
        _chip(tmp_path, "chip_001_2023")
        with pytest.raises(MosaicYearError, match="2023"):
            _run(tmp_path, lambda year: year != 2023, modes=("mosaics",))

    def test_a_scene_year_conflict_stops_before_the_mosaics(
        self, tmp_path: Path, mock_selection_result: SceneSelectionResult
    ) -> None:
        _parent, path = _chip(tmp_path, "chip_001_2024")
        _add_scenes(path, mock_selection_result)  # calendar year 2024
        calls: list[dict] = []
        with (
            patch(_YEAR_AVAILABLE, return_value=True),
            patch.object(selection_workflow, "_select_mode", _fake_select(calls)),
            pytest.raises(SelectionConflictError, match="has scenes for 2024"),
        ):
            select_imagery_modes_for_catalog(tmp_path, ("scenes", "mosaics"), year=2022)
        assert calls == []

    def test_a_failing_availability_lookup_stops_before_the_scenes(self, tmp_path: Path) -> None:
        _chip(tmp_path, "chip_001_2024")
        with pytest.raises(OSError, match="offline"):
            _run(tmp_path, OSError("offline"))

    def test_a_mosaic_conflict_stops_before_the_scenes(self, tmp_path: Path) -> None:
        _parent, path = _chip(tmp_path)
        _add_mosaics(path, requested=2022)
        calls: list[dict] = []
        with (
            patch(_YEAR_AVAILABLE, return_value=True),
            patch.object(selection_workflow, "_select_mode", _fake_select(calls)),
            pytest.raises(SelectionConflictError, match="has mosaics for 2022"),
        ):
            select_imagery_modes_for_catalog(tmp_path, ("scenes", "mosaics"), year=2024)
        assert calls == []
        assert (path.parent / "chip_001_q1_s2.json").exists()  # nothing was cleared


class TestChipItemsReadOnce:
    def test_a_bare_string_is_one_mode(self, tmp_path: Path) -> None:
        calls: list[dict] = []
        with patch.object(selection_workflow, "_select_mode", _fake_select(calls)):
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
            patch.object(selection_workflow, "_select_mode", _fake_select(calls)),
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
