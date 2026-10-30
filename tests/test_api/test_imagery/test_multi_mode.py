"""Tests for chips holding both imagery modes: scenes and quarterly mosaics side by side."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pystac
import pytest

from ftw_dataset_tools.api.imagery import download_workflow, selection_workflow
from ftw_dataset_tools.api.imagery.catalog_ops import (
    clear_chip_selections,
    has_mode_selection,
    mosaic_requested_year,
    normalize_imagery_properties,
    preserve_imagery_selection,
    preview_slot,
    selected_modes,
    selection_conflict,
)
from ftw_dataset_tools.api.imagery.mosaic_selection import MosaicYearError
from ftw_dataset_tools.api.imagery.selection_workflow import (
    MultiModeSelectionResult,
    SelectionWorkflowResult,
    resolve_selection_conflicts,
    select_imagery_for_catalog,
    select_imagery_modes_for_catalog,
)
from ftw_dataset_tools.api.imagery.slots import (
    MOSAIC_SLOTS,
    InvalidImageryModeError,
    normalize_imagery_modes,
)
from ftw_dataset_tools.api.imagery.stac_child_items import (
    create_child_items_from_selection,
    create_mosaic_child_items,
)
from ftw_dataset_tools.api.imagery.summary import selection_summary_lines
from tests.test_api.test_imagery.test_mosaics import _chip, _selection

if TYPE_CHECKING:
    from pathlib import Path

    from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult

_YEAR_AVAILABLE = "ftw_dataset_tools.api.imagery.mosaic_selection.year_available"
_SCENE_RELS = {"ftw:planting", "ftw:harvest"}
_MOSAIC_RELS = {f"ftw:{slot}" for slot in MOSAIC_SLOTS}


def _add_scenes(path: Path, scenes: SceneSelectionResult) -> pystac.Item:
    parent = pystac.Item.from_file(str(path))
    create_child_items_from_selection(
        chip_dir=path.parent,
        parent_item=parent,
        result=scenes,
        year=2024,
        cloud_cover_chip=2.0,
        buffer_days=14,
    )
    return pystac.Item.from_file(str(path))


def _add_mosaics(path: Path, year: int = 2024, requested: int = 2024) -> pystac.Item:
    parent = pystac.Item.from_file(str(path))
    create_mosaic_child_items(path.parent, parent, _selection(year=year, requested=requested))
    return pystac.Item.from_file(str(path))


def _bare_item(item_id: str) -> pystac.Item:
    return pystac.Item(
        id=item_id,
        geometry=None,
        bbox=None,
        datetime=datetime(2024, 1, 1, tzinfo=UTC),
        properties={},
    )


def _preview(path: Path, slot: str) -> Path:
    """Write ``slot``'s downloaded preview image next to the chip item."""
    preview = path.parent / f"chip_001_{slot}_image_s2.webp"
    preview.write_bytes(b"preview")
    return preview


def _rels(item: pystac.Item) -> set[str]:
    return {link.rel for link in item.links}


@pytest.fixture
def both(tmp_path: Path, mock_selection_result: SceneSelectionResult) -> Path:
    """A chip holding scenes (2024) and then mosaics (requested 2024, used 2025)."""
    _parent, path = _chip(tmp_path)
    _add_scenes(path, mock_selection_result)
    _add_mosaics(path, year=2025)
    return path


class TestNormalizeImageryModes:
    def test_string_is_one_mode(self) -> None:
        assert normalize_imagery_modes("mosaics") == ("mosaics",)

    def test_list_runs_scenes_first_and_collapses_repeats(self) -> None:
        assert normalize_imagery_modes(["mosaics", "scenes", "mosaics"]) == ("scenes", "mosaics")

    @pytest.mark.parametrize("value", [[], (), "monthly", ["scenes", "monthly"], 3])
    def test_bad_values_raise(self, value: object) -> None:
        with pytest.raises(InvalidImageryModeError):
            normalize_imagery_modes(value)  # type: ignore[arg-type]


class TestCoexistence:
    def test_mosaics_join_scenes(self, both: Path) -> None:
        saved = pystac.Item.from_file(str(both))
        assert _rels(saved) >= _SCENE_RELS | _MOSAIC_RELS
        assert selected_modes(saved) == ("scenes", "mosaics")
        props = saved.properties
        assert props["ftw:imagery_modes"] == ["scenes", "mosaics"]
        assert props["ftw:calendar_year"] == 2024
        assert props["ftw:planting_cloud_cover"] == 1.5
        assert props["ftw:mosaic_requested_year"] == 2024
        assert props["ftw:mosaic_year"] == 2025
        for slot in ("planting", "harvest", *MOSAIC_SLOTS):
            assert (both.parent / f"chip_001_{slot}_s2.json").exists()

    def test_temporal_range_spans_both_modes(self, both: Path) -> None:
        props = pystac.Item.from_file(str(both)).properties
        assert props["start_datetime"] == "2024-06-15T10:00:00+00:00"  # planting scene
        assert props["end_datetime"].startswith("2025-12-31")  # end of the 2025 Q4 mosaic

    def test_scenes_join_mosaics(
        self, tmp_path: Path, mock_selection_result: SceneSelectionResult
    ) -> None:
        _parent, path = _chip(tmp_path)
        _add_mosaics(path)
        saved = _add_scenes(path, mock_selection_result)
        assert selected_modes(saved) == ("scenes", "mosaics")
        assert saved.properties["ftw:mosaic_requested_year"] == 2024
        assert saved.properties["ftw:calendar_year"] == 2024

    def test_reselecting_scenes_leaves_mosaics(
        self, both: Path, mock_selection_result: SceneSelectionResult
    ) -> None:
        saved = _add_scenes(both, mock_selection_result)
        assert has_mode_selection(saved, "mosaics")
        assert saved.properties["ftw:mosaic_year"] == 2025
        assert sum(link.rel == "ftw:q1" for link in saved.links) == 1

    def test_reselecting_mosaics_leaves_scenes_and_their_assets(self, both: Path) -> None:
        _preview(both, "planting")
        parent = pystac.Item.from_file(str(both))
        for key in ("planting_image", "q1_image", "thumbnail"):
            parent.add_asset(key, pystac.Asset(href=f"./{key}.tif"))
        create_mosaic_child_items(both.parent, parent, _selection(year=2024))

        saved = pystac.Item.from_file(str(both))
        assert has_mode_selection(saved, "scenes")
        assert saved.properties["ftw:planting_day"] == 150
        assert saved.properties["ftw:mosaic_year"] == 2024
        # The old mosaic images described the replaced selection; the planting
        # image and the planting-drawn preview did not.
        assert "q1_image" not in saved.assets
        assert {"planting_image", "thumbnail"} <= set(saved.assets)


class TestPerModeConflicts:
    def test_only_the_same_mode_can_conflict(self, both: Path) -> None:
        item = pystac.Item.from_file(str(both))
        assert selection_conflict(item, "scenes", 2030) is None
        assert selection_conflict(item, "mosaics", 2024) is None
        assert selection_conflict(item, "mosaics", 2026) == "has mosaics for 2024"

    def test_force_clears_only_the_conflicting_mode(self, both: Path) -> None:
        item = pystac.Item.from_file(str(both))
        cleared = resolve_selection_conflicts(
            [item], imagery_modes="mosaics", year=2026, force=True
        )
        assert cleared == 1
        saved = pystac.Item.from_file(str(both))
        assert selected_modes(saved) == ("scenes",)
        assert (both.parent / "chip_001_planting_s2.json").exists()
        assert not list(both.parent.glob("chip_001_q*_s2.json"))

    def test_a_chip_with_mosaics_still_gets_scenes(self, tmp_path: Path) -> None:
        _parent, path = _chip(tmp_path)
        _add_mosaics(path)
        success = MagicMock(success=True, skipped_reason=None, candidates_checked=1)
        with (
            patch.object(selection_workflow, "run_chip_selection", return_value=success) as run,
            patch.object(selection_workflow, "ensure_crop_calendar_exists"),
            patch.object(selection_workflow, "_build_scene_index", return_value=None),
        ):
            scenes = select_imagery_for_catalog(tmp_path, year=2024, imagery_mode="scenes")
            mosaics = select_imagery_for_catalog(tmp_path, year=2024, imagery_mode="mosaics")
        assert scenes.successful == 1
        assert mosaics.skipped_details == [
            {"chip": "chip_001", "reason": "Already has imagery selections"}
        ]
        assert run.call_count == 1


class TestLegacySingleModeCatalog:
    """Catalogs written before a chip could hold both modes."""

    def _legacy_mosaic_chip(self, tmp_path: Path) -> Path:
        _parent, path = _chip(tmp_path)
        item = _add_mosaics(path)
        props = item.properties
        props.pop("ftw:imagery_modes")
        props["ftw:imagery_mode"] = "mosaics"
        props["ftw:requested_year"] = props.pop("ftw:mosaic_requested_year")
        props["ftw:imagery_year"] = props.pop("ftw:mosaic_year")
        item.save_object(dest_href=str(path))
        return path

    def test_is_read(self, tmp_path: Path) -> None:
        item = pystac.Item.from_file(str(self._legacy_mosaic_chip(tmp_path)))
        assert selected_modes(item) == ("mosaics",)
        assert mosaic_requested_year(item) == 2024
        assert selection_conflict(item, "mosaics", 2025) == "has mosaics for 2024"
        assert selection_conflict(item, "scenes", 2024) is None

    def test_legacy_scene_chip_is_read(self) -> None:
        item = _bare_item("c")
        for rel in _SCENE_RELS:
            item.add_link(pystac.Link(rel=rel, target="./x.json"))
        assert selected_modes(item) == ("scenes",)

    def test_is_migrated_when_scenes_are_added(
        self, tmp_path: Path, mock_selection_result: SceneSelectionResult
    ) -> None:
        saved = _add_scenes(self._legacy_mosaic_chip(tmp_path), mock_selection_result)
        props = saved.properties
        assert props["ftw:imagery_modes"] == ["scenes", "mosaics"]
        assert props["ftw:mosaic_requested_year"] == 2024
        assert props["ftw:mosaic_year"] == 2024
        assert not {"ftw:imagery_mode", "ftw:requested_year", "ftw:imagery_year"} & set(props)

    def test_is_migrated_across_catalog_regeneration(self, tmp_path: Path) -> None:
        path = self._legacy_mosaic_chip(tmp_path)
        fresh = _bare_item("chip_001")
        assert preserve_imagery_selection(fresh, path)
        assert fresh.properties["ftw:imagery_modes"] == ["mosaics"]
        assert fresh.properties["ftw:mosaic_requested_year"] == 2024
        assert "ftw:imagery_mode" not in fresh.properties

    def test_normalize_keeps_a_current_key_over_a_legacy_one(self) -> None:
        item = _bare_item("c")
        item.properties.update({"ftw:mosaic_year": 2025, "ftw:imagery_year": 2019})
        normalize_imagery_properties(item)
        assert item.properties["ftw:mosaic_year"] == 2025
        assert "ftw:imagery_year" not in item.properties
        assert "ftw:imagery_modes" not in item.properties


class TestClearPerMode:
    def _with_preview(self, path: Path) -> pystac.Item:
        _preview(path, "planting")
        (path.parent / "chip_001_overlay.webp").write_bytes(b"preview")
        item = pystac.Item.from_file(str(path))
        item.add_asset("thumbnail", pystac.Asset(href="./chip_001_overlay.webp"))
        item.save_object(dest_href=str(path))
        return pystac.Item.from_file(str(path))

    def test_clearing_mosaics_keeps_scenes_and_their_preview(self, both: Path) -> None:
        item = self._with_preview(both)
        result = clear_chip_selections(item, modes=("mosaics",))
        assert result.stac_items_deleted == 4

        saved = pystac.Item.from_file(str(both))
        assert selected_modes(saved) == ("scenes",)
        assert saved.properties["ftw:imagery_modes"] == ["scenes"]
        assert "ftw:mosaic_year" not in saved.properties
        assert saved.properties["ftw:planting_day"] == 150
        assert saved.properties["end_datetime"] == "2024-09-15T10:00:00+00:00"
        assert "thumbnail" in saved.assets
        assert (both.parent / "chip_001_overlay.webp").exists()

    def test_clearing_scenes_drops_the_preview_drawn_from_them(self, both: Path) -> None:
        item = self._with_preview(both)
        clear_chip_selections(item, modes=("scenes",))

        saved = pystac.Item.from_file(str(both))
        assert selected_modes(saved) == ("mosaics",)
        assert "ftw:calendar_year" not in saved.properties
        assert saved.properties["ftw:mosaic_year"] == 2025
        assert "thumbnail" not in saved.assets
        assert not (both.parent / "chip_001_overlay.webp").exists()
        assert saved.properties["start_datetime"].startswith("2025-01-01")

    def test_clearing_scenes_falls_back_to_the_q3_preview(self, both: Path) -> None:
        _preview(both, "q3")
        item = self._with_preview(both)
        clear_chip_selections(item, modes=("scenes",))

        saved = pystac.Item.from_file(str(both))
        assert saved.assets["thumbnail"].href == "./chip_001_q3_image_s2.webp"
        assert saved.assets["thumbnail"].title == "Chip preview (q3)"

    def test_clearing_scenes_redraws_the_overlay_from_q3(self, both: Path) -> None:
        from ftw_dataset_tools.api.imagery import catalog_ops

        _preview(both, "q3")
        (both.parent / "chip_001_semantic_3_class.tif").write_bytes(b"")
        item = self._with_preview(both)

        def overlay(src: Path, _mask: Path, dst: Path) -> None:
            dst.write_bytes(src.read_bytes() + b"+fields")

        with patch.object(catalog_ops, "generate_overlay_thumbnail", side_effect=overlay):
            clear_chip_selections(item, modes=("scenes",))

        saved = pystac.Item.from_file(str(both))
        assert saved.assets["thumbnail"].href == "./chip_001_overlay.webp"
        assert (both.parent / "chip_001_overlay.webp").read_bytes() == b"preview+fields"

    def test_clearing_everything_restores_a_single_datetime(self, both: Path) -> None:
        clear_chip_selections(pystac.Item.from_file(str(both)))
        saved = pystac.Item.from_file(str(both))
        assert selected_modes(saved) == ()
        assert "start_datetime" not in saved.properties
        assert saved.properties["datetime"].startswith("2024-01-01")
        assert "ftw:imagery_modes" not in saved.properties


class TestPreviewSlot:
    def test_planting_wins_over_q3(self, both: Path) -> None:
        _preview(both, "q3")
        _preview(both, "planting")
        assert preview_slot(both.parent, "chip_001") == "planting"

    def test_q3_when_planting_has_no_preview(self, both: Path) -> None:
        # The planting scene is selected but its image never downloaded.
        _preview(both, "q3")
        assert preview_slot(both.parent, "chip_001") == "q3"

    def test_selected_slots_without_previews_give_none(self, both: Path) -> None:
        assert preview_slot(both.parent, "chip_001") is None

    def test_none_without_imagery(self, tmp_path: Path) -> None:
        _parent, path = _chip(tmp_path)
        assert preview_slot(path.parent, "chip_001") is None


def _fake_select(calls: list[dict]) -> object:
    def fake(**kwargs: object) -> SelectionWorkflowResult:
        calls.append(kwargs)
        return SelectionWorkflowResult(
            successful=2,
            skipped=1,
            skipped_details=[{"chip": "c", "reason": "r"}],
        )

    return fake


class TestSelectImageryModes:
    def test_runs_each_mode_with_its_year(self, tmp_path: Path) -> None:
        calls: list[dict] = []
        with (
            patch(_YEAR_AVAILABLE, return_value=True),
            patch.object(selection_workflow, "select_imagery_for_catalog", _fake_select(calls)),
        ):
            result = select_imagery_modes_for_catalog(
                tmp_path, ["mosaics", "scenes"], year=2022, workers=3
            )
        assert [(c["imagery_mode"], c["year"]) for c in calls] == [
            ("scenes", 2022),
            ("mosaics", 2022),
        ]
        assert all(c["workers"] == 3 for c in calls)
        assert result.successful == 4
        assert result.skipped == 2
        assert {d["mode"] for d in result.skipped_details} == {"scenes", "mosaics"}

    def test_mosaic_year_defaults_to_year(self, tmp_path: Path) -> None:
        calls: list[dict] = []
        with (
            patch(_YEAR_AVAILABLE, return_value=True),
            patch.object(selection_workflow, "select_imagery_for_catalog", _fake_select(calls)),
        ):
            select_imagery_modes_for_catalog(tmp_path, ["mosaics"], year=2021)
        assert calls[0]["year"] == 2021

    @pytest.mark.parametrize("year", [2023, 2026])
    def test_year_without_mosaics_skips_them_and_runs_scenes(
        self, tmp_path: Path, year: int, caplog: pytest.LogCaptureFixture
    ) -> None:
        calls: list[dict] = []
        with (
            patch(_YEAR_AVAILABLE, return_value=False),
            patch.object(selection_workflow, "select_imagery_for_catalog", _fake_select(calls)),
        ):
            result = select_imagery_modes_for_catalog(tmp_path, ["scenes", "mosaics"], year=year)
        assert [c["imagery_mode"] for c in calls] == ["scenes"]
        assert list(result.by_mode) == ["scenes"]
        assert f"Mosaics for {year} are not available" in result.skipped_modes["mosaics"]
        assert any(
            r.levelname == "WARNING" and "mosaics imagery mode was skipped" in r.message
            for r in caplog.records
        )

    def test_year_without_mosaics_fails_when_they_are_the_only_mode(self, tmp_path: Path) -> None:
        calls: list[dict] = []
        with (
            patch(_YEAR_AVAILABLE, return_value=False),
            patch.object(selection_workflow, "select_imagery_for_catalog", _fake_select(calls)),
            pytest.raises(MosaicYearError, match="2023"),
        ):
            select_imagery_modes_for_catalog(tmp_path, ["mosaics"], year=2023)
        assert not calls

    def test_scenes_alone_never_check_mosaic_years(self, tmp_path: Path) -> None:
        calls: list[dict] = []
        with (
            patch(_YEAR_AVAILABLE, side_effect=AssertionError("checked")),
            patch.object(selection_workflow, "select_imagery_for_catalog", _fake_select(calls)),
        ):
            select_imagery_modes_for_catalog(tmp_path, ["scenes"], year=2023)
        assert len(calls) == 1


class TestSelectionSummaryLines:
    def test_one_mode_keeps_the_single_line(self) -> None:
        result = MultiModeSelectionResult(by_mode={"scenes": SelectionWorkflowResult(3)})
        assert selection_summary_lines(result) == ["  Imagery selection: 3 ok, 0 skipped, 0 failed"]

    def test_modes_and_skips_get_a_line_each(self) -> None:
        result = MultiModeSelectionResult(
            by_mode={"scenes": SelectionWorkflowResult(3)},
            skipped_modes={"mosaics": "Mosaics for 2023 are not available."},
        )
        assert selection_summary_lines(result) == [
            "  Imagery selection (scenes): 3 ok, 0 skipped, 0 failed",
            "  Imagery selection (mosaics): skipped - Mosaics for 2023 are not available.",
        ]

    def test_a_plain_result_still_works(self) -> None:
        plain = SimpleNamespace(successful=1, skipped=0, failed=0, failed_details=[])
        assert selection_summary_lines(plain) == ["  Imagery selection: 1 ok, 0 skipped, 0 failed"]


class TestPreviewDrawnFromPlanting:
    """On a chip holding both modes the q3 download leaves the planting preview alone."""

    def _download_q3(self, path: Path) -> MagicMock:
        import numpy as np
        from rasterio.transform import from_bounds

        from ftw_dataset_tools.api.imagery import image_download

        image = path.parent / "chip_001_q3_image_s2.tif"
        image_download.write_cog(
            image,
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
            },
        )
        (path.parent / "chip_001_semantic_3_class.tif").write_bytes(b"")
        child_path = path.parent / "chip_001_q3_s2.json"

        def thumbnail(_src: Path, dst: Path) -> None:
            dst.write_bytes(b"thumb")

        with (
            patch.object(image_download, "generate_thumbnail", side_effect=thumbnail),
            patch.object(image_download, "generate_overlay_thumbnail") as overlay,
            patch.object(image_download, "update_parent_item"),
        ):
            image_download.process_downloaded_scene(
                item=pystac.Item.from_file(str(child_path)),
                item_path=child_path,
                output_path=image,
                output_filename=image.name,
                band_list=["red", "green", "blue", "nir"],
                season="q3",
                base_id="chip_001",
            )
        return overlay

    def test_q3_does_not_draw_over_the_planting_preview(self, both: Path) -> None:
        _preview(both, "planting")
        self._download_q3(both).assert_not_called()

    def test_q3_draws_while_planting_has_no_preview(self, both: Path) -> None:
        self._download_q3(both).assert_called_once()

    def test_q3_draws_the_preview_for_mosaics_alone(self, tmp_path: Path) -> None:
        _parent, path = _chip(tmp_path)
        _add_mosaics(path)
        self._download_q3(path).assert_called_once()


class TestDownloadSlots:
    def test_slots_limit_what_is_downloaded(self, both: Path) -> None:
        seen: list[str] = []

        def fake_run(tasks: list, **_kwargs: object) -> None:
            seen.extend(task.season for task in tasks)

        with patch.object(download_workflow, "run_in_parallel", fake_run):
            download_workflow.download_imagery_for_catalog(
                both.parents[3],
                slots=MOSAIC_SLOTS,
                show_progress_bar=False,
            )
        assert sorted(seen) == sorted(MOSAIC_SLOTS)
