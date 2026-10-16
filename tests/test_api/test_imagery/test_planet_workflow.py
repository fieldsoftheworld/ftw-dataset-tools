"""PlanetScope selection through ``select_imagery_for_catalog`` (no network)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pystac
import pytest

from ftw_dataset_tools.api.imagery import selection_workflow
from ftw_dataset_tools.api.imagery.catalog_ops import has_existing_scenes
from ftw_dataset_tools.api.imagery.crop_calendar import CropCalendarDates
from ftw_dataset_tools.api.imagery.planet_scenes import PENDING_REASON
from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult, SelectedScene
from ftw_dataset_tools.api.imagery.selection_workflow import select_imagery_for_catalog

from .test_selection_workflow import _write_chip_catalog

if TYPE_CHECKING:
    from pathlib import Path

CHIP = "ftw-33UXP0101_2024"


def _scene(cache: Path, scene_id: str, season: str) -> SelectedScene:
    order = cache / "order-1"
    order.mkdir(parents=True, exist_ok=True)
    analytic = order / f"{scene_id}_3B_AnalyticMS_SR_clip.tif"
    udm2 = order / f"{scene_id}_3B_udm2_clip.tif"
    analytic.write_bytes(b"x")
    udm2.write_bytes(b"x")
    item = pystac.Item(
        scene_id,
        {"type": "Point", "coordinates": [10.0, 50.0]},
        [10.0, 50.0, 10.01, 50.01],
        datetime(2024, 4, 6, tzinfo=UTC),
        {
            "pl:clear_percent": 88,
            "pl:satellite_id": "24c0",
            "pl:gsd": 3.7,
            "ftw:planet_order_id": "order-1",
        },
    )
    item.add_asset("analytic", pystac.Asset(str(analytic)))
    item.add_asset("udm2", pystac.Asset(str(udm2)))
    return SelectedScene(item, season, 0.5, item.datetime, f"https://api.planet.com/x/{scene_id}")


def _selection(chip: str, planting=None, harvest=None, reason=None) -> SceneSelectionResult:
    return SceneSelectionResult(
        chip_id=chip,
        bbox=(10.0, 50.0, 10.01, 50.01),
        year=2024,
        crop_calendar=CropCalendarDates(97, 263),
        planting_scene=planting,
        harvest_scene=harvest,
        skipped_reason=reason,
        selection_params={"cloud_cover_chip_threshold": 2.0, "buffer_days": 14},
    )


@pytest.fixture
def stub_planet(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    def install(make_result):
        def select(chips, cache_root, **kwargs):
            calls.append({"chips": [c.chip_id for c in chips], "cache": cache_root, **kwargs})
            return {c.chip_id: make_result(c.chip_id, cache_root) for c in chips}

        monkeypatch.setattr(selection_workflow, "select_planet_scenes", select)
        return calls

    return install


def _both(chip: str, cache: Path) -> SceneSelectionResult:
    return _selection(chip, _scene(cache, "p1", "planting"), _scene(cache, "h1", "harvest"))


def _load(catalog: Path, name: str) -> pystac.Item:
    return pystac.Item.from_file(str(catalog / "chips" / "33UXP" / CHIP / f"{name}.json"))


class TestPlanetWorkflow:
    def test_writes_planet_children_and_parent_links(self, tmp_path: Path, stub_planet) -> None:
        catalog = _write_chip_catalog(tmp_path, [CHIP])
        calls = stub_planet(_both)

        result = select_imagery_for_catalog(catalog, year=2024, source="planetscope")

        assert result.successful == 1
        assert calls[0]["cache"] == catalog / ".planet"
        parent = _load(catalog, CHIP)
        assert has_existing_scenes(parent, "planetscope")
        assert not has_existing_scenes(parent, "sentinel-2")
        assert parent.properties["ftw:planet:planting_day"] == 97
        assert parent.properties["ftw:planet:harvest_cloud_cover"] == 0.5
        child = _load(catalog, f"{CHIP}_planting_planet")
        assert child.properties["ftw:source"] == "planetscope"
        assert child.properties["ftw:scene_id"] == "p1"
        assert child.properties["ftw:planet_order_id"] == "order-1"
        assert child.properties["ftw:cloud_mask"] == "udm2"
        assert child.properties["platform"] == "24c0"
        assert child.assets["analytic"].href.startswith("../../../.planet/order-1/")
        assert child.get_single_link("via").href.endswith("/p1")

    def test_sentinel2_selection_on_the_chip_is_left_alone(
        self, tmp_path: Path, stub_planet
    ) -> None:
        catalog = _write_chip_catalog(tmp_path, [CHIP])
        parent_path = catalog / "chips" / "33UXP" / CHIP / f"{CHIP}.json"
        parent = pystac.Item.from_file(str(parent_path))
        parent.add_link(pystac.Link("ftw:planting", f"./{CHIP}_planting_s2.json"))
        parent.add_link(pystac.Link("ftw:harvest", f"./{CHIP}_harvest_s2.json"))
        parent.properties["ftw:planting_cloud_cover"] = 1.0
        parent.save_object(dest_href=str(parent_path))
        stub_planet(_both)

        select_imagery_for_catalog(catalog, year=2024, source="planetscope")

        parent = _load(catalog, CHIP)
        assert has_existing_scenes(parent, "sentinel-2")
        assert has_existing_scenes(parent, "planetscope")
        assert parent.properties["ftw:planting_cloud_cover"] == 1.0

    def test_chip_with_planet_imagery_is_skipped_unless_forced(
        self, tmp_path: Path, stub_planet
    ) -> None:
        catalog = _write_chip_catalog(tmp_path, [CHIP])
        calls = stub_planet(_both)
        select_imagery_for_catalog(catalog, year=2024, source="planetscope")

        again = select_imagery_for_catalog(catalog, year=2024, source="planetscope")
        assert again.skipped_details == [{"chip": CHIP, "reason": "Already has imagery selections"}]
        assert len(calls) == 1

        forced = select_imagery_for_catalog(catalog, year=2024, source="planetscope", force=True)
        assert forced.successful == 1

    def test_pending_orders_count_as_skipped(self, tmp_path: Path, stub_planet) -> None:
        catalog = _write_chip_catalog(tmp_path, [CHIP])
        stub_planet(lambda chip, _cache: _selection(chip, reason=PENDING_REASON))

        result = select_imagery_for_catalog(catalog, year=2024, source="planetscope")

        assert result.successful == 0
        assert result.skipped_details[0]["reason"] == PENDING_REASON
        assert not has_existing_scenes(_load(catalog, CHIP), "planetscope")

    def test_options_reach_planet_selection(self, tmp_path: Path, stub_planet) -> None:
        catalog = _write_chip_catalog(tmp_path, [CHIP])
        calls = stub_planet(_both)

        select_imagery_for_catalog(
            catalog,
            year=2024,
            source="planetscope",
            cloud_cover_chip=5.0,
            planet_wait=False,
            planet_timeout_minutes=10,
        )

        assert calls[0]["chips"] == [CHIP]
        assert calls[0]["cloud_cover_chip"] == 5.0
        assert calls[0]["wait"] is False
        assert calls[0]["timeout_minutes"] == 10

    def test_planet_mosaics_are_not_supported_yet(self, tmp_path: Path) -> None:
        catalog = _write_chip_catalog(tmp_path, [CHIP])

        with pytest.raises(ValueError, match="scenes mode only"):
            select_imagery_for_catalog(
                catalog, year=2024, source="planetscope", imagery_mode="mosaics"
            )


class TestCatalogBookkeeping:
    def test_catalog_rebuild_keeps_a_planet_only_selection(
        self, tmp_path: Path, stub_planet
    ) -> None:
        from ftw_dataset_tools.api.imagery.catalog_ops import preserve_imagery_selection

        catalog = _write_chip_catalog(tmp_path, [CHIP])
        stub_planet(_both)
        select_imagery_for_catalog(catalog, year=2024, source="planetscope")
        parent_path = catalog / "chips" / "33UXP" / CHIP / f"{CHIP}.json"
        rebuilt = pystac.Item(CHIP, None, None, datetime(2024, 1, 1, tzinfo=UTC), {})

        assert preserve_imagery_selection(rebuilt, parent_path)

        assert has_existing_scenes(rebuilt, "planetscope")
        assert rebuilt.properties["ftw:planet:planting_day"] == 97

    def test_clearing_sentinel2_keeps_planet_imagery(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.imagery.catalog_ops import clear_chip_selections

        catalog = _write_chip_catalog(tmp_path, [CHIP])
        chip_dir = catalog / "chips" / "33UXP" / CHIP
        s2 = chip_dir / f"{CHIP}_planting_image_s2.tif"
        planet = chip_dir / f"{CHIP}_planting_image_planet.tif"
        s2.write_bytes(b"x")
        planet.write_bytes(b"x")

        clear_chip_selections(_load(catalog, CHIP))

        assert not s2.exists()
        assert planet.exists()
