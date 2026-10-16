"""Tests for PlanetScope scene search and per-chip scene choice (no network)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from ftw_dataset_tools.api.imagery import planet_api, planet_scenes
from ftw_dataset_tools.api.imagery.planet_scenes import (
    query_planet,
    select_best_planet_scene,
    to_item,
)
from ftw_dataset_tools.api.imagery.scene_selection import select_scenes_for_chip

CHIP = (6.05, 49.60, 6.078, 49.618)
# A footprint that covers the chip with room to spare, and one that covers half of it.
COVERING = [[[6.0, 49.5], [6.2, 49.5], [6.2, 49.7], [6.0, 49.7], [6.0, 49.5]]]
HALF = [[[6.064, 49.5], [6.2, 49.5], [6.2, 49.7], [6.064, 49.7], [6.064, 49.5]]]


def _feature(scene_id: str, *, clear: int = 90, cloud: float = 0.05, coords=COVERING) -> dict:
    return {
        "id": scene_id,
        "geometry": {"type": "Polygon", "coordinates": coords},
        "properties": {
            "acquired": "2024-06-03T10:42:34.71Z",
            "cloud_cover": cloud,
            "clear_percent": clear,
            "instrument": "PSB.SD",
            "satellite_id": "24fb",
            "gsd": 4,
        },
    }


class TestToItem:
    def test_converts_cloud_to_percent_and_keeps_planet_fields(self) -> None:
        item = to_item(_feature("s1", cloud=0.16, clear=74))

        assert item.id == "s1"
        assert item.datetime == datetime(2024, 6, 3, 10, 42, 34, 710000, tzinfo=UTC)
        assert item.properties["eo:cloud_cover"] == 16.0
        assert item.properties["pl:clear_percent"] == 74
        assert item.get_self_href().endswith("/item-types/PSScene/items/s1")


class TestQueryPlanet:
    def test_sorts_clearest_first_and_uses_the_window(self) -> None:
        features = [_feature("cloudy", clear=40), _feature("clear", clear=95)]
        with patch.object(planet_api, "search_scenes", return_value=features) as search:
            result = query_planet(CHIP, datetime(2024, 6, 10), 75, 14)

        assert [item.id for item in result.items] == ["clear", "cloudy"]
        _bbox, start, end = search.call_args.args
        assert (start.date(), end.date()) == (
            datetime(2024, 5, 27).date(),
            datetime(2024, 6, 24).date(),
        )
        assert search.call_args.kwargs == {"cloud_cover_max": 75}

    def test_api_errors_become_query_errors(self) -> None:
        error = planet_api.PlanetAPIError("u", "boom", 500)
        with (
            patch.object(planet_api, "search_scenes", side_effect=error),
            pytest.raises(ValueError, match="boom"),
        ):
            query_planet(CHIP, datetime(2024, 6, 10), 75, 14)


class TestSelectBest:
    def test_picks_the_clearest_chip_not_the_clearest_scene(self) -> None:
        # Mirrors the live Luxembourg check: the scene that is less clear overall is
        # nearly cloud-free over this chip.
        items = [
            to_item(_feature("scene_clear", clear=74)),
            to_item(_feature("chip_clear", clear=66)),
        ]
        chip_clear = {"scene_clear": 80.0, "chip_clear": 99.0}
        with patch.object(
            planet_api, "chip_clear_percent", side_effect=lambda i, _b: chip_clear[i]
        ):
            scene = select_best_planet_scene(
                items, "planting", CHIP, cloud_cover_chip=5, nodata_max=0
            )

        assert scene is not None
        assert scene.id == "chip_clear"
        assert scene.cloud_cover == pytest.approx(1.0)
        assert scene.season == "planting"

    def test_scene_not_covering_the_chip_is_skipped(self) -> None:
        items = [to_item(_feature("half", coords=HALF))]
        with patch.object(planet_api, "chip_clear_percent") as coverage:
            assert (
                select_best_planet_scene(items, "harvest", CHIP, cloud_cover_chip=5, nodata_max=0)
                is None
            )
        coverage.assert_not_called()

    def test_nothing_within_the_cloud_limit(self) -> None:
        items = [to_item(_feature("s1"))]
        with patch.object(planet_api, "chip_clear_percent", return_value=50.0):
            assert (
                select_best_planet_scene(items, "harvest", CHIP, cloud_cover_chip=5, nodata_max=0)
                is None
            )

    def test_failed_udm2_falls_back_to_scene_cloud(self) -> None:
        items = [to_item(_feature("s1", cloud=0.01))]
        error = planet_api.PlanetAPIError("u", "coverage did not complete")
        with patch.object(planet_api, "chip_clear_percent", side_effect=error):
            scene = select_best_planet_scene(
                items, "planting", CHIP, cloud_cover_chip=5, nodata_max=0
            )

        assert scene is not None
        assert scene.cloud_cover == 1.0

    def test_missing_key_is_not_swallowed(self) -> None:
        items = [to_item(_feature("s1"))]
        error = planet_api.PlanetUnavailableError("Set PL_API_KEY")
        with (
            patch.object(planet_api, "chip_clear_percent", side_effect=error),
            pytest.raises(planet_api.PlanetUnavailableError),
        ):
            select_best_planet_scene(items, "planting", CHIP, cloud_cover_chip=5, nodata_max=0)

    def test_rates_at_most_the_top_candidates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(planet_scenes, "MAX_UDM2_CANDIDATES", 2)
        items = [to_item(_feature(f"s{n}")) for n in range(5)]
        with patch.object(planet_api, "chip_clear_percent", return_value=100.0) as coverage:
            select_best_planet_scene(items, "planting", CHIP, cloud_cover_chip=5, nodata_max=0)

        assert sorted(call.args[0] for call in coverage.call_args_list) == ["s0", "s1"]


class TestSelectScenesForChip:
    def test_planetscope_source_uses_planet_search_and_rating(self) -> None:
        features = [_feature("s1")]
        with (
            patch.object(planet_api, "search_scenes", return_value=features),
            patch.object(planet_api, "chip_clear_percent", return_value=100.0),
        ):
            result = select_scenes_for_chip("chip_2023", CHIP, 2023, source="planetscope")

        assert result.success
        assert result.planting_scene.id == "s1"
        assert result.harvest_scene.id == "s1"
        assert result.selection_params["stac_host"] == "planet-data-api"
