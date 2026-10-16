"""Tests for PlanetScope scene selection over prefetched deliveries (no network)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from ftw_dataset_tools.api.imagery import planet_scenes
from ftw_dataset_tools.api.imagery.crop_calendar import CropCalendarDates
from ftw_dataset_tools.api.imagery.planet_prefetch import PlanetCache, PrefetchResult
from ftw_dataset_tools.api.imagery.planet_scenes import (
    PENDING_REASON,
    PlanetChip,
    select_planet_scenes,
    to_item,
)

if TYPE_CHECKING:
    from pathlib import Path

CHIP = PlanetChip("ftw-32ULA0101_2024", (6.05, 49.60, 6.078, 49.618), 2024)
FOOTPRINT = [[[5.9, 49.4], [6.4, 49.4], [6.4, 49.8], [5.9, 49.8], [5.9, 49.4]]]


def _feature(scene: str, clear: int = 90) -> dict:
    return {
        "id": scene,
        "geometry": {"type": "Polygon", "coordinates": FOOTPRINT},
        "properties": {
            "acquired": "2024-04-06T09:47:29.80Z",
            "cloud_cover": 0.05,
            "clear_percent": clear,
            "satellite_id": "24c0",
            "gsd": 3.7,
        },
    }


def _deliver(
    cache: PlanetCache, scene: str, *, cloud_fraction: float, blackfill: float = 0.0
) -> None:
    """Write a UDM2 over the chip with the given not-clear and blackfill shares."""
    size = 100
    clear = np.ones((size, size), dtype=np.uint8)
    clear[:, : int(size * cloud_fraction)] = 0
    unusable = np.zeros((size, size), dtype=np.uint8)
    unusable[: int(size * blackfill), :] = 1
    bands = np.zeros((8, size, size), dtype=np.uint8)
    bands[0], bands[7] = clear, unusable
    order_dir = cache.root / f"order-{scene}"
    order_dir.mkdir(parents=True, exist_ok=True)
    udm2 = order_dir / f"{scene}_3B_udm2_clip.tif"
    with rasterio.open(
        udm2,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=8,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_bounds(*CHIP.bbox, size, size),
    ) as dst:
        dst.write(bands)
    analytic = order_dir / f"{scene}_3B_AnalyticMS_SR_clip.tif"
    analytic.write_bytes(b"x")
    cache.deliveries.setdefault(scene, []).append(
        {
            "order_id": f"order-{scene}",
            "analytic": str(analytic.relative_to(cache.root)),
            "udm2": str(udm2.relative_to(cache.root)),
        }
    )
    cache.save()


@pytest.fixture
def calendar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        planet_scenes, "get_crop_calendar_dates", lambda _bbox: CropCalendarDates(97, 263)
    )


class StubPrefetch:
    """Hands out fixed candidates per season and records the windows asked for."""

    def __init__(self, by_season: dict[str, list[dict]], pending: bool = False) -> None:
        self.by_season = by_season
        self.pending = pending
        self.calls: list[list] = []

    def __call__(self, windows, _root, **_kwargs) -> PrefetchResult:
        self.calls.append(list(windows))
        result = PrefetchResult(pending_orders=["order-x"] if self.pending else [])
        for w in windows:
            result.candidates[(w.chip_id, w.season)] = self.by_season.get(w.season, [])
        return result


def _select(tmp_path: Path, **kwargs) -> planet_scenes.SceneSelectionResult:
    options = {
        "cloud_cover_chip": 2.0,
        "nodata_max": 0.0,
        "buffer_days": 14,
        "num_buffer_expansions": 1,
        "buffer_expansion_size": 14,
        "workers": 2,
    }
    options.update(kwargs)
    return select_planet_scenes([CHIP], tmp_path, **options)[CHIP.chip_id]


class TestToItem:
    def test_converts_cloud_to_percent_and_keeps_planet_fields(self) -> None:
        item = to_item(_feature("s1", clear=74))

        assert item.datetime == datetime(2024, 4, 6, 9, 47, 29, 800000, tzinfo=UTC)
        assert item.properties["eo:cloud_cover"] == 5.0
        assert item.properties["pl:clear_percent"] == 74
        assert item.get_self_href().endswith("/item-types/PSScene/items/s1")


class TestSelection:
    @pytest.mark.usefixtures("calendar")
    def test_picks_the_clearest_chip_not_the_clearest_scene(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = PlanetCache(tmp_path)
        _deliver(cache, "scene_clear", cloud_fraction=0.2)
        _deliver(cache, "chip_clear", cloud_fraction=0.0)
        features = [_feature("scene_clear", 74), _feature("chip_clear", 66)]
        monkeypatch.setattr(
            planet_scenes,
            "prefetch_scenes",
            StubPrefetch({"planting": features, "harvest": features}),
        )

        result = _select(tmp_path)

        assert result.success
        assert result.planting_scene.id == "chip_clear"
        assert result.planting_scene.cloud_cover == pytest.approx(0.0)
        assets = result.planting_scene.item.assets
        assert assets["analytic"].href.endswith("chip_clear_3B_AnalyticMS_SR_clip.tif")
        assert result.planting_scene.item.properties["ftw:planet_order_id"] == "order-chip_clear"
        assert result.selection_params["stac_host"] == "planet-data-api"

    @pytest.mark.usefixtures("calendar")
    def test_cloud_and_nodata_limits_apply(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = PlanetCache(tmp_path)
        _deliver(cache, "cloudy", cloud_fraction=0.3)
        _deliver(cache, "edge", cloud_fraction=0.0, blackfill=0.3)
        features = [_feature("cloudy"), _feature("edge")]
        monkeypatch.setattr(
            planet_scenes,
            "prefetch_scenes",
            StubPrefetch({"planting": features, "harvest": features}),
        )

        result = _select(tmp_path, num_buffer_expansions=0)

        assert not result.success
        assert result.skipped_reason == "No cloud-free scenes found for either season"

    @pytest.mark.usefixtures("calendar")
    def test_unresolved_season_widens_and_tries_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = PlanetCache(tmp_path)
        _deliver(cache, "clear", cloud_fraction=0.0)
        stub = StubPrefetch({"planting": [_feature("clear")], "harvest": []})
        monkeypatch.setattr(planet_scenes, "prefetch_scenes", stub)

        result = _select(tmp_path, num_buffer_expansions=1)

        assert [len(call) for call in stub.calls] == [2, 1]
        assert stub.calls[1][0].season == "harvest"
        assert stub.calls[1][0].buffer_days == 28
        assert result.skipped_reason == "No cloud-free harvest scene found"
        assert result.harvest_buffer_used == 28
        assert result.expansions_performed == 1

    @pytest.mark.usefixtures("calendar")
    def test_pending_orders_are_reported_as_pending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub = StubPrefetch({}, pending=True)
        monkeypatch.setattr(planet_scenes, "prefetch_scenes", stub)

        result = _select(tmp_path, num_buffer_expansions=2)

        assert result.skipped_reason == PENDING_REASON
        assert len(stub.calls) == 1  # no widening while orders are processing

    def test_crop_calendar_error_skips_the_chip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(_bbox):
            raise ValueError("no calendar here")

        monkeypatch.setattr(planet_scenes, "get_crop_calendar_dates", fail)
        stub = StubPrefetch({})
        monkeypatch.setattr(planet_scenes, "prefetch_scenes", stub)

        result = _select(tmp_path)

        assert result.skipped_reason == "Crop calendar error: no calendar here"
        assert stub.calls == []
