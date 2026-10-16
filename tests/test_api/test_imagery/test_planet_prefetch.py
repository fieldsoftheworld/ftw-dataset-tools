"""Tests for the run-level PlanetScope prefetch (no network)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from ftw_dataset_tools.api.imagery import planet_api, planet_prefetch
from ftw_dataset_tools.api.imagery.planet_prefetch import ChipWindow, PlanetCache, prefetch_scenes

if TYPE_CHECKING:
    from pathlib import Path

# Two chips in square 32ULA and one in 32ULB; every scene footprint covers all three.
FOOTPRINT = [[[5.9, 49.4], [6.4, 49.4], [6.4, 49.8], [5.9, 49.8], [5.9, 49.4]]]
JUNE = datetime(2024, 6, 10)


def _window(chip: str, *, season: str = "planting", x: float = 6.05) -> ChipWindow:
    return ChipWindow(chip, (x, 49.60, x + 0.028, 49.618), season, JUNE, 14)


def _feature(
    scene: str, *, clear: int, acquired: str = "2024-06-05T10:00:00Z", coords=FOOTPRINT
) -> dict:
    return {
        "id": scene,
        "geometry": {"type": "Polygon", "coordinates": coords},
        "properties": {"acquired": acquired, "clear_percent": clear, "cloud_cover": 0.1},
    }


class FakePlanet:
    """Planet's search, orders and downloads, in memory."""

    def __init__(self, features: list[dict], final_state: str = "success") -> None:
        self.features = features
        self.final_state = final_state
        self.searches: list[tuple] = []
        self.orders: dict[str, dict] = {}

    def search_scenes(self, bbox, start, end, *, cloud_cover_max):
        self.searches.append((bbox, start, end, cloud_cover_max))
        return self.features

    def create_order(self, name, item_ids, *, aoi, harmonize=False):
        order_id = f"order-{len(self.orders) + 1}"
        self.orders[order_id] = {
            "name": name,
            "items": list(item_ids),
            "aoi": aoi,
            "harmonize": harmonize,
        }
        return order_id

    def get_order(self, order_id):
        return {"id": order_id, "state": self.final_state}

    def download_order(self, order, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for scene in self.orders[order["id"]]["items"]:
            for suffix in ("_3B_AnalyticMS_SR_clip.tif", "_3B_udm2_clip.tif", "_metadata.json"):
                path = directory / f"{scene}{suffix}"
                path.write_bytes(b"x")
                paths.append(path)
        return paths


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakePlanet:
    fake = FakePlanet(
        [_feature("best", clear=99), _feature("good", clear=90), _feature("poor", clear=40)]
    )
    for name in ("search_scenes", "create_order", "get_order", "download_order"):
        monkeypatch.setattr(planet_api, name, getattr(fake, name))
    return fake


def _prefetch(windows: list[ChipWindow], root: Path, **kwargs) -> planet_prefetch.PrefetchResult:
    return prefetch_scenes(
        windows, root, cloud_cover_max=75, nodata_max=0, poll_seconds=0, **kwargs
    )


class TestSearchAndChoice:
    def test_one_search_per_square_and_season(self, fake: FakePlanet, tmp_path: Path) -> None:
        windows = [
            _window("ftw-32ULA0101_2024"),
            _window("ftw-32ULA0102_2024", x=6.08),
            _window("ftw-32ULB0101_2024"),
            _window("ftw-32ULA0101_2024", season="harvest"),
        ]

        _prefetch(windows, tmp_path)

        assert len(fake.searches) == 3

    def test_each_chip_keeps_its_clearest_covering_scenes(
        self, fake: FakePlanet, tmp_path: Path
    ) -> None:
        result = _prefetch([_window("ftw-32ULA0101_2024")], tmp_path, candidates_per_chip=2)

        picks = result.candidates[("ftw-32ULA0101_2024", "planting")]
        assert [f["id"] for f in picks] == ["best", "good"]
        assert fake.orders["order-1"]["items"] == ["best", "good"]

    def test_scenes_outside_the_window_or_chip_are_not_candidates(
        self, fake: FakePlanet, tmp_path: Path
    ) -> None:
        elsewhere = [[[7.0, 49.4], [7.5, 49.4], [7.5, 49.8], [7.0, 49.8], [7.0, 49.4]]]
        fake.features = [
            _feature("too_late", clear=99, acquired="2024-08-01T10:00:00Z"),
            _feature("elsewhere", clear=99, coords=elsewhere),
            _feature("ok", clear=50),
        ]

        result = _prefetch([_window("ftw-32ULA0101_2024")], tmp_path)

        assert [f["id"] for f in result.candidates[("ftw-32ULA0101_2024", "planting")]] == ["ok"]

    def test_shared_scenes_are_ordered_once_per_group(
        self, fake: FakePlanet, tmp_path: Path
    ) -> None:
        windows = [_window("ftw-32ULA0101_2024"), _window("ftw-32ULA0102_2024", x=6.08)]

        _prefetch(windows, tmp_path)

        assert len(fake.orders) == 1
        assert sorted(fake.orders["order-1"]["items"]) == ["best", "good", "poor"]
        assert fake.orders["order-1"]["aoi"]["type"] in ("Polygon", "MultiPolygon")
        assert fake.orders["order-1"]["harmonize"] is False

    def test_large_groups_are_split_into_batches(
        self, fake: FakePlanet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(planet_prefetch, "ORDER_BATCH", 2)

        _prefetch([_window("ftw-32ULA0101_2024")], tmp_path)

        assert [len(o["items"]) for o in fake.orders.values()] == [2, 1]


class TestDeliveryAndResume:
    @pytest.mark.usefixtures("fake")
    def test_deliveries_are_recorded_in_the_manifest(self, tmp_path: Path) -> None:
        _prefetch([_window("ftw-32ULA0101_2024")], tmp_path)

        cache = PlanetCache(tmp_path)
        (delivery,) = cache.delivered("best")
        assert delivery.order_id == "order-1"
        assert delivery.analytic.name == "best_3B_AnalyticMS_SR_clip.tif"
        assert delivery.udm2.name == "best_3B_udm2_clip.tif"
        assert delivery.analytic.is_file()

    def test_rerun_does_not_order_again(self, fake: FakePlanet, tmp_path: Path) -> None:
        window = _window("ftw-32ULA0101_2024")
        _prefetch([window], tmp_path)

        result = _prefetch([window], tmp_path)

        assert len(fake.orders) == 1
        assert len(result.candidates[(window.chip_id, "planting")]) == 3

    def test_pending_orders_leave_chips_without_candidates(
        self, fake: FakePlanet, tmp_path: Path
    ) -> None:
        fake.final_state = "running"

        result = _prefetch([_window("ftw-32ULA0101_2024")], tmp_path, wait=False)

        assert result.pending_orders == ["order-1"]
        assert result.candidates[("ftw-32ULA0101_2024", "planting")] == []

    def test_pending_order_is_collected_on_the_next_run(
        self, fake: FakePlanet, tmp_path: Path
    ) -> None:
        window = _window("ftw-32ULA0101_2024")
        fake.final_state = "running"
        _prefetch([window], tmp_path, wait=False)

        fake.final_state = "success"
        result = _prefetch([window], tmp_path, wait=False)

        assert len(fake.orders) == 1
        assert result.pending_orders == []
        assert len(result.candidates[(window.chip_id, "planting")]) == 3

    def test_failed_order_is_reported_and_reordered_next_time(
        self, fake: FakePlanet, tmp_path: Path
    ) -> None:
        window = _window("ftw-32ULA0101_2024")
        fake.final_state = "failed"

        result = _prefetch([window], tmp_path)
        assert result.failed_orders == ["order-1"]

        fake.final_state = "success"
        _prefetch([window], tmp_path)
        assert len(fake.orders) == 2
