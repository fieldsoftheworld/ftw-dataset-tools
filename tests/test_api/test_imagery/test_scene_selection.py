"""Tests for scene selection internals that the thread pool depends on."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from ftw_dataset_tools.api.imagery import scene_selection


@pytest.fixture(autouse=True)
def _clear_thread_client_cache() -> None:
    """Drop this thread's cached clients so each test starts cold."""
    if hasattr(scene_selection._CLIENTS, "by_url"):
        del scene_selection._CLIENTS.by_url


class TestGetStacClient:
    """Each selection thread gets its own client, and reuses it."""

    def test_same_thread_reuses_one_client(self) -> None:
        with patch.object(
            scene_selection.pystac_client.Client, "open", side_effect=lambda url: MagicMock(url=url)
        ) as opened:
            first = scene_selection._get_stac_client("https://example.test/stac")
            second = scene_selection._get_stac_client("https://example.test/stac")

        assert first is second
        assert opened.call_count == 1

    def test_different_urls_get_different_clients(self) -> None:
        with patch.object(
            scene_selection.pystac_client.Client, "open", side_effect=lambda url: MagicMock(url=url)
        ):
            first = scene_selection._get_stac_client("https://example.test/a")
            second = scene_selection._get_stac_client("https://example.test/b")

        assert first is not second

    def test_threads_do_not_share_a_client(self) -> None:
        """A client owns an HTTP session and a plain resolved-object cache.

        Neither is built for concurrent use, so selection must not hand the same
        client to chips running side by side.
        """
        clients: list[object] = []
        clients_lock = threading.Lock()

        def grab() -> None:
            client = scene_selection._get_stac_client("https://example.test/stac")
            with clients_lock:
                clients.append(client)

        with patch.object(
            scene_selection.pystac_client.Client, "open", side_effect=lambda url: MagicMock(url=url)
        ):
            threads = [threading.Thread(target=grab) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

        assert len(clients) == 4
        assert len({id(client) for client in clients}) == 4


def _canned_item(item_id: str, dt) -> object:
    """A trusted-clear scene item: no COG reads needed to select it."""
    import pystac
    from pystac.extensions.eo import EOExtension

    item = pystac.Item(
        id=item_id,
        geometry={
            "type": "Polygon",
            "coordinates": [[[14.9, 45.9], [15.1, 45.9], [15.1, 46.1], [14.9, 46.1], [14.9, 45.9]]],
        },
        bbox=[14.9, 45.9, 15.1, 46.1],
        datetime=dt,
        properties={"eo:cloud_cover": 0.05, "s2:nodata_pixel_percentage": 0.0},
    )
    EOExtension.ext(item, add_if_missing=True)
    item.set_self_href(f"https://example.com/items/{item_id}")
    return item


class TestSearchBackendDispatch:
    """select_scenes_for_chip routes queries by search_backend."""

    BBOX = (14.9, 45.9, 15.1, 46.1)

    @pytest.fixture(autouse=True)
    def _crop_calendar(self, monkeypatch):
        from ftw_dataset_tools.api.imagery.crop_calendar import CropCalendarDates

        monkeypatch.setattr(
            scene_selection,
            "get_crop_calendar_dates",
            lambda _bbox, on_progress=None: CropCalendarDates(150, 270),  # noqa: ARG005
        )

    def test_parquet_backend_queries_the_mirror(self, monkeypatch):
        from datetime import UTC, datetime

        calls = []

        def fake_query_scenes(bbox, start, end, cloud_cover_max, **_kwargs):
            calls.append((bbox, start, end, cloud_cover_max))
            return [_canned_item("S2A_33TVM_fake", datetime(2021, 5, 30, 10, 0, tzinfo=UTC))]

        monkeypatch.setattr(scene_selection.parquet_search, "query_scenes", fake_query_scenes)
        result = scene_selection.select_scenes_for_chip(
            chip_id="chip_001",
            bbox=self.BBOX,
            year=2021,
            search_backend="parquet",
        )
        assert calls, "parquet backend was not queried"
        assert result.success
        assert result.selection_params["stac_host"] == "parquet-mirror"

    def test_earth_search_backend_uses_query_stac(self, monkeypatch):
        from datetime import UTC, datetime

        calls = []

        def fake_query_stac(**kwargs):
            calls.append(kwargs)
            return scene_selection.STACQueryResult(
                items=[_canned_item("S2A_33TVM_fake", datetime(2021, 5, 30, 10, 0, tzinfo=UTC))],
                catalog_url="https://earth-search.aws.element84.com/v1",
                collection="sentinel-2-c1-l2a",
                bbox=self.BBOX,
                date_range="2021-05-16/2021-06-13",
                cloud_cover_max=75,
            )

        monkeypatch.setattr(scene_selection, "_query_stac", fake_query_stac)
        result = scene_selection.select_scenes_for_chip(
            chip_id="chip_001",
            bbox=self.BBOX,
            year=2021,
            search_backend="earth-search",
        )
        assert calls, "earth-search backend was not queried"
        assert result.success
        assert result.selection_params["stac_host"] == "earthsearch"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="search_backend"):
            scene_selection.select_scenes_for_chip(
                chip_id="chip_001",
                bbox=self.BBOX,
                year=2021,
                search_backend="bogus",
            )

    @pytest.mark.parametrize(
        ("s2_collection", "expected"),
        [("c1", "sentinel-2-c1-l2a"), ("old-baseline", "sentinel-2-l2a")],
    )
    def test_parquet_backend_maps_s2_collection(self, monkeypatch, s2_collection, expected):
        from datetime import UTC, datetime

        seen = {}

        def fake_query_scenes(**kwargs):
            seen.update(kwargs)
            return [_canned_item("S2A_T33TVM_fake", datetime(2021, 5, 30, 10, 0, tzinfo=UTC))]

        monkeypatch.setattr(scene_selection.parquet_search, "query_scenes", fake_query_scenes)
        scene_selection.select_scenes_for_chip(
            chip_id="chip_001",
            bbox=self.BBOX,
            year=2021,
            s2_collection=s2_collection,
            search_backend="parquet",
        )
        assert seen["collection"] == expected


class TestBufferExpansion:
    """Each season widens its own search window until it finds a scene."""

    BBOX = (14.9, 45.9, 15.1, 46.1)

    @pytest.fixture(autouse=True)
    def _crop_calendar(self, monkeypatch):
        from ftw_dataset_tools.api.imagery.crop_calendar import CropCalendarDates

        # Planting 2021-05-30, harvest 2021-09-27
        monkeypatch.setattr(
            scene_selection,
            "get_crop_calendar_dates",
            lambda _bbox, on_progress=None: CropCalendarDates(150, 270),  # noqa: ARG005
        )

    @staticmethod
    def _scene(scene_id: str, cloud_cover: float = 0.05):
        from datetime import UTC, datetime

        item = _canned_item(scene_id, datetime(2021, 6, 1, 10, 0, tzinfo=UTC))
        item.properties["eo:cloud_cover"] = cloud_cover
        return item

    @staticmethod
    def _backend(monkeypatch, scenes_for) -> list[tuple[str, int]]:
        """Answer each search with scenes_for(season, buffer_days); record the calls."""
        calls: list[tuple[str, int]] = []

        def fake_query_parquet(bbox, center_date, cloud_cover_max, buffer_days, _collection):
            season = "planting" if center_date.month < 7 else "harvest"
            calls.append((season, buffer_days))
            return scene_selection.STACQueryResult(
                items=scenes_for(season, buffer_days),
                catalog_url="mirror",
                collection="sentinel-2-c1-l2a",
                bbox=bbox,
                date_range="",
                cloud_cover_max=cloud_cover_max,
            )

        monkeypatch.setattr(scene_selection, "_query_parquet", fake_query_parquet)
        return calls

    def _select(self, **kwargs):
        return scene_selection.select_scenes_for_chip(
            chip_id="chip_001", bbox=self.BBOX, year=2021, buffer_days=14, **kwargs
        )

    def test_only_the_missing_season_expands(self, monkeypatch):
        cloudy = self._scene("harvest_cloudy", cloud_cover=50.0)

        def scenes_for(season, buffer_days):
            if season == "planting":
                return [self._scene("planting_clear")]
            if buffer_days < 28:
                return [cloudy]
            return [cloudy, self._scene("harvest_clear")]

        calls = self._backend(monkeypatch, scenes_for)
        result = self._select(num_buffer_expansions=3, buffer_expansion_size=14)

        assert result.success
        assert calls == [("planting", 14), ("harvest", 14), ("harvest", 28)]
        assert (result.planting_buffer_used, result.harvest_buffer_used) == (14, 28)
        assert result.expansions_performed == 1
        # The cloudy harvest scene came back twice but is only checked once
        assert result.candidates_checked == 3

    def test_gives_up_after_the_last_expansion(self, monkeypatch):
        def scenes_for(season, _buffer_days):
            return [self._scene("planting_clear")] if season == "planting" else []

        calls = self._backend(monkeypatch, scenes_for)
        result = self._select(num_buffer_expansions=2, buffer_expansion_size=14)

        assert not result.success
        assert result.skipped_reason == "No cloud-free harvest scene found"
        assert [buffer for season, buffer in calls if season == "harvest"] == [14, 28, 42]
        assert result.harvest_buffer_used == 42
        assert result.expansions_performed == 2

    def test_planting_query_error_stops_selection(self, monkeypatch):
        def scenes_for(season, buffer_days):
            if season == "planting" and buffer_days > 14:
                raise ValueError("Query date range extends into the future")
            return [] if season == "planting" else [self._scene("harvest_clear")]

        self._backend(monkeypatch, scenes_for)
        result = self._select(num_buffer_expansions=3, buffer_expansion_size=14)

        assert result.skipped_reason == (
            "Planting query error: Query date range extends into the future"
        )
        assert result.planting_scene is None
        assert result.planting_buffer_used == 28

    def test_harvest_query_error_keeps_the_planting_scene(self, monkeypatch):
        def scenes_for(season, _buffer_days):
            if season == "harvest":
                raise ValueError("boom")
            return [self._scene("planting_clear")]

        self._backend(monkeypatch, scenes_for)
        result = self._select()

        assert result.skipped_reason == "Harvest query error: boom"
        assert result.planting_scene is not None
        assert result.planting_scene.id == "planting_clear"
        assert result.candidates_checked == 1


class TestSelectBestSceneNodata:
    """Scene-level nodata metadata must not reject a chip whose window is clean.

    Old-baseline products carry granule-edge nodata in
    s2:nodata_pixel_percentage even when a chip's window is untouched, so a
    non-zero scene value only means the chip's pixels must be checked.
    """

    BBOX = (14.9, 45.9, 15.1, 46.1)

    def _item(self, nodata_pct):
        from datetime import UTC, datetime

        item = _canned_item("S2A_33TVM_20210605_0_L2A", datetime(2021, 6, 5, 10, 0, tzinfo=UTC))
        item.properties["s2:nodata_pixel_percentage"] = nodata_pct
        import pystac

        item.add_asset("nir", pystac.Asset(href="https://example.com/B08.tif"))
        return item

    def test_zero_scene_nodata_accepts_without_pixel_check(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            scene_selection,
            "calculate_nodata_percentage",
            lambda href, _bbox: called.append(href) or 0.0,
        )
        scene = scene_selection._select_best_scene(
            [self._item(0.0)], season="planting", bbox=self.BBOX
        )
        assert scene is not None
        assert not called

    def test_scene_nodata_triggers_chip_pixel_check_and_accepts_clean_window(self, monkeypatch):
        """A scene with edge nodata is kept when the chip's own window has none."""
        monkeypatch.setattr(
            scene_selection, "calculate_nodata_percentage", lambda _href, _bbox: 0.0
        )
        scene = scene_selection._select_best_scene(
            [self._item(18.3)], season="planting", bbox=self.BBOX
        )
        assert scene is not None

    def test_chip_window_nodata_still_rejects(self, monkeypatch):
        monkeypatch.setattr(
            scene_selection, "calculate_nodata_percentage", lambda _href, _bbox: 42.0
        )
        scene = scene_selection._select_best_scene(
            [self._item(18.3)], season="planting", bbox=self.BBOX
        )
        assert scene is None

    @staticmethod
    def _failing_check(_href, _bbox):
        raise OSError("object not found")

    def test_failed_pixel_check_falls_back_to_scene_nodata(self, monkeypatch):
        """A scene whose metadata reports nodata is rejected when the pixel check fails."""
        monkeypatch.setattr(scene_selection, "calculate_nodata_percentage", self._failing_check)
        scene = scene_selection._select_best_scene(
            [self._item(18.3)], season="planting", bbox=self.BBOX
        )
        assert scene is None

    def test_failed_pixel_check_without_scene_nodata_continues(self, monkeypatch):
        """With no scene-level value to fall back on, a failed check does not reject."""
        monkeypatch.setattr(scene_selection, "calculate_nodata_percentage", self._failing_check)
        scene = scene_selection._select_best_scene(
            [self._item(None)], season="planting", bbox=self.BBOX
        )
        assert scene is not None
