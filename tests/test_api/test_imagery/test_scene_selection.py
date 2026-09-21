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
