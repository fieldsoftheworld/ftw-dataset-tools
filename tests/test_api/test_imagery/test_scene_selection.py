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
