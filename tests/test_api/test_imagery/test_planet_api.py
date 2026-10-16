"""Tests for the Planet Data and Orders API client (no network)."""

from __future__ import annotations

import base64
import io
import json
import urllib.error
from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from ftw_dataset_tools.api.imagery import planet_api
from ftw_dataset_tools.api.imagery.planet_api import (
    PlanetAPIError,
    PlanetUnavailableError,
    chip_clear_percent,
    create_order,
    download_order,
    search_scenes,
)

if TYPE_CHECKING:
    from pathlib import Path

BBOX = (6.05, 49.60, 6.078, 49.618)


class _Response(io.BytesIO):
    status = 200

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class FakePlanet:
    """Stands in for urlopen: records requests and replies from a queue per URL prefix."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict | None, dict]] = []
        self.replies: list[tuple[str, object]] = []

    def reply(self, url_prefix: str, payload: object) -> None:
        self.replies.append((url_prefix, payload))

    def __call__(self, request: object, timeout: float = 0) -> _Response:  # noqa: ARG002
        url = request.full_url  # type: ignore[attr-defined]
        body = json.loads(request.data) if request.data else None  # type: ignore[attr-defined]
        self.requests.append((request.get_method(), url, body, dict(request.header_items())))  # type: ignore[attr-defined]
        for i, (prefix, payload) in enumerate(self.replies):
            if url.startswith(prefix):
                del self.replies[i]
                if isinstance(payload, Exception):
                    raise payload
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                return _Response(data)
        raise AssertionError(f"unexpected request {url}")


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakePlanet:
    monkeypatch.setenv("PL_API_KEY", "test-key")
    monkeypatch.setattr(planet_api.time, "sleep", lambda _s: None)
    fake = FakePlanet()
    monkeypatch.setattr(planet_api.urllib.request, "urlopen", fake)
    return fake


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(b"detail"))  # type: ignore[arg-type]


class TestAuth:
    def test_missing_key_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PL_API_KEY", raising=False)
        with pytest.raises(PlanetUnavailableError, match="PL_API_KEY"):
            planet_api.require_api_key()

    def test_requests_use_basic_auth_with_the_key(self, fake: FakePlanet) -> None:
        fake.reply(planet_api.DATA_URL, {"features": []})
        search_scenes(BBOX, datetime(2024, 6, 1), datetime(2024, 6, 21), cloud_cover_max=20)
        headers = fake.requests[0][3]
        assert headers["Authorization"] == "Basic " + base64.b64encode(b"test-key:").decode()


class TestSearch:
    def test_follows_pagination_and_filters(self, fake: FakePlanet) -> None:
        next_url = f"{planet_api.DATA_URL}/searches/abc/results?_page=2"
        fake.reply(
            f"{planet_api.DATA_URL}/quick-search",
            {"features": [{"id": "a"}], "_links": {"_next": next_url}},
        )
        fake.reply(next_url, {"features": [{"id": "b"}], "_links": {}})

        features = search_scenes(
            BBOX, datetime(2024, 6, 1), datetime(2024, 6, 21), cloud_cover_max=20
        )

        assert [f["id"] for f in features] == ["a", "b"]
        method, _url, body, _ = fake.requests[0]
        assert method == "POST"
        assert body["item_types"] == ["PSScene"]
        filters = {f["type"]: f for f in body["filter"]["config"]}
        assert filters["AssetFilter"]["config"] == ["ortho_analytic_4b_sr", "ortho_udm2"]
        assert filters["RangeFilter"]["config"] == {"lte": 0.2}
        assert fake.requests[1][0] == "GET"

    def test_rate_limit_is_retried(self, fake: FakePlanet) -> None:
        fake.reply(planet_api.DATA_URL, _http_error(429))
        fake.reply(planet_api.DATA_URL, {"features": [{"id": "a"}]})

        features = search_scenes(
            BBOX, datetime(2024, 6, 1), datetime(2024, 6, 21), cloud_cover_max=20
        )

        assert [f["id"] for f in features] == ["a"]

    def test_client_errors_are_not_retried(self, fake: FakePlanet) -> None:
        fake.reply(planet_api.DATA_URL, _http_error(400))

        with pytest.raises(PlanetAPIError) as err:
            search_scenes(BBOX, datetime(2024, 6, 1), datetime(2024, 6, 21), cloud_cover_max=20)

        assert err.value.status == 400
        assert len(fake.requests) == 1


class TestCoverage:
    def test_polls_until_complete(self, fake: FakePlanet) -> None:
        coverage = f"{planet_api.DATA_URL}/item-types/PSScene/items/s1/coverage"
        fake.reply(coverage, {"status": "activating"})
        fake.reply(coverage, {"status": "complete", "clear_percent": 87})

        assert chip_clear_percent("s1", BBOX) == 87.0
        assert "mode=UDM2" in fake.requests[0][1]

    def test_gives_up_after_polling(
        self, fake: FakePlanet, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(planet_api, "_COVERAGE_POLLS", 2)
        coverage = f"{planet_api.DATA_URL}/item-types/PSScene/items/s1/coverage"
        fake.reply(coverage, {"status": "activating"})
        fake.reply(coverage, {"status": "activating"})

        with pytest.raises(PlanetAPIError, match="did not complete"):
            chip_clear_percent("s1", BBOX)


class TestOrders:
    def test_order_is_clipped_sr_without_harmonize_by_default(self, fake: FakePlanet) -> None:
        fake.reply(planet_api.ORDERS_URL, {"id": "order-1"})

        assert create_order("chip_planting", ["s1"], BBOX) == "order-1"

        body = fake.requests[0][2]
        assert body["products"] == [
            {"item_ids": ["s1"], "item_type": "PSScene", "product_bundle": "analytic_sr_udm2"}
        ]
        assert [next(iter(tool)) for tool in body["tools"]] == ["clip"]

    def test_harmonize_is_opt_in(self, fake: FakePlanet) -> None:
        fake.reply(planet_api.ORDERS_URL, {"id": "order-1"})

        create_order("chip_planting", ["s1"], BBOX, harmonize=True)

        tools = fake.requests[0][2]["tools"]
        assert {"harmonize": {"target_sensor": "Sentinel-2"}} in tools

    def test_download_writes_each_result(self, fake: FakePlanet, tmp_path: Path) -> None:
        fake.reply("https://dl/a", b"image-bytes")
        fake.reply("https://dl/b", b"udm-bytes")
        order = {
            "_links": {
                "results": [
                    {
                        "name": "order-1/PSScene/s1_AnalyticMS_SR_clip.tif",
                        "location": "https://dl/a",
                    },
                    {"name": "order-1/PSScene/s1_udm2_clip.tif", "location": "https://dl/b"},
                ]
            }
        }

        paths = download_order(order, tmp_path / "order-1")

        assert [p.name for p in paths] == ["s1_AnalyticMS_SR_clip.tif", "s1_udm2_clip.tif"]
        assert paths[0].read_bytes() == b"image-bytes"
