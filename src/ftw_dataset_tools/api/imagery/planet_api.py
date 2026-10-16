"""Planet Data and Orders API calls for PlanetScope (PSScene) imagery.

Plain HTTP through urllib, like the rest of ftwd, so the planetscope source needs no
extra dependency. Authentication is the account's API key in ``PL_API_KEY``.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shapely.geometry import box

from ftw_dataset_tools import __version__
from ftw_dataset_tools.api.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

logger = get_logger(__name__)

__all__ = [
    "API_KEY_ENV",
    "DATA_URL",
    "ITEM_TYPE",
    "ORDERS_URL",
    "SCENE_ASSET",
    "PlanetAPIError",
    "PlanetUnavailableError",
    "chip_clear_percent",
    "create_order",
    "download_order",
    "get_order",
    "require_api_key",
    "search_scenes",
]

API_KEY_ENV = "PL_API_KEY"
DATA_URL = "https://api.planet.com/data/v1"
ORDERS_URL = "https://api.planet.com/compute/ops/orders/v2"
ITEM_TYPE = "PSScene"
#: 4-band (blue, green, red, NIR) surface reflectance, the product Fields of the Planet uses.
SCENE_ASSET = "ortho_analytic_4b_sr"
SCENE_BUNDLE = "analytic_sr_udm2"
#: Pixel size of PlanetScope ortho products, and of the masks and chips made for them.
GRID_RESOLUTION_M = 3.0

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_ATTEMPTS = 4
_COVERAGE_POLLS = 10


class PlanetUnavailableError(Exception):
    """The planetscope source cannot run here: no API key is configured."""


class PlanetAPIError(Exception):
    """A Planet API request failed."""

    def __init__(self, url: str, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.url = url
        self.status = status


def require_api_key() -> str:
    """The Planet API key, or PlanetUnavailableError when it is not set."""
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise PlanetUnavailableError(f"Set {API_KEY_ENV} to use the planetscope source.")
    return key


def _headers() -> dict[str, str]:
    token = base64.b64encode(f"{require_api_key()}:".encode()).decode()
    return {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
        "User-Agent": f"ftw-dataset-tools/{__version__}",
    }


def _request(method: str, url: str, body: dict | None = None) -> Any:
    """Send one JSON request, retrying rate limits and server errors with backoff."""
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(1, _ATTEMPTS + 1):
        request = urllib.request.Request(url, data=data, method=method, headers=_headers())
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
                return json.loads(payload) if payload else None
        except urllib.error.HTTPError as err:
            if err.code in _RETRY_STATUSES and attempt < _ATTEMPTS:
                time.sleep(2**attempt)
                continue
            detail = err.read().decode(errors="replace")[:300]
            raise PlanetAPIError(url, f"{method} {url} -> {err.code}: {detail}", err.code) from err
        except urllib.error.URLError as err:
            if attempt < _ATTEMPTS:
                time.sleep(2**attempt)
                continue
            raise PlanetAPIError(url, f"{method} {url} failed: {err.reason}") from err
    raise AssertionError("unreachable")  # pragma: no cover


def _bbox_geometry(bbox: tuple[float, float, float, float], pad: float = 0.0) -> dict:
    minx, miny, maxx, maxy = bbox
    return box(minx - pad, miny - pad, maxx + pad, maxy + pad).__geo_interface__


def _search_filter(
    bbox: tuple[float, float, float, float],
    start: datetime,
    end: datetime,
    cloud_cover_max: float,
) -> dict:
    return {
        "type": "AndFilter",
        "config": [
            {"type": "GeometryFilter", "field_name": "geometry", "config": _bbox_geometry(bbox)},
            {
                "type": "DateRangeFilter",
                "field_name": "acquired",
                "config": {
                    "gte": f"{start:%Y-%m-%dT%H:%M:%SZ}",
                    "lte": f"{end:%Y-%m-%dT%H:%M:%SZ}",
                },
            },
            {
                "type": "RangeFilter",
                "field_name": "cloud_cover",
                "config": {"lte": cloud_cover_max / 100},
            },
            {"type": "StringInFilter", "field_name": "quality_category", "config": ["standard"]},
            {"type": "StringInFilter", "field_name": "publishing_stage", "config": ["finalized"]},
            {"type": "AssetFilter", "config": [SCENE_ASSET, "ortho_udm2"]},
            {"type": "PermissionFilter", "config": ["assets:download"]},
        ],
    }


def search_scenes(
    bbox: tuple[float, float, float, float],
    start: datetime,
    end: datetime,
    *,
    cloud_cover_max: float,
) -> list[dict]:
    """PSScene features over ``bbox`` acquired in [start, end], as Data API GeoJSON features."""
    url = f"{DATA_URL}/quick-search?_page_size=250"
    body: dict | None = {
        "item_types": [ITEM_TYPE],
        "filter": _search_filter(bbox, start, end, cloud_cover_max),
    }
    features: list[dict] = []
    while url:
        page = _request("POST" if body else "GET", url, body)
        features.extend(page.get("features", []))
        url = page.get("_links", {}).get("_next")
        body = None
    return features


def chip_clear_percent(item_id: str, bbox: tuple[float, float, float, float]) -> float:
    """Percent of the chip the scene's UDM2 marks clear, from the Data API coverage endpoint.

    The endpoint computes coverage asynchronously, so it is polled until complete.
    """
    query = urllib.parse.urlencode({"mode": "UDM2", "band": "clear"})
    url = f"{DATA_URL}/item-types/{ITEM_TYPE}/items/{item_id}/coverage?{query}"
    for _ in range(_COVERAGE_POLLS):
        result = _request("POST", url, {"geometry": _bbox_geometry(bbox)})
        if result.get("status") == "complete":
            return float(result["clear_percent"])
        time.sleep(3)
    raise PlanetAPIError(url, f"UDM2 coverage for {item_id} did not complete")


def create_order(
    name: str,
    item_ids: Iterable[str],
    *,
    aoi: dict,
    harmonize: bool = False,
) -> str:
    """Submit an order for ``item_ids`` clipped to the GeoJSON ``aoi`` and return its id.

    The bundle is 4-band surface reflectance plus UDM2. Scenes stay plain surface
    reflectance by default; ``harmonize`` asks Planet to adjust them to Sentinel-2
    radiometry instead.
    """
    tools: list[dict] = [{"clip": {"aoi": aoi}}]
    if harmonize:
        tools.append({"harmonize": {"target_sensor": "Sentinel-2"}})
    order = _request(
        "POST",
        ORDERS_URL,
        {
            "name": name,
            "products": [
                {
                    "item_ids": list(item_ids),
                    "item_type": ITEM_TYPE,
                    "product_bundle": SCENE_BUNDLE,
                }
            ],
            "tools": tools,
        },
    )
    return order["id"]


def get_order(order_id: str) -> dict:
    """The order's current record, including ``state`` and, once done, result links."""
    return _request("GET", f"{ORDERS_URL}/{order_id}")


def download_order(order: dict, directory: Path) -> list[Path]:
    """Download every result file of a finished order into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for result in order.get("_links", {}).get("results", []):
        # Result names are "<order id>/<item type>/<file>"; keep just the file.
        path = directory / Path(result["name"]).name
        request = urllib.request.Request(
            result["location"], headers={"User-Agent": f"ftw-dataset-tools/{__version__}"}
        )
        with urllib.request.urlopen(request, timeout=300) as response, path.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        paths.append(path)
    return paths
