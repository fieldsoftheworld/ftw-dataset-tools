"""PlanetScope discovery and bounded asset activation through the Data API."""

import time
from datetime import UTC, datetime

from pyproj import Geod
from shapely.geometry import shape
from shapely.geometry.polygon import orient

from ftw_dataset_tools.api.planet.http import PlanetClient, PlanetError
from ftw_dataset_tools.api.planet.state import safe_id

DATA_URL = "https://api.planet.com/data/v1"
ASSETS = ("ortho_analytic_4b_sr", "ortho_udm2")


def utc(value: str) -> datetime:
    """Interpret date-only/naive input as UTC, and normalize explicit offsets."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def validate_dates(row: dict) -> None:
    """Require an explicit half-open search interval and target inside it."""
    if not utc(row["start"]) <= utc(row["target_date"]) < utc(row["end"]):
        raise ValueError(f"Require start <= target_date < end for {row['id']}")


def geodesic_area(geometry: object) -> float:
    if geometry.is_empty:
        return 0.0
    if geometry.geom_type in {"GeometryCollection", "MultiPolygon"}:
        return sum(geodesic_area(part) for part in geometry.geoms)
    if geometry.geom_type != "Polygon":
        return 0.0
    return abs(Geod(ellps="WGS84").geometry_area_perimeter(orient(geometry))[0])


def search_candidates(
    client: PlanetClient, row: dict, max_cloud: float, min_coverage: float, max_candidates: int
) -> list[dict]:
    """Search all pages, rank by date/cloud/id, retain a bounded alternate list."""
    filters = [
        {"type": "GeometryFilter", "field_name": "geometry", "config": row["geometry"]},
        {
            "type": "DateRangeFilter",
            "field_name": "acquired",
            "config": {"gte": utc(row["start"]).isoformat(), "lt": utc(row["end"]).isoformat()},
        },
        {"type": "RangeFilter", "field_name": "cloud_cover", "config": {"lte": max_cloud}},
        {"type": "StringInFilter", "field_name": "quality_category", "config": ["standard"]},
        {"type": "PermissionFilter", "config": ["assets:download"]},
        *({"type": "AssetFilter", "config": [asset]} for asset in ASSETS),
    ]
    page = client.request(
        "POST",
        f"{DATA_URL}/quick-search",
        json={"item_types": ["PSScene"], "filter": {"type": "AndFilter", "config": filters}},
    )
    polygon = shape(row["geometry"])
    total_area = geodesic_area(polygon)
    target = utc(row["target_date"])
    candidates = {}
    visited = set()
    while True:
        for item in page["features"]:
            coverage = geodesic_area(polygon.intersection(shape(item["geometry"]))) / total_area
            if coverage < min_coverage:
                continue
            props = item["properties"]
            identifier = safe_id(item["id"])
            candidates[identifier] = {
                "item_id": identifier,
                "acquired": props["acquired"],
                "cloud_cover": props.get("cloud_cover", 1),
                "coverage": coverage,
            }
        url = page.get("_links", {}).get("_next")
        if not url:
            break
        if url in visited:
            raise PlanetError("Data API repeated a pagination link")
        visited.add(url)
        page = client.request("GET", url)
    return sorted(
        candidates.values(),
        key=lambda candidate: (
            abs((utc(candidate["acquired"]) - target).total_seconds()),
            candidate["cloud_cover"],
            candidate["item_id"],
        ),
    )[:max_candidates]


def asset_url(
    client: PlanetClient,
    item_id: str,
    asset_type: str,
    *,
    timeout: float = 3600,
    poll_interval: float = 10,
    refresh: bool = False,
) -> str:
    """Activate once, poll within a deadline, and return a fresh signed URL.

    ``refresh`` requests activation even for an active asset, for recovery from
    expired or broken URLs. URLs stay in memory and are never journaled.
    """
    url = f"{DATA_URL}/item-types/PSScene/items/{safe_id(item_id)}/assets"
    deadline = time.monotonic() + timeout
    requested = False
    while True:
        asset = client.request("GET", url).get(asset_type)
        if asset is None:
            raise PlanetError(f"Asset {asset_type} is unavailable")
        if asset.get("status") == "active" and asset.get("location") and not refresh:
            return asset["location"]
        if not requested:
            client.request("GET", asset["_links"]["activate"])
            requested = True
            refresh = False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PlanetError(f"Activation timed out for {asset_type}")
        time.sleep(min(poll_interval, remaining))
