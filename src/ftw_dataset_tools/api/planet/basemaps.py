"""Discover available Planet mosaics and cache each required quad once."""

from pathlib import Path

from shapely.geometry import shape

from ftw_dataset_tools.api.planet.http import PlanetClient, PlanetError
from ftw_dataset_tools.api.planet.raster import validate_raster
from ftw_dataset_tools.api.planet.state import safe_id

BASE_URL = "https://api.planet.com/basemaps/v1/mosaics"


def resolve_mosaic(client: PlanetClient, name: str) -> dict:
    """Resolve an exact mosaic name; never silently substitute another period/product."""
    result = client.request("GET", BASE_URL, params={"name__is": name})
    matches = [row for row in result["mosaics"] if row["name"] == name]
    if len(matches) != 1:
        raise PlanetError(f"Mosaic {name} is missing or ambiguous")
    mosaic = matches[0]
    safe_id(mosaic["id"])
    if not mosaic.get("quad_download", False):
        raise PlanetError(f"Quad downloads are unavailable for {name}")
    return mosaic


def list_quads(client: PlanetClient, mosaic_id: str, geometry: dict) -> list[str]:
    """Query actual availability for this mosaic and AOI, following every page."""
    bounds = shape(geometry).bounds
    page = client.request(
        "GET",
        f"{BASE_URL}/{safe_id(mosaic_id)}/quads",
        params={"bbox": ",".join(str(value) for value in bounds)},
    )
    identifiers = set()
    visited = set()
    while True:
        for item in page["items"]:
            if item.get("_links", {}).get("download"):
                identifiers.add(safe_id(item["id"]))
        next_url = page.get("_links", {}).get("_next")
        if not next_url:
            return sorted(identifiers)
        if next_url in visited:
            raise PlanetError("Basemaps API repeated a pagination link")
        visited.add(next_url)
        page = client.request("GET", next_url)


def quad_path(workdir: Path, mosaic_id: str, quad_id: str) -> Path:
    return workdir / "quads" / safe_id(mosaic_id) / f"{safe_id(quad_id)}.tif"


def download_quad(client: PlanetClient, workdir: Path, task: dict) -> dict:
    """Validate cached quads; atomically repair missing or corrupt files."""
    path = quad_path(workdir, task["mosaic_id"], task["quad_id"])
    try:
        validate_raster(path)
    except (OSError, ValueError):
        # /full resolves a fresh signed URL on every attempt. No expiring links in plans.
        url = f"{BASE_URL}/{safe_id(task['mosaic_id'])}/quads/{safe_id(task['quad_id'])}/full"
        client.download(url, path, validate_raster)
    return {"id": task["id"], "status": "ok"}
