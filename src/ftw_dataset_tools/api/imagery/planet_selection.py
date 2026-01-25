"""Planet scene selection from STAC API based on crop calendar dates.

This module provides scene selection for PlanetScope imagery, following the same
pattern as the Sentinel-2 scene_selection.py module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

import httpx

from ftw_dataset_tools.api.imagery.crop_calendar import (
    CropCalendarDates,
    get_crop_calendar_dates,
)
from ftw_dataset_tools.api.imagery.planet_client import (
    DEFAULT_BUFFER_DAYS,
    DEFAULT_NUM_ITERATIONS,
    PLANET_TILES_URL,
    PlanetClient,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pystac

__all__ = [
    "PlanetScene",
    "PlanetSelectionResult",
    "generate_thumbnail",
    "get_clear_coverage",
    "select_planet_scenes_for_chip",
]

# Default cloud cover threshold (same as S2)
DEFAULT_CLOUD_COVER_CHIP = 2.0

# PSScene collection name
PSSCENE_COLLECTION = "PSScene"


@dataclass
class PlanetScene:
    """A selected PlanetScope scene. Wraps pystac.Item like S2's SelectedScene."""

    item: pystac.Item
    season: Literal["planting", "harvest"]
    clear_coverage: float  # 0-100, from coverage API
    datetime: datetime
    stac_url: str

    @property
    def id(self) -> str:
        """Scene ID."""
        return self.item.id

    @property
    def cloud_cover(self) -> float:
        """Cloud cover percentage (100 - clear_coverage)."""
        return 100.0 - self.clear_coverage


@dataclass
class PlanetSelectionResult:
    """Result of Planet imagery selection for a chip. Mirrors S2's SceneSelectionResult."""

    chip_id: str
    bbox: tuple[float, float, float, float]
    year: int
    crop_calendar: CropCalendarDates
    planting_scene: PlanetScene | None = None
    harvest_scene: PlanetScene | None = None
    skipped_reason: str | None = None
    candidates_checked: int = 0
    iterations_used: int = 0
    selection_params: dict = field(default_factory=dict)
    planting_buffer_used: int = 0
    harvest_buffer_used: int = 0

    @property
    def success(self) -> bool:
        """Whether both scenes were successfully selected."""
        return self.planting_scene is not None and self.harvest_scene is not None


def _format_date_range(center_date: datetime, buffer_days: int) -> str:
    """Format date range for STAC API queries."""
    start = (center_date - timedelta(days=buffer_days)).strftime("%Y-%m-%dT00:00:00Z")
    end = (center_date + timedelta(days=buffer_days)).strftime("%Y-%m-%dT23:59:59Z")
    return f"{start}/{end}"


def _validate_date_not_future(center_date: datetime, buffer_days: int) -> None:
    """Validate that query dates are not in the future."""
    end_date = center_date + timedelta(days=buffer_days)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    if end_date > today:
        raise ValueError(
            f"Query date range extends into the future. "
            f"Center date {center_date.date()} + {buffer_days} buffer days = {end_date.date()}. "
            f"Try using an earlier calendar year."
        )


def _get_item_datetime(item: pystac.Item) -> datetime:
    """Get datetime from STAC item, ensuring timezone-aware."""
    if item.datetime is not None:
        dt = item.datetime
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt

    # Fall back to properties
    datetime_prop = item.properties.get("datetime")
    if datetime_prop is not None:
        if isinstance(datetime_prop, str):
            normalized = datetime_prop.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        if isinstance(datetime_prop, datetime):
            if datetime_prop.tzinfo is None:
                return datetime_prop.replace(tzinfo=UTC)
            return datetime_prop

    raise ValueError(f"STAC item {item.id} has no datetime")


def _bbox_to_geometry(bbox: tuple[float, float, float, float]) -> dict:
    """Convert bbox to GeoJSON Polygon geometry."""
    minx, miny, maxx, maxy = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [minx, miny],
                [maxx, miny],
                [maxx, maxy],
                [minx, maxy],
                [minx, miny],
            ]
        ],
    }


def get_clear_coverage(
    client: PlanetClient,
    item_id: str,
    geometry: dict,
) -> float:
    """Get clear coverage percentage using Planet's estimate API.

    Falls back to item's cloud_cover property on API error.

    Args:
        client: Authenticated Planet client
        item_id: Planet scene ID
        geometry: GeoJSON geometry for the AOI

    Returns:
        Clear coverage percentage (0-100)
    """
    # Planet Data API v2 endpoint for clear coverage estimate
    url = f"https://api.planet.com/data/v1/item-types/PSScene/items/{item_id}/coverage"

    try:
        response = httpx.post(
            url,
            auth=(client.api_key, ""),
            json={"geometry": geometry},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        # The API returns clear_percent as 0-1, convert to 0-100
        return data.get("clear_percent", 0.0) * 100.0
    except Exception:
        # Fall back to cloud_cover from item if available
        return 0.0  # Conservative fallback


def generate_thumbnail(
    client: PlanetClient,
    item_id: str,
    output_path: Path,
    width: int = 256,
) -> Path | None:
    """Generate thumbnail via Planet tiles endpoint.

    Args:
        client: Authenticated Planet client
        item_id: Planet scene ID
        output_path: Path to save the thumbnail
        width: Thumbnail width in pixels

    Returns:
        Path to saved thumbnail, or None on failure
    """
    # Planet Tiles API for thumbnails
    url = f"{PLANET_TILES_URL}PSScene/{item_id}/thumb"

    try:
        response = httpx.get(
            url,
            params={"width": width},
            auth=(client.api_key, ""),
            timeout=30.0,
        )
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        return output_path
    except Exception:
        return None


def _query_planet_stac(
    client: PlanetClient,
    bbox: tuple[float, float, float, float],
    center_date: datetime,
    buffer_days: int,
    cloud_cover_max: float = 75.0,
) -> list[pystac.Item]:
    """Query Planet STAC API for PSScene items.

    Args:
        client: Authenticated Planet client
        bbox: Bounding box (minx, miny, maxx, maxy)
        center_date: Center date for search
        buffer_days: Days to search around center_date
        cloud_cover_max: Maximum scene-level cloud cover filter

    Returns:
        List of STAC items sorted by cloud cover ascending
    """
    _validate_date_not_future(center_date, buffer_days)

    date_range = _format_date_range(center_date, buffer_days)
    stac_client = client.get_stac_client()

    search = stac_client.search(
        collections=[PSSCENE_COLLECTION],
        bbox=list(bbox),
        datetime=date_range,
        query={"eo:cloud_cover": {"lte": cloud_cover_max}},
    )

    items = list(search.items())

    # Sort by cloud cover ascending
    items.sort(key=lambda item: item.properties.get("eo:cloud_cover", 100))

    return items


def _select_best_planet_scene(
    client: PlanetClient,
    items: list[pystac.Item],
    season: Literal["planting", "harvest"],
    geometry: dict,
    cloud_cover_chip: float = DEFAULT_CLOUD_COVER_CHIP,
    on_progress: Callable[[str], None] | None = None,
) -> PlanetScene | None:
    """Select the best Planet scene from candidates.

    Args:
        client: Authenticated Planet client
        items: List of candidate STAC items (pre-sorted by cloud cover)
        season: Season identifier
        geometry: GeoJSON geometry for coverage calculation
        cloud_cover_chip: Maximum chip-level cloud cover percentage
        on_progress: Optional callback for progress messages

    Returns:
        PlanetScene or None if no suitable scene found
    """
    if not items:
        return None

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    for item in items:
        item_id = item.id
        scene_cloud_cover = item.properties.get("eo:cloud_cover", 100.0)

        # Get clear coverage from Planet API
        clear_coverage = get_clear_coverage(client, item_id, geometry)

        # If coverage API fails, estimate from cloud_cover
        if clear_coverage == 0.0 and scene_cloud_cover < 100:
            clear_coverage = 100.0 - scene_cloud_cover
            log(f"  {item_id}: using scene cloud cover ({scene_cloud_cover:.1f}%)")
        else:
            log(f"  {item_id}: scene {scene_cloud_cover:.1f}% -> chip {100 - clear_coverage:.1f}%")

        # Check if chip cloud cover exceeds threshold
        chip_cloud_cover = 100.0 - clear_coverage
        if chip_cloud_cover > cloud_cover_chip:
            log(f"  Skipping {item_id}: {chip_cloud_cover:.1f}% cloud")
            continue

        # Get scene datetime
        try:
            scene_dt = _get_item_datetime(item)
        except ValueError as e:
            log(f"  {item_id}: skipping - {e}")
            continue

        return PlanetScene(
            item=item,
            season=season,
            clear_coverage=clear_coverage,
            datetime=scene_dt,
            stac_url=item.get_self_href() or "",
        )

    return None


def select_planet_scenes_for_chip(
    client: PlanetClient,
    chip_id: str,
    bbox: tuple[float, float, float, float],
    year: int,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
    num_iterations: int = DEFAULT_NUM_ITERATIONS,
    cloud_cover_chip: float = DEFAULT_CLOUD_COVER_CHIP,
    on_progress: Callable[[str], None] | None = None,
) -> PlanetSelectionResult:
    """Select optimal Planet scenes for a chip based on crop calendar.

    Args:
        client: Authenticated Planet client
        chip_id: Chip identifier
        bbox: Bounding box (minx, miny, maxx, maxy) in EPSG:4326
        year: Calendar year for the crop cycle
        buffer_days: Initial days to search around crop calendar dates
        num_iterations: Number of buffer expansion iterations
        cloud_cover_chip: Maximum chip-level cloud cover percentage (0-100)
        on_progress: Optional callback for progress messages

    Returns:
        PlanetSelectionResult with selected scenes or skip reason
    """

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    # Get crop calendar dates
    try:
        crop_dates = get_crop_calendar_dates(bbox, on_progress=on_progress)
    except ValueError as e:
        return PlanetSelectionResult(
            chip_id=chip_id,
            bbox=bbox,
            year=year,
            crop_calendar=CropCalendarDates(0, 0),
            skipped_reason=f"Crop calendar error: {e}",
        )

    # Convert to datetime
    planting_dt, harvest_dt = crop_dates.to_datetime(year)

    log(f"Crop calendar: planting={planting_dt.date()}, harvest={harvest_dt.date()}")

    # Store selection parameters
    selection_params = {
        "stac_host": "planet",
        "cloud_cover_chip_threshold": cloud_cover_chip,
        "buffer_days": buffer_days,
        "num_iterations": num_iterations,
    }

    # Geometry for coverage API
    geometry = _bbox_to_geometry(bbox)

    # Helper to check if a scene meets threshold
    def _meets_threshold(scene: PlanetScene | None) -> bool:
        if scene is None:
            return False
        return scene.cloud_cover <= cloud_cover_chip

    # Initialize buffers and scenes
    planting_buffer = buffer_days
    harvest_buffer = buffer_days
    planting_scene: PlanetScene | None = None
    harvest_scene: PlanetScene | None = None
    candidates_checked = 0
    iterations_used = 0

    # Track already-checked scene IDs
    planting_checked_ids: set[str] = set()
    harvest_checked_ids: set[str] = set()

    # Iterative buffer expansion loop
    for iteration in range(num_iterations + 1):
        # Query and select planting scene if not yet found
        if not _meets_threshold(planting_scene):
            try:
                if iteration > 0:
                    log(f"Iteration {iteration}: planting buffer now {planting_buffer} days")
                log(f"Searching for planting scene around {planting_dt.date()}...")

                items = _query_planet_stac(
                    client=client,
                    bbox=bbox,
                    center_date=planting_dt,
                    buffer_days=planting_buffer,
                )

                # Filter already-checked scenes
                new_items = [item for item in items if item.id not in planting_checked_ids]
                candidates_checked += len(new_items)

                if iteration > 0 and len(items) > len(new_items):
                    log(f"Found {len(items)} total, {len(new_items)} new planting candidates")
                else:
                    log(f"Found {len(new_items)} planting scene candidates")

                for item in new_items[:5]:
                    cc = item.properties.get("eo:cloud_cover", 0.0)
                    log(f"  - {item.id}: {cc:.1f}% cloud, {item.datetime}")

                # Track checked IDs
                planting_checked_ids.update(item.id for item in new_items)

                planting_scene = _select_best_planet_scene(
                    client=client,
                    items=new_items,
                    season="planting",
                    geometry=geometry,
                    cloud_cover_chip=cloud_cover_chip,
                    on_progress=on_progress,
                )

                if planting_scene:
                    log(
                        f"Selected planting scene: {planting_scene.id} "
                        f"({planting_scene.cloud_cover:.1f}% cloud)"
                    )
            except ValueError as e:
                return PlanetSelectionResult(
                    chip_id=chip_id,
                    bbox=bbox,
                    year=year,
                    crop_calendar=crop_dates,
                    skipped_reason=f"Planting query error: {e}",
                    selection_params=selection_params,
                    planting_buffer_used=planting_buffer,
                    harvest_buffer_used=harvest_buffer,
                    iterations_used=iterations_used,
                )

        # Query and select harvest scene if not yet found
        if not _meets_threshold(harvest_scene):
            try:
                if iteration > 0:
                    log(f"Iteration {iteration}: harvest buffer now {harvest_buffer} days")
                log(f"Searching for harvest scene around {harvest_dt.date()}...")

                items = _query_planet_stac(
                    client=client,
                    bbox=bbox,
                    center_date=harvest_dt,
                    buffer_days=harvest_buffer,
                )

                # Filter already-checked scenes
                new_items = [item for item in items if item.id not in harvest_checked_ids]
                candidates_checked += len(new_items)

                if iteration > 0 and len(items) > len(new_items):
                    log(f"Found {len(items)} total, {len(new_items)} new harvest candidates")
                else:
                    log(f"Found {len(new_items)} harvest scene candidates")

                for item in new_items[:5]:
                    cc = item.properties.get("eo:cloud_cover", 0.0)
                    log(f"  - {item.id}: {cc:.1f}% cloud, {item.datetime}")

                # Track checked IDs
                harvest_checked_ids.update(item.id for item in new_items)

                harvest_scene = _select_best_planet_scene(
                    client=client,
                    items=new_items,
                    season="harvest",
                    geometry=geometry,
                    cloud_cover_chip=cloud_cover_chip,
                    on_progress=on_progress,
                )

                if harvest_scene:
                    log(
                        f"Selected harvest scene: {harvest_scene.id} "
                        f"({harvest_scene.cloud_cover:.1f}% cloud)"
                    )
            except ValueError as e:
                return PlanetSelectionResult(
                    chip_id=chip_id,
                    bbox=bbox,
                    year=year,
                    crop_calendar=crop_dates,
                    planting_scene=planting_scene,
                    skipped_reason=f"Harvest query error: {e}",
                    candidates_checked=candidates_checked,
                    selection_params=selection_params,
                    planting_buffer_used=planting_buffer,
                    harvest_buffer_used=harvest_buffer,
                    iterations_used=iterations_used,
                )

        # Check if both seasons meet threshold
        planting_ok = _meets_threshold(planting_scene)
        harvest_ok = _meets_threshold(harvest_scene)

        if planting_ok and harvest_ok:
            log(f"Both seasons meet threshold after {iteration} iteration(s)")
            break

        # Expand buffers for failing seasons
        if iteration < num_iterations:
            if not planting_ok:
                planting_buffer += buffer_days
            if not harvest_ok:
                harvest_buffer += buffer_days
            iterations_used = iteration + 1

    # Determine skip reason if any scene is missing
    skipped_reason = None
    if planting_scene is None and harvest_scene is None:
        skipped_reason = "No cloud-free scenes found for either season"
    elif planting_scene is None:
        skipped_reason = "No cloud-free planting scene found"
    elif harvest_scene is None:
        skipped_reason = "No cloud-free harvest scene found"

    return PlanetSelectionResult(
        chip_id=chip_id,
        bbox=bbox,
        year=year,
        crop_calendar=crop_dates,
        planting_scene=planting_scene,
        harvest_scene=harvest_scene,
        skipped_reason=skipped_reason,
        candidates_checked=candidates_checked,
        selection_params=selection_params,
        planting_buffer_used=planting_buffer,
        harvest_buffer_used=harvest_buffer,
        iterations_used=iterations_used,
    )
