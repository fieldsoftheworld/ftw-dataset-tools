"""Scene selection from STAC catalogs based on crop calendar dates."""

from __future__ import annotations

import time
import urllib.error
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import pystac_client
from pystac.extensions.eo import EOExtension

if TYPE_CHECKING:
    import pystac

from ftw_dataset_tools.api.imagery.cloud_analysis import calculate_pixel_cloud_cover
from ftw_dataset_tools.api.imagery.crop_calendar import (
    CropCalendarDates,
    get_crop_calendar_dates,
)
from ftw_dataset_tools.api.imagery.nodata_analysis import (
    calculate_nodata_percentage,
    get_nodata_from_metadata,
)
from ftw_dataset_tools.api.imagery.settings import (
    DEFAULT_BUFFER_DAYS,
    DEFAULT_BUFFER_EXPANSION_SIZE,
    DEFAULT_CLOUD_COVER_CHIP,
    DEFAULT_CLOUD_COVER_SCENE,
    DEFAULT_NODATA_MAX,
    DEFAULT_NUM_BUFFER_EXPANSIONS,
    PIXEL_CHECK_SKIP_THRESHOLD,
    S2_COLLECTIONS,
    STAC_URL,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "STACQueryResult",
    "SceneSelectionResult",
    "SelectedScene",
    "select_scenes_for_chip",
]


@dataclass
class SelectedScene:
    """Information about a selected Sentinel-2 scene."""

    item: pystac.Item
    season: Literal["planting", "harvest"]
    cloud_cover: float
    datetime: datetime
    stac_url: str

    @property
    def id(self) -> str:
        """Scene ID."""
        return self.item.id

    def get_asset_href(self, band: str) -> str | None:
        """Get href for a specific band asset."""
        asset = self.item.assets.get(band)
        return asset.href if asset else None


@dataclass
class SceneSelectionResult:
    """Result of scene selection for a chip."""

    chip_id: str
    bbox: tuple[float, float, float, float]
    year: int
    crop_calendar: CropCalendarDates
    planting_scene: SelectedScene | None = None
    harvest_scene: SelectedScene | None = None
    skipped_reason: str | None = None
    candidates_checked: int = 0
    selection_params: dict = field(default_factory=dict)
    planting_buffer_used: int = 0
    harvest_buffer_used: int = 0
    expansions_performed: int = 0

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


@lru_cache(maxsize=4)
def _get_stac_client(catalog_url: str) -> pystac_client.Client:
    """Get cached STAC client for a catalog URL."""
    return pystac_client.Client.open(catalog_url)


@dataclass
class STACQueryResult:
    """Result of a STAC query with debug information."""

    items: list[pystac.Item]
    catalog_url: str
    collection: str
    bbox: tuple[float, float, float, float]
    date_range: str
    cloud_cover_max: int


def _query_stac(
    bbox: tuple[float, float, float, float],
    center_date: datetime,
    cloud_cover_max: int,
    buffer_days: int,
    s2_collection: str = "c1",
    max_retries: int = 3,
) -> STACQueryResult:
    """
    Query STAC catalog for Sentinel-2 scenes with retry logic.

    Args:
        bbox: Bounding box (minx, miny, maxx, maxy) in EPSG:4326
        center_date: Center date for the search
        cloud_cover_max: Maximum cloud cover percentage
        buffer_days: Days to search around center_date
        s2_collection: Sentinel-2 collection identifier ("c1" or "old-baseline")
        max_retries: Maximum number of retries for transient errors

    Returns:
        STACQueryResult with items and query details

    Raises:
        Exception: If all retries fail
    """
    _validate_date_not_future(center_date, buffer_days)

    date_range = _format_date_range(center_date, buffer_days)
    catalog_url = STAC_URL
    collection = S2_COLLECTIONS.get(s2_collection, "sentinel-2-c1-l2a")

    last_error = None
    for attempt in range(max_retries):
        try:
            # Use cached STAC client
            catalog = _get_stac_client(catalog_url)
            search = catalog.search(
                collections=[collection],
                bbox=list(bbox),
                datetime=date_range,
                query={"eo:cloud_cover": {"lt": cloud_cover_max}},
            )

            items = list(search.items())

            # Sort by cloud cover (ascending)
            items.sort(key=lambda item: EOExtension.ext(item).cloud_cover or 100)

            return STACQueryResult(
                items=items,
                catalog_url=catalog_url,
                collection=collection,
                bbox=bbox,
                date_range=date_range,
                cloud_cover_max=cloud_cover_max,
            )

        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            last_error = e
            # Check if it's a retryable error (502, 503, 504, connection issues)
            is_retryable = False
            if (isinstance(e, urllib.error.HTTPError) and e.code in (502, 503, 504)) or isinstance(
                e, (urllib.error.URLError, OSError)
            ):
                is_retryable = True

            if is_retryable and attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s
                wait_time = 2**attempt
                time.sleep(wait_time)
                continue
            raise

    # This should not be reached, but just in case
    if last_error:
        raise last_error
    return STACQueryResult(
        items=[],
        catalog_url=catalog_url,
        collection=collection,
        bbox=bbox,
        date_range=date_range,
        cloud_cover_max=cloud_cover_max,
    )


def _parse_iso_datetime(value: str) -> datetime:
    """Parse ISO datetime string to timezone-aware datetime."""

    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _get_item_datetime(item: pystac.Item) -> datetime:
    """
    Get datetime from STAC item, falling back to start_datetime if needed.

    Always returns a timezone-aware datetime.
    """

    if item.datetime is not None:
        dt = item.datetime
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt

    # Fall back to start_datetime
    start_dt = item.properties.get("start_datetime")
    if start_dt is None:
        # Fall back to datetime property
        datetime_prop = item.properties.get("datetime")
        if datetime_prop is not None:
            if isinstance(datetime_prop, str):
                return _parse_iso_datetime(datetime_prop)
            if isinstance(datetime_prop, datetime):
                if datetime_prop.tzinfo is None:
                    return datetime_prop.replace(tzinfo=UTC)
                return datetime_prop
        raise ValueError(f"STAC item {item.id} has no datetime or start_datetime")

    if isinstance(start_dt, str):
        return _parse_iso_datetime(start_dt)
    if isinstance(start_dt, datetime):
        if start_dt.tzinfo is None:
            return start_dt.replace(tzinfo=UTC)
        return start_dt

    raise ValueError(f"STAC item {item.id} has invalid start_datetime type: {type(start_dt)}")


def _short_date(item: pystac.Item) -> str:
    """Extract short date (M-DD) from item datetime or ID."""
    # Try datetime first
    dt = item.datetime
    if dt:
        return f"{dt.month}-{dt.day:02d}"
    # Fall back to parsing from ID like "S2A_T54TWN_20240928T012657_L2A"
    parts = item.id.split("_")
    if len(parts) >= 3:
        date_part = parts[2][:8]  # "20240928"
        if len(date_part) == 8 and date_part.isdigit():
            month = int(date_part[4:6])
            day = int(date_part[6:8])
            return f"{month}-{day:02d}"
    return item.id[:15]  # Fallback to truncated ID


def _select_best_scene(
    items: list[pystac.Item],
    season: str,
    bbox: tuple[float, float, float, float],
    cloud_cover_chip: float = DEFAULT_CLOUD_COVER_CHIP,
    nodata_max: float = DEFAULT_NODATA_MAX,
    on_progress: Callable[[str], None] | None = None,
) -> SelectedScene | None:
    """
    Select the best scene from candidates.

    Args:
        items: List of candidate STAC items (pre-sorted by cloud cover)
        season: Season identifier ("planting" or "harvest")
        bbox: Bounding box for chip-level cloud calculation
        cloud_cover_chip: Maximum chip-level cloud cover percentage (0-100)
        nodata_max: Maximum nodata percentage (0-100). Default 0 rejects any nodata.
        on_progress: Optional callback for progress messages

    Returns:
        SelectedScene or None if no suitable scene found
    """
    if not items:
        return None

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    for item in items:
        short_dt = _short_date(item)

        # Check nodata first (fail fast - uses metadata then overviews)
        nodata_pct = get_nodata_from_metadata(item)
        if nodata_pct is not None:
            # Use STAC metadata if available
            if nodata_pct > nodata_max:
                log(f"  Skipping {short_dt}: {nodata_pct:.1f}% nodata (from metadata)")
                continue
        elif nodata_max < 100:
            # Check using actual pixel data if threshold is restrictive
            try:
                # Use NIR band for nodata check (available in all scenes)
                nir_asset = item.assets.get("nir")
                if nir_asset:
                    nodata_pct = calculate_nodata_percentage(nir_asset.href, bbox)
                    if nodata_pct > nodata_max:
                        log(f"  Skipping {short_dt}: {nodata_pct:.1f}% nodata")
                        continue
            except Exception as e:
                log(f"  {item.id}: nodata check failed ({e}), continuing")

        scene_cloud_cover = EOExtension.ext(item).cloud_cover or 0.0

        # For very clear scenes (< 0.1% reported), trust the metadata
        if scene_cloud_cover < PIXEL_CHECK_SKIP_THRESHOLD:
            log(
                f"  {item.id}: scene cloud {scene_cloud_cover:.1f}% (trusted, < {PIXEL_CHECK_SKIP_THRESHOLD}%)"
            )
            actual_cloud_cover = scene_cloud_cover
        else:
            # Calculate actual chip-level cloud cover using SCL
            scl_asset = item.assets.get("scl")
            if scl_asset:
                try:
                    actual_cloud_cover = calculate_pixel_cloud_cover(
                        cloud_href=scl_asset.href,
                        bbox=bbox,
                        cloud_type="scl",
                    )
                    log(
                        f"  {item.id}: scene {scene_cloud_cover:.1f}% -> chip {actual_cloud_cover:.1f}%"
                    )
                except Exception as e:
                    log(f"  {item.id}: chip check failed ({e}), using scene cloud cover")
                    actual_cloud_cover = scene_cloud_cover
            else:
                log(f"  {item.id}: no SCL asset, using scene cloud cover {scene_cloud_cover:.1f}%")
                actual_cloud_cover = scene_cloud_cover

        # Check if chip cloud cover exceeds threshold
        if actual_cloud_cover > cloud_cover_chip:
            log(f"  Skipping {short_dt}: {actual_cloud_cover:.1f}% cloud")
            continue

        # Get scene datetime using helper (handles None datetime with start_datetime fallback)
        try:
            scene_dt = _get_item_datetime(item)
        except ValueError as e:
            log(f"  {item.id}: skipping - {e}")
            continue

        return SelectedScene(
            item=item,
            season=season,
            cloud_cover=actual_cloud_cover,
            datetime=scene_dt,
            stac_url=item.get_self_href() or "",
        )

    return None


def select_scenes_for_chip(
    chip_id: str,
    bbox: tuple[float, float, float, float],
    year: int,
    cloud_cover_chip: float = DEFAULT_CLOUD_COVER_CHIP,
    nodata_max: float = DEFAULT_NODATA_MAX,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
    s2_collection: str = "c1",
    num_buffer_expansions: int = DEFAULT_NUM_BUFFER_EXPANSIONS,
    buffer_expansion_size: int = DEFAULT_BUFFER_EXPANSION_SIZE,
    on_progress: Callable[[str], None] | None = None,
) -> SceneSelectionResult:
    """
    Select optimal Sentinel-2 scenes for a chip based on crop calendar.

    Args:
        chip_id: Chip identifier
        bbox: Bounding box (minx, miny, maxx, maxy) in EPSG:4326
        year: Calendar year for the crop cycle
        cloud_cover_chip: Maximum chip-level cloud cover percentage (0-100)
        nodata_max: Maximum nodata percentage (0-100). Default 0 rejects any nodata.
        buffer_days: Days to search around crop calendar dates
        s2_collection: Sentinel-2 collection identifier
        num_buffer_expansions: Number of times to expand buffer for seasons without cloud-free scenes
        buffer_expansion_size: Days to add to buffer on each expansion
        on_progress: Optional callback for progress messages

    Returns:
        SceneSelectionResult with selected scenes or skip reason

    Note:
        Buffer expansion works per-season. If planting finds a cloud-free scene
        but harvest doesn't, only harvest's buffer will be expanded.
    """

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    # Get crop calendar dates
    try:
        crop_dates = get_crop_calendar_dates(bbox, on_progress=on_progress)
    except ValueError as e:
        return SceneSelectionResult(
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
        "stac_host": "earthsearch",  # Always earthsearch
        "cloud_cover_chip_threshold": cloud_cover_chip,
        "buffer_days": buffer_days,
        "num_buffer_expansions": num_buffer_expansions,
        "buffer_expansion_size": buffer_expansion_size,
    }

    # Helper to check if a scene meets threshold
    def _meets_threshold(scene: SelectedScene | None) -> bool:
        if scene is None:
            return False
        return scene.cloud_cover <= cloud_cover_chip

    # Initialize buffers and scenes for iterative expansion
    planting_buffer = buffer_days
    harvest_buffer = buffer_days
    planting_scene: SelectedScene | None = None
    harvest_scene: SelectedScene | None = None
    candidates_checked = 0
    expansions_performed = 0

    # Track already-checked scene IDs to avoid re-checking on expansion
    planting_checked_ids: set[str] = set()
    harvest_checked_ids: set[str] = set()

    # Iterative buffer expansion loop
    for expansion in range(num_buffer_expansions + 1):
        # Query and select planting scene if not yet found/meeting threshold
        if not _meets_threshold(planting_scene):
            try:
                if expansion > 0:
                    log(f"Expansion {expansion}: planting buffer now {planting_buffer} days")
                log(f"Searching for planting scene around {planting_dt.date()}...")
                planting_result = _query_stac(
                    bbox=bbox,
                    center_date=planting_dt,
                    cloud_cover_max=DEFAULT_CLOUD_COVER_SCENE,  # Internal scene-level filter
                    buffer_days=planting_buffer,
                    s2_collection=s2_collection,
                )
                log(f"STAC Query: {planting_result.catalog_url}")
                log(f"  Collection: {planting_result.collection}")
                log(f"  Bbox: {planting_result.bbox}")
                log(f"  Date range: {planting_result.date_range}")
                log(f"  Cloud cover max: {planting_result.cloud_cover_max}%")

                # Filter out already-checked scenes
                new_items = [
                    item for item in planting_result.items if item.id not in planting_checked_ids
                ]
                candidates_checked += len(new_items)

                if expansion > 0 and len(planting_result.items) > len(new_items):
                    log(
                        f"Found {len(planting_result.items)} total, "
                        f"{len(new_items)} new planting scene candidates"
                    )
                else:
                    log(f"Found {len(new_items)} planting scene candidates")

                for item in new_items[:5]:
                    cc = EOExtension.ext(item).cloud_cover or 0.0
                    log(f"  - {item.id}: {cc:.1f}% cloud, {item.datetime}")

                # Track all items we're about to check
                planting_checked_ids.update(item.id for item in new_items)

                planting_scene = _select_best_scene(
                    new_items,
                    season="planting",
                    bbox=bbox,
                    cloud_cover_chip=cloud_cover_chip,
                    nodata_max=nodata_max,
                    on_progress=on_progress,
                )

                if planting_scene:
                    log(
                        f"Selected planting scene: {planting_scene.id} "
                        f"({planting_scene.cloud_cover:.1f}% cloud)"
                    )
            except ValueError as e:
                return SceneSelectionResult(
                    chip_id=chip_id,
                    bbox=bbox,
                    year=year,
                    crop_calendar=crop_dates,
                    skipped_reason=f"Planting query error: {e}",
                    selection_params=selection_params,
                    planting_buffer_used=planting_buffer,
                    harvest_buffer_used=harvest_buffer,
                    expansions_performed=expansions_performed,
                )

        # Query and select harvest scene if not yet found/meeting threshold
        if not _meets_threshold(harvest_scene):
            try:
                if expansion > 0:
                    log(f"Expansion {expansion}: harvest buffer now {harvest_buffer} days")
                log(f"Searching for harvest scene around {harvest_dt.date()}...")
                harvest_result = _query_stac(
                    bbox=bbox,
                    center_date=harvest_dt,
                    cloud_cover_max=DEFAULT_CLOUD_COVER_SCENE,  # Internal scene-level filter
                    buffer_days=harvest_buffer,
                    s2_collection=s2_collection,
                )
                log(f"STAC Query: {harvest_result.catalog_url}")
                log(f"  Collection: {harvest_result.collection}")
                log(f"  Bbox: {harvest_result.bbox}")
                log(f"  Date range: {harvest_result.date_range}")
                log(f"  Cloud cover max: {harvest_result.cloud_cover_max}%")

                # Filter out already-checked scenes
                new_items = [
                    item for item in harvest_result.items if item.id not in harvest_checked_ids
                ]
                candidates_checked += len(new_items)

                if expansion > 0 and len(harvest_result.items) > len(new_items):
                    log(
                        f"Found {len(harvest_result.items)} total, "
                        f"{len(new_items)} new harvest scene candidates"
                    )
                else:
                    log(f"Found {len(new_items)} harvest scene candidates")

                for item in new_items[:5]:
                    cc = EOExtension.ext(item).cloud_cover or 0.0
                    log(f"  - {item.id}: {cc:.1f}% cloud, {item.datetime}")

                # Track all items we're about to check
                harvest_checked_ids.update(item.id for item in new_items)

                harvest_scene = _select_best_scene(
                    new_items,
                    season="harvest",
                    bbox=bbox,
                    cloud_cover_chip=cloud_cover_chip,
                    nodata_max=nodata_max,
                    on_progress=on_progress,
                )

                if harvest_scene:
                    log(
                        f"Selected harvest scene: {harvest_scene.id} "
                        f"({harvest_scene.cloud_cover:.1f}% cloud)"
                    )
            except ValueError as e:
                return SceneSelectionResult(
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
                    expansions_performed=expansions_performed,
                )

        # Check if both seasons meet threshold
        planting_ok = _meets_threshold(planting_scene)
        harvest_ok = _meets_threshold(harvest_scene)

        if planting_ok and harvest_ok:
            log(f"Both seasons meet threshold after {expansion} expansion(s)")
            break

        # If we have more expansions to try, expand buffers for failing seasons
        if expansion < num_buffer_expansions:
            if not planting_ok:
                planting_buffer += buffer_expansion_size
            if not harvest_ok:
                harvest_buffer += buffer_expansion_size
            expansions_performed = expansion + 1

    # Determine skip reason if any scene is missing
    skipped_reason = None
    if planting_scene is None and harvest_scene is None:
        skipped_reason = "No cloud-free scenes found for either season"
    elif planting_scene is None:
        skipped_reason = "No cloud-free planting scene found"
    elif harvest_scene is None:
        skipped_reason = "No cloud-free harvest scene found"

    return SceneSelectionResult(
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
        expansions_performed=expansions_performed,
    )
