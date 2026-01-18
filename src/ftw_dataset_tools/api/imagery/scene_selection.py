"""Scene selection from STAC catalogs based on crop calendar dates."""

from __future__ import annotations

import time
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
from ftw_dataset_tools.api.imagery.settings import (
    DEFAULT_BUFFER_DAYS,
    DEFAULT_BUFFER_EXPANSION_SIZE,
    DEFAULT_CLOUD_COVER_SCENE,
    DEFAULT_NUM_BUFFER_EXPANSIONS,
    EARTHSEARCH_URL,
    MSPC_URL,
    PIXEL_CHECK_SKIP_THRESHOLD,
    S2_COLLECTIONS,
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
    stac_host: Literal["earthsearch", "mspc"],
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
        stac_host: STAC host to query
        cloud_cover_max: Maximum cloud cover percentage
        buffer_days: Days to search around center_date
        s2_collection: Sentinel-2 collection identifier (earthsearch only)
        max_retries: Maximum number of retries for transient errors

    Returns:
        STACQueryResult with items and query details

    Raises:
        Exception: If all retries fail
    """
    _validate_date_not_future(center_date, buffer_days)

    date_range = _format_date_range(center_date, buffer_days)

    if stac_host == "earthsearch":
        catalog_url = EARTHSEARCH_URL
        collection = S2_COLLECTIONS["earthsearch"].get(s2_collection, "sentinel-2-c1-l2a")
    elif stac_host == "mspc":
        catalog_url = MSPC_URL
        collection = S2_COLLECTIONS["mspc"]["default"]
    else:
        raise ValueError(f"Unknown STAC host: {stac_host}")

    last_error = None
    for attempt in range(max_retries):
        try:
            # Open catalog and search
            catalog = pystac_client.Client.open(catalog_url)
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
    season: Literal["planting", "harvest"],
    bbox: tuple[float, float, float, float],
    pixel_check: bool = False,
    cloud_cover_pixel: float = 0.0,
    on_progress: Callable[[str], None] | None = None,
) -> SelectedScene | None:
    """
    Select the best scene from candidates.

    Args:
        items: List of candidate STAC items (pre-sorted by cloud cover)
        season: Season identifier
        bbox: Bounding box for pixel-level cloud calculation
        pixel_check: Whether to perform pixel-level cloud check
        cloud_cover_pixel: Maximum pixel-level cloud cover (if pixel_check enabled)
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
        scene_cloud_cover = EOExtension.ext(item).cloud_cover or 0.0
        actual_cloud_cover = scene_cloud_cover  # Default to scene-level

        # If pixel check is enabled, calculate actual cloud cover over the bbox
        if pixel_check:
            # For very clear scenes (< 0.1% reported), trust the metadata
            if scene_cloud_cover < PIXEL_CHECK_SKIP_THRESHOLD:
                log(
                    f"  {item.id}: scene cloud {scene_cloud_cover:.1f}% (trusted, < {PIXEL_CHECK_SKIP_THRESHOLD}%)"
                )
                actual_cloud_cover = scene_cloud_cover
            else:
                # Calculate actual pixel-level cloud cover using SCL
                scl_asset = item.assets.get("scl")
                if scl_asset:
                    try:
                        actual_cloud_cover = calculate_pixel_cloud_cover(
                            cloud_href=scl_asset.href,
                            bbox=bbox,
                            cloud_type="scl",
                        )
                        log(
                            f"  {item.id}: scene {scene_cloud_cover:.1f}% -> pixel {actual_cloud_cover:.1f}%"
                        )
                    except Exception as e:
                        log(f"  {item.id}: pixel check failed ({e}), using scene cloud cover")
                        actual_cloud_cover = scene_cloud_cover
                else:
                    log(
                        f"  {item.id}: no SCL asset, using scene cloud cover {scene_cloud_cover:.1f}%"
                    )
                    actual_cloud_cover = scene_cloud_cover

                # Check if pixel cloud cover exceeds threshold
                if actual_cloud_cover > cloud_cover_pixel:
                    short_dt = _short_date(item)
                    log(f"  Skipping {short_dt}: {actual_cloud_cover:.1f}% cloud")
                    continue

        # Get scene datetime
        scene_dt = item.datetime or item.properties.get("datetime")
        if isinstance(scene_dt, str):
            scene_dt = datetime.fromisoformat(scene_dt.replace("Z", "+00:00"))

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
    stac_host: Literal["earthsearch", "mspc"] = "earthsearch",
    cloud_cover_scene: int = DEFAULT_CLOUD_COVER_SCENE,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
    pixel_check: bool = True,
    cloud_cover_pixel: float = 0.0,
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
        stac_host: STAC host to query ("earthsearch" or "mspc")
        cloud_cover_scene: Maximum scene-level cloud cover percentage
        buffer_days: Days to search around crop calendar dates
        pixel_check: Enable pixel-level cloud filtering using SCL band
        cloud_cover_pixel: Maximum pixel-level cloud cover (if pixel_check enabled)
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
        "stac_host": stac_host,
        "cloud_cover_scene_threshold": cloud_cover_scene,
        "buffer_days": buffer_days,
        "pixel_check": pixel_check,
        "num_buffer_expansions": num_buffer_expansions,
        "buffer_expansion_size": buffer_expansion_size,
    }
    if pixel_check:
        selection_params["cloud_cover_pixel_threshold"] = cloud_cover_pixel

    # Helper to check if a scene meets threshold
    def _meets_threshold(scene: SelectedScene | None) -> bool:
        if scene is None:
            return False
        threshold = cloud_cover_pixel if pixel_check else cloud_cover_scene
        return scene.cloud_cover <= threshold

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
                    stac_host=stac_host,
                    cloud_cover_max=cloud_cover_scene,
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
                    pixel_check=pixel_check,
                    cloud_cover_pixel=cloud_cover_pixel,
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
                    stac_host=stac_host,
                    cloud_cover_max=cloud_cover_scene,
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
                    pixel_check=pixel_check,
                    cloud_cover_pixel=cloud_cover_pixel,
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
