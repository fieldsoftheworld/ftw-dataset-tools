"""Scene selection from STAC catalogs based on crop calendar dates."""

from __future__ import annotations

import threading
import time
import urllib.error
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

import pystac_client
from pystac.extensions.eo import EOExtension

if TYPE_CHECKING:
    import pystac

from ftw_dataset_tools.api.imagery import parquet_search
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


# One client per thread. A client owns an HTTP session and a plain dict of
# resolved STAC objects, neither of which is built for concurrent use, and
# selection runs several chips at once - so caching per thread keeps the reuse
# without the sharing.
_CLIENTS = threading.local()


def _get_stac_client(catalog_url: str) -> pystac_client.Client:
    """Get this thread's STAC client for a catalog URL, opening it on first use."""
    clients: dict[str, pystac_client.Client] | None = getattr(_CLIENTS, "by_url", None)
    if clients is None:
        clients = {}
        _CLIENTS.by_url = clients

    client = clients.get(catalog_url)
    if client is None:
        client = pystac_client.Client.open(catalog_url)
        clients[catalog_url] = client

    return client


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


def _query_parquet(
    bbox: tuple[float, float, float, float],
    center_date: datetime,
    cloud_cover_max: int,
    buffer_days: int,
    s2_collection: str = "c1",
) -> STACQueryResult:
    """Query the Sentinel-2 STAC-GeoParquet mirror for scenes.

    Runs the same search as :func:`_query_stac` (bbox, window, scene-level
    cloud filter, sorted by cloud cover) against the parquet mirror, which has
    no API and therefore no rate limit.
    """
    _validate_date_not_future(center_date, buffer_days)
    start = (center_date - timedelta(days=buffer_days)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end = (center_date + timedelta(days=buffer_days)).replace(
        hour=23, minute=59, second=59, microsecond=0
    )
    collection = S2_COLLECTIONS.get(s2_collection, parquet_search.COLLECTION_C1)
    items = parquet_search.query_scenes(
        bbox=bbox, start=start, end=end, cloud_cover_max=cloud_cover_max, collection=collection
    )
    return STACQueryResult(
        items=items,
        catalog_url=f"{parquet_search.DEFAULT_MIRROR_ROOT}/{collection}",
        collection=collection,
        bbox=bbox,
        date_range=_format_date_range(center_date, buffer_days),
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

        # Check nodata first (fail fast). A scene-level value of zero proves the
        # chip's window is clean; any other value says nothing about this chip
        # (granule-edge nodata may sit far from it), so check the actual pixels.
        # Old-baseline products in particular carry edge nodata on most scenes.
        scene_nodata = get_nodata_from_metadata(item)
        if not (scene_nodata is not None and scene_nodata <= 0) and nodata_max < 100:
            try:
                # Use NIR band for nodata check (available in all scenes)
                nir_asset = item.assets.get("nir")
                if nir_asset:
                    nodata_pct = calculate_nodata_percentage(nir_asset.href, bbox)
                    if nodata_pct > nodata_max:
                        log(f"  Skipping {short_dt}: {nodata_pct:.1f}% nodata in chip window")
                        continue
            except Exception as e:
                # Without the pixel check, fall back to the scene-level value.
                if scene_nodata is not None and scene_nodata > nodata_max:
                    log(
                        f"  Skipping {short_dt}: nodata check failed ({e}), "
                        f"scene reports {scene_nodata:.1f}% nodata"
                    )
                    continue
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


@dataclass
class _SeasonSearch:
    """One season's search state, carried across buffer expansions."""

    season: Literal["planting", "harvest"]
    center_date: datetime
    buffer_days: int
    scene: SelectedScene | None = None
    candidates_checked: int = 0
    checked_ids: set[str] = field(default_factory=set)

    def meets_threshold(self, cloud_cover_chip: float) -> bool:
        return self.scene is not None and self.scene.cloud_cover <= cloud_cover_chip


def _log_query(query: STACQueryResult, log: Callable[[str], None]) -> None:
    log(f"STAC Query: {query.catalog_url}")
    log(f"  Collection: {query.collection}")
    log(f"  Bbox: {query.bbox}")
    log(f"  Date range: {query.date_range}")
    log(f"  Cloud cover max: {query.cloud_cover_max}%")


def _search_season(
    state: _SeasonSearch,
    expansion: int,
    run_search: Callable[[datetime, int], STACQueryResult],
    *,
    bbox: tuple[float, float, float, float],
    cloud_cover_chip: float,
    nodata_max: float,
    on_progress: Callable[[str], None] | None,
) -> None:
    """Query one season's window and select its best scene among unseen candidates.

    Raises:
        ValueError: If the scene query fails.
    """

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    season = state.season
    if expansion > 0:
        log(f"Expansion {expansion}: {season} buffer now {state.buffer_days} days")
    log(f"Searching for {season} scene around {state.center_date.date()}...")
    query = run_search(state.center_date, state.buffer_days)
    _log_query(query, log)

    # Filter out scenes already checked on an earlier expansion
    new_items = [item for item in query.items if item.id not in state.checked_ids]
    state.candidates_checked += len(new_items)

    if expansion > 0 and len(query.items) > len(new_items):
        log(f"Found {len(query.items)} total, {len(new_items)} new {season} scene candidates")
    else:
        log(f"Found {len(new_items)} {season} scene candidates")

    for item in new_items[:5]:
        cc = EOExtension.ext(item).cloud_cover or 0.0
        log(f"  - {item.id}: {cc:.1f}% cloud, {item.datetime}")

    state.checked_ids.update(item.id for item in new_items)
    state.scene = _select_best_scene(
        new_items,
        season=season,
        bbox=bbox,
        cloud_cover_chip=cloud_cover_chip,
        nodata_max=nodata_max,
        on_progress=on_progress,
    )
    if state.scene:
        log(f"Selected {season} scene: {state.scene.id} ({state.scene.cloud_cover:.1f}% cloud)")


def _missing_scene_reason(planting: _SeasonSearch, harvest: _SeasonSearch) -> str | None:
    if planting.scene is None and harvest.scene is None:
        return "No cloud-free scenes found for either season"
    if planting.scene is None:
        return "No cloud-free planting scene found"
    if harvest.scene is None:
        return "No cloud-free harvest scene found"
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
    search_backend: str = "parquet",
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
        s2_collection: Sentinel-2 collection identifier ("c1" or "old-baseline"),
            used by both backends
        num_buffer_expansions: Number of times to expand buffer for seasons without cloud-free scenes
        buffer_expansion_size: Days to add to buffer on each expansion
        search_backend: "parquet" (the STAC-GeoParquet mirror, default) or
            "earth-search" (the Earth Search STAC API)
        on_progress: Optional callback for progress messages

    Returns:
        SceneSelectionResult with selected scenes or skip reason

    Note:
        Buffer expansion works per-season. If planting finds a cloud-free scene
        but harvest doesn't, only harvest's buffer will be expanded.
    """

    if search_backend not in ("parquet", "earth-search"):
        raise ValueError(
            f"Unknown search_backend {search_backend!r}; use 'parquet' or 'earth-search'."
        )

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    def _run_search(center_date: datetime, buffer: int) -> STACQueryResult:
        if search_backend == "parquet":
            return _query_parquet(
                bbox, center_date, DEFAULT_CLOUD_COVER_SCENE, buffer, s2_collection
            )
        return _query_stac(
            bbox=bbox,
            center_date=center_date,
            cloud_cover_max=DEFAULT_CLOUD_COVER_SCENE,
            buffer_days=buffer,
            s2_collection=s2_collection,
        )

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
        "stac_host": "parquet-mirror" if search_backend == "parquet" else "earthsearch",
        "cloud_cover_chip_threshold": cloud_cover_chip,
        "buffer_days": buffer_days,
        "num_buffer_expansions": num_buffer_expansions,
        "buffer_expansion_size": buffer_expansion_size,
    }

    planting = _SeasonSearch("planting", planting_dt, buffer_days)
    harvest = _SeasonSearch("harvest", harvest_dt, buffer_days)
    expansions_performed = 0

    def result(**kwargs) -> SceneSelectionResult:
        return SceneSelectionResult(
            chip_id=chip_id,
            bbox=bbox,
            year=year,
            crop_calendar=crop_dates,
            planting_scene=planting.scene,
            selection_params=selection_params,
            planting_buffer_used=planting.buffer_days,
            harvest_buffer_used=harvest.buffer_days,
            expansions_performed=expansions_performed,
            **kwargs,
        )

    # Iterative buffer expansion loop; each season expands only until it has a scene
    for expansion in range(num_buffer_expansions + 1):
        for state in (planting, harvest):
            if state.meets_threshold(cloud_cover_chip):
                continue
            try:
                _search_season(
                    state,
                    expansion,
                    _run_search,
                    bbox=bbox,
                    cloud_cover_chip=cloud_cover_chip,
                    nodata_max=nodata_max,
                    on_progress=on_progress,
                )
            except ValueError as e:
                if state is planting:
                    return result(skipped_reason=f"Planting query error: {e}")
                return result(
                    skipped_reason=f"Harvest query error: {e}",
                    candidates_checked=planting.candidates_checked + harvest.candidates_checked,
                )

        pending = [s for s in (planting, harvest) if not s.meets_threshold(cloud_cover_chip)]
        if not pending:
            log(f"Both seasons meet threshold after {expansion} expansion(s)")
            break

        # If we have more expansions to try, expand buffers for failing seasons
        if expansion < num_buffer_expansions:
            for state in pending:
                state.buffer_days += buffer_expansion_size
            expansions_performed = expansion + 1

    return result(
        harvest_scene=harvest.scene,
        skipped_reason=_missing_scene_reason(planting, harvest),
        candidates_checked=planting.candidates_checked + harvest.candidates_checked,
    )
