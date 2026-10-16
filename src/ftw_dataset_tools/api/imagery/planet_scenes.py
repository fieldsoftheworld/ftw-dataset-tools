"""PlanetScope scene search and per-chip scene choice for scenes mode.

Plugs into :func:`scene_selection.select_scenes_for_chip` in place of the Sentinel-2
search and pixel check, so the crop calendar windows and buffer expansion stay shared.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pystac
from pystac.extensions.eo import EOExtension
from shapely.geometry import box, shape

from ftw_dataset_tools.api.imagery import planet_api
from ftw_dataset_tools.api.imagery.scene_selection import (
    SelectedScene,
    STACQueryResult,
    _format_date_range,
    _status,
    _validate_date_not_future,
)
from ftw_dataset_tools.api.logging_config import get_logger

if TYPE_CHECKING:
    from typing import Literal

logger = get_logger(__name__)

__all__ = ["MAX_UDM2_CANDIDATES", "query_planet", "select_best_planet_scene", "to_item"]

#: Candidates rated with UDM2 per season window. Rating one takes minutes the first
#: time (Planet activates the scene's UDM2), so they are rated together, and the
#: scene-level clear percent picks which ones.
MAX_UDM2_CANDIDATES = 10


def to_item(feature: dict) -> pystac.Item:
    """A Data API search result as a pystac Item, with cloud cover in percent."""
    props = feature["properties"]
    properties: dict = {}
    if props.get("cloud_cover") is not None:
        properties["eo:cloud_cover"] = round(props["cloud_cover"] * 100, 2)
    for key in ("clear_percent", "instrument", "satellite_id", "gsd"):
        if key in props:
            properties[f"pl:{key}"] = props[key]
    geometry = feature["geometry"]
    item = pystac.Item(
        id=feature["id"],
        geometry=geometry,
        bbox=list(shape(geometry).bounds),
        datetime=datetime.fromisoformat(props["acquired"].replace("Z", "+00:00")),
        properties=properties,
    )
    EOExtension.ext(item, add_if_missing=True)
    item.set_self_href(
        f"{planet_api.DATA_URL}/item-types/{planet_api.ITEM_TYPE}/items/{feature['id']}"
    )
    return item


def _scene_clear(item: pystac.Item) -> float:
    clear = item.properties.get("pl:clear_percent")
    if clear is not None:
        return float(clear)
    return 100.0 - float(item.properties.get("eo:cloud_cover") or 0.0)


def query_planet(
    bbox: tuple[float, float, float, float],
    center_date: datetime,
    cloud_cover_max: int,
    buffer_days: int,
) -> STACQueryResult:
    """PSScene candidates in the window around ``center_date``, clearest scene first."""
    _validate_date_not_future(center_date, buffer_days)
    start = (center_date - timedelta(days=buffer_days)).replace(hour=0, minute=0, second=0)
    end = (center_date + timedelta(days=buffer_days)).replace(hour=23, minute=59, second=59)
    try:
        features = planet_api.search_scenes(bbox, start, end, cloud_cover_max=cloud_cover_max)
    except planet_api.PlanetAPIError as err:
        # The season loop reports a failed query as the chip's skip reason.
        raise ValueError(str(err)) from err
    items = sorted((to_item(f) for f in features), key=_scene_clear, reverse=True)
    return STACQueryResult(
        items=items,
        catalog_url=planet_api.DATA_URL,
        collection=f"{planet_api.ITEM_TYPE}/{planet_api.SCENE_ASSET}",
        bbox=bbox,
        date_range=_format_date_range(center_date, buffer_days),
        cloud_cover_max=cloud_cover_max,
    )


def _outside_footprint_pct(item: pystac.Item, bbox: tuple[float, float, float, float]) -> float:
    chip = box(*bbox)
    return 100.0 * (1.0 - chip.intersection(shape(item.geometry)).area / chip.area)


def _chip_cloud(item: pystac.Item, bbox: tuple[float, float, float, float]) -> float | Exception:
    """Not-clear percent of the chip from the scene's UDM2, or the error that stopped it."""
    try:
        return 100.0 - planet_api.chip_clear_percent(item.id, bbox)
    except planet_api.PlanetUnavailableError:
        raise
    except Exception as e:
        return e


def select_best_planet_scene(
    items: list[pystac.Item],
    season: Literal["planting", "harvest"],
    bbox: tuple[float, float, float, float],
    cloud_cover_chip: float,
    nodata_max: float,
) -> SelectedScene | None:
    """The clearest candidate over the chip within ``cloud_cover_chip``, or None.

    Nodata is the part of the chip outside the scene footprint. Cloud is the chip's
    not-clear share in the scene's UDM2, falling back to the scene-level value when
    the coverage request fails.
    """
    usable: list[pystac.Item] = []
    for item in items:
        outside = _outside_footprint_pct(item, bbox)
        if outside > nodata_max:
            _status("✗", f"Skipping {item.id}: {outside:.1f}% of chip outside the scene")
        else:
            usable.append(item)
    candidates = usable[:MAX_UDM2_CANDIDATES]
    if not candidates:
        return None

    # Rated on a pool, but logged here: log capture follows the calling thread.
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        ratings = list(pool.map(lambda item: _chip_cloud(item, bbox), candidates))

    best: tuple[float, pystac.Item] | None = None
    for item, rating in zip(candidates, ratings, strict=True):
        if isinstance(rating, Exception):
            cloud = float(item.properties.get("eo:cloud_cover") or 0.0)
            logger.debug(f"{item.id}: UDM2 check failed ({rating}), using scene cloud cover")
        else:
            cloud = rating
            logger.debug(f"{item.id}: chip {cloud:.1f}% not clear (UDM2)")
        if cloud > cloud_cover_chip:
            _status("✗", f"Skipping {item.id}: {cloud:.1f}% cloud over the chip")
        elif best is None or cloud < best[0]:
            best = (cloud, item)

    if best is None:
        return None
    cloud, item = best
    return SelectedScene(
        item=item,
        season=season,
        cloud_cover=cloud,
        datetime=item.datetime,
        stac_url=item.get_self_href() or "",
    )
