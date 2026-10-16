"""PlanetScope planting and harvest scene selection for a run's chips.

The Planet counterpart of ``select_scenes_for_chip``: same crop calendar windows,
same cloud and nodata limits, same buffer expansion. The difference is where the
candidates come from. :func:`planet_prefetch.prefetch_scenes` orders and downloads
them for the whole run at once, and each chip then rates its candidates from their
delivered UDM2 files, so no chip ever calls the Planet API itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import pystac
from pystac.extensions.eo import EOExtension
from shapely.geometry import shape

from ftw_dataset_tools.api.imagery import planet_api
from ftw_dataset_tools.api.imagery.cloud_analysis import read_udm2_chip_stats
from ftw_dataset_tools.api.imagery.crop_calendar import CropCalendarDates, get_crop_calendar_dates
from ftw_dataset_tools.api.imagery.parallel import ParallelOutcome, run_in_parallel
from ftw_dataset_tools.api.imagery.planet_prefetch import (
    ChipWindow,
    PlanetCache,
    PrefetchResult,
    prefetch_scenes,
)
from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult, SelectedScene
from ftw_dataset_tools.api.imagery.settings import DEFAULT_CLOUD_COVER_SCENE
from ftw_dataset_tools.api.logging_config import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

__all__ = ["PENDING_REASON", "PlanetChip", "select_planet_scenes", "to_item"]

PENDING_REASON = "Planet orders still processing; rerun to finish"
SEASONS = ("planting", "harvest")


@dataclass
class PlanetChip:
    """A chip to select PlanetScope scenes for."""

    chip_id: str
    bbox: tuple[float, float, float, float]
    year: int


@dataclass
class _Season:
    window: ChipWindow
    scene: SelectedScene | None = None
    candidates_checked: int = 0
    pending: bool = False


@dataclass
class _ChipState:
    chip: PlanetChip
    calendar: CropCalendarDates
    seasons: dict[str, _Season] = field(default_factory=dict)
    expansions: int = 0


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


def _choose(
    window: ChipWindow,
    features: list[dict],
    cache: PlanetCache,
    *,
    cloud_cover_chip: float,
    nodata_max: float,
) -> SelectedScene | None:
    """The candidate clearest over the chip, judged from its delivered UDM2."""
    best: tuple[float, dict, object] | None = None
    for feature in features:
        for delivery in cache.delivered(feature["id"]):
            stats = read_udm2_chip_stats(str(delivery.udm2), window.bbox)
            if stats.nodata_pct > nodata_max or stats.cloud_pct > cloud_cover_chip:
                continue
            if best is None or stats.cloud_pct < best[0]:
                best = (stats.cloud_pct, feature, delivery)
    if best is None:
        return None
    cloud, feature, delivery = best
    item = to_item(feature)
    item.add_asset("analytic", pystac.Asset(href=str(delivery.analytic), roles=["data"]))
    item.add_asset("udm2", pystac.Asset(href=str(delivery.udm2), roles=["cloud"]))
    item.properties["ftw:planet_order_id"] = delivery.order_id
    return SelectedScene(
        item=item,
        season=window.season,
        cloud_cover=cloud,
        datetime=item.datetime,
        stac_url=item.get_self_href() or "",
    )


def _start(chip: PlanetChip, buffer_days: int) -> _ChipState | SceneSelectionResult:
    """The chip's seasonal windows, or a skipped result when the crop calendar fails."""
    try:
        calendar = get_crop_calendar_dates(chip.bbox)
    except ValueError as e:
        return SceneSelectionResult(
            chip_id=chip.chip_id,
            bbox=chip.bbox,
            year=chip.year,
            crop_calendar=CropCalendarDates(0, 0),
            skipped_reason=f"Crop calendar error: {e}",
        )
    state = _ChipState(chip, calendar)
    for season, center in zip(SEASONS, calendar.to_datetime(chip.year), strict=True):
        state.seasons[season] = _Season(
            ChipWindow(chip.chip_id, chip.bbox, season, center, buffer_days)
        )
    return state


def _rate_round(
    open_seasons: list[_Season],
    prefetched: PrefetchResult,
    cache: PlanetCache,
    *,
    cloud_cover_chip: float,
    nodata_max: float,
    workers: int,
) -> None:
    """Rate each open season's delivered candidates; record chosen scenes on the states."""
    pending_groups = bool(prefetched.pending_orders)

    def work(season: _Season) -> SelectedScene | None:
        features = prefetched.candidates.get((season.window.chip_id, season.window.season), [])
        season.candidates_checked += len(features)
        return _choose(
            season.window,
            features,
            cache,
            cloud_cover_chip=cloud_cover_chip,
            nodata_max=nodata_max,
        )

    def apply(outcome: ParallelOutcome[_Season, SelectedScene | None]) -> None:
        season = outcome.task
        if outcome.error is not None:
            logger.warning(f"{season.window.chip_id} {season.window.season}: {outcome.error}")
            return
        season.scene = outcome.value
        season.pending = season.scene is None and pending_groups

    run_in_parallel(open_seasons, work=work, apply=apply, workers=workers)


def _result(state: _ChipState, selection_params: dict) -> SceneSelectionResult:
    planting, harvest = state.seasons["planting"], state.seasons["harvest"]
    missing = [s for s in SEASONS if state.seasons[s].scene is None]
    reason = None
    if any(state.seasons[s].pending for s in missing):
        reason = PENDING_REASON
    elif len(missing) == 2:
        reason = "No cloud-free scenes found for either season"
    elif missing:
        reason = f"No cloud-free {missing[0]} scene found"
    return SceneSelectionResult(
        chip_id=state.chip.chip_id,
        bbox=state.chip.bbox,
        year=state.chip.year,
        crop_calendar=state.calendar,
        planting_scene=planting.scene,
        harvest_scene=harvest.scene,
        skipped_reason=reason,
        candidates_checked=planting.candidates_checked + harvest.candidates_checked,
        selection_params=selection_params,
        planting_buffer_used=planting.window.buffer_days,
        harvest_buffer_used=harvest.window.buffer_days,
        expansions_performed=state.expansions,
    )


def select_planet_scenes(
    chips: list[PlanetChip],
    cache_root: Path,
    *,
    cloud_cover_chip: float,
    nodata_max: float,
    buffer_days: int,
    num_buffer_expansions: int,
    buffer_expansion_size: int,
    workers: int,
    harmonize: bool = False,
    wait: bool = True,
    timeout_minutes: float = 120.0,
) -> dict[str, SceneSelectionResult]:
    """Choose a planting and a harvest PlanetScope scene for every chip.

    Each round prefetches the candidates of every season still open, then rates
    them chip by chip; seasons left without a scene widen their window for the next
    round, as ``select_scenes_for_chip`` does for Sentinel-2.
    """
    results: dict[str, SceneSelectionResult] = {}
    states: list[_ChipState] = []
    for chip in chips:
        started = _start(chip, buffer_days)
        if isinstance(started, SceneSelectionResult):
            results[chip.chip_id] = started
        else:
            states.append(started)

    selection_params = {
        "stac_host": "planet-data-api",
        "cloud_cover_chip_threshold": cloud_cover_chip,
        "buffer_days": buffer_days,
        "num_buffer_expansions": num_buffer_expansions,
        "buffer_expansion_size": buffer_expansion_size,
    }
    cache = PlanetCache(cache_root)
    for expansion in range(num_buffer_expansions + 1):
        open_seasons = [s for st in states for s in st.seasons.values() if s.scene is None]
        if not open_seasons:
            break
        logger.info(f"Planet round {expansion}: {len(open_seasons)} chip season(s) to select")
        prefetched = prefetch_scenes(
            [s.window for s in open_seasons],
            cache_root,
            cloud_cover_max=DEFAULT_CLOUD_COVER_SCENE,
            nodata_max=nodata_max,
            harmonize=harmonize,
            wait=wait,
            timeout_minutes=timeout_minutes,
        )
        cache = PlanetCache(cache_root)
        _rate_round(
            open_seasons,
            prefetched,
            cache,
            cloud_cover_chip=cloud_cover_chip,
            nodata_max=nodata_max,
            workers=workers,
        )
        if prefetched.pending_orders or expansion == num_buffer_expansions:
            # Orders still processing: widening windows now would just order more.
            break
        for state in states:
            widened = False
            for season in state.seasons.values():
                if season.scene is None:
                    w = season.window
                    season.window = ChipWindow(
                        w.chip_id, w.bbox, w.season, w.center, w.buffer_days + buffer_expansion_size
                    )
                    widened = True
            if widened:
                state.expansions = expansion + 1

    for state in states:
        results[state.chip.chip_id] = _result(state, selection_params)
    return results
