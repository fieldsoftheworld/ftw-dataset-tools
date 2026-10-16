"""Fetch the PlanetScope scenes a selection run can choose from, once per run.

Asking Planet about every candidate scene of every chip does not scale to millions of
chips. This step works per area instead, the way ``_build_scene_index`` prefetches the
Sentinel-2 catalog: chips are grouped by MGRS 100 km square and season, each group
runs one search, each chip keeps its few clearest covering scenes, and the union of
those scenes is ordered (clipped to the group's chips, UDM2 included) and downloaded
into a local cache. Chips then choose among the delivered scenes from their UDM2
files, with no further API calls.

The cache keeps a manifest of searches' features, submitted orders and delivered
files, so an interrupted run resumes without placing orders twice.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from ftw_dataset_tools.api.imagery import planet_api
from ftw_dataset_tools.api.logging_config import get_logger
from ftw_dataset_tools.api.masks import get_mgrs_square

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

logger = get_logger(__name__)

__all__ = [
    "CANDIDATES_PER_CHIP",
    "ChipWindow",
    "Delivery",
    "PlanetCache",
    "PrefetchResult",
    "prefetch_scenes",
]

#: Clearest covering scenes ordered per chip and season window. Every one is
#: downloaded, so this bounds the download per chip; the chip's own UDM2 check then
#: picks among them.
CANDIDATES_PER_CHIP = 5
#: Scenes per order. Planet allows 500; smaller orders finish and fail independently.
ORDER_BATCH = 100
_ORDER_DONE = frozenset({"success", "partial"})
_ORDER_FAILED = frozenset({"failed", "cancelled"})
# Margin around the chips in the clip AOI, so warping has pixels at the edges.
_CLIP_PAD_DEGREES = 0.0005
_MANIFEST = "manifest.json"


@dataclass(frozen=True)
class ChipWindow:
    """One chip's search window for one season."""

    chip_id: str
    bbox: tuple[float, float, float, float]
    season: str
    center: datetime
    buffer_days: int

    @property
    def start(self) -> datetime:
        return (self.center - timedelta(days=self.buffer_days)).replace(hour=0, minute=0, second=0)

    @property
    def end(self) -> datetime:
        return (self.center + timedelta(days=self.buffer_days)).replace(
            hour=23, minute=59, second=59
        )


@dataclass(frozen=True)
class Delivery:
    """One delivered, clipped copy of a scene: its 4-band SR image and UDM2."""

    order_id: str
    analytic: Path
    udm2: Path


@dataclass
class PrefetchResult:
    """What each chip window can choose from, and how the fetch went."""

    #: (chip id, season) -> features of its delivered candidate scenes, clearest first.
    candidates: dict[tuple[str, str], list[dict]] = field(default_factory=dict)
    #: Orders still being processed when the run stopped waiting.
    pending_orders: list[str] = field(default_factory=list)
    failed_orders: list[str] = field(default_factory=list)


class PlanetCache:
    """The run's ``.planet`` directory: delivered files plus a JSON manifest."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._path = root / _MANIFEST
        data = json.loads(self._path.read_text()) if self._path.exists() else {}
        self.features: dict[str, dict] = data.get("features", {})
        #: order id -> {"scenes": [...], "state": ..., "group": ...}
        self.orders: dict[str, dict] = data.get("orders", {})
        #: scene id -> [{"order_id", "analytic", "udm2"}]
        self.deliveries: dict[str, list[dict]] = data.get("deliveries", {})

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {"features": self.features, "orders": self.orders, "deliveries": self.deliveries},
                indent=1,
            )
        )
        tmp.replace(self._path)

    def delivered(self, scene_id: str) -> list[Delivery]:
        return [
            Delivery(d["order_id"], self.root / d["analytic"], self.root / d["udm2"])
            for d in self.deliveries.get(scene_id, [])
        ]

    def ordered_for(self, group: str) -> set[str]:
        """Scenes already ordered for ``group`` by an order that has not failed."""
        return {
            scene
            for order in self.orders.values()
            if order["group"] == group and order["state"] not in _ORDER_FAILED
            for scene in order["scenes"]
        }


def _scene_clear(feature: dict) -> float:
    props = feature["properties"]
    if props.get("clear_percent") is not None:
        return float(props["clear_percent"])
    return 100.0 - 100.0 * float(props.get("cloud_cover") or 0.0)


def _acquired(feature: dict) -> datetime:
    return datetime.fromisoformat(feature["properties"]["acquired"].replace("Z", "+00:00"))


def _coverage(feature: dict, bbox: tuple[float, float, float, float]) -> float:
    chip = box(*bbox)
    return 100.0 * chip.intersection(shape(feature["geometry"])).area / chip.area


def _group_key(window: ChipWindow) -> str:
    # Chip item ids carry the dataset year ("ftw-33UXP0410_2024"); the square
    # comes from the grid id before it.
    grid_id = re.sub(r"_\d{4}$", "", window.chip_id)
    return f"{get_mgrs_square(grid_id)}/{window.season}"


def _chip_candidates(
    window: ChipWindow, features: Iterable[dict], nodata_max: float, limit: int
) -> list[dict]:
    """The window's clearest scenes that were acquired in it and cover the chip."""
    start, end = window.start.timestamp(), window.end.timestamp()
    covering = [
        f
        for f in features
        if start <= _acquired(f).timestamp() <= end
        and _coverage(f, window.bbox) >= 100.0 - nodata_max
    ]
    return sorted(covering, key=_scene_clear, reverse=True)[:limit]


def _search_group(windows: list[ChipWindow], cloud_cover_max: float) -> list[dict]:
    bbox = unary_union([box(*w.bbox) for w in windows]).bounds
    start = min(w.start for w in windows)
    end = max(w.end for w in windows)
    return planet_api.search_scenes(bbox, start, end, cloud_cover_max=cloud_cover_max)


def _clip_aoi(windows: list[ChipWindow]) -> dict:
    return mapping(unary_union([box(*w.bbox).buffer(_CLIP_PAD_DEGREES) for w in windows]))


def _submit_orders(
    cache: PlanetCache, group: str, scenes: list[str], aoi: dict, *, harmonize: bool
) -> None:
    for start in range(0, len(scenes), ORDER_BATCH):
        batch = scenes[start : start + ORDER_BATCH]
        name = f"ftwd {group} {start // ORDER_BATCH + 1}"
        order_id = planet_api.create_order(name, batch, aoi=aoi, harmonize=harmonize)
        cache.orders[order_id] = {"scenes": batch, "state": "queued", "group": group}
        cache.save()
        logger.info(f"Submitted Planet order {order_id} for {len(batch)} scenes ({group})")


def _scene_of(filename: str) -> str | None:
    """Scene id of a delivered PSScene file ('<scene>_3B_AnalyticMS_SR_clip.tif')."""
    return filename.split("_3B_", 1)[0] if "_3B_" in filename else None


def _record_delivery(cache: PlanetCache, order_id: str, paths: list[Path]) -> None:
    by_scene: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in paths:
        scene = _scene_of(path.name)
        if scene is None or path.suffix != ".tif":
            continue
        kind = (
            "udm2" if "udm2" in path.name else "analytic" if "AnalyticMS_SR" in path.name else None
        )
        if kind:
            by_scene[scene][kind] = path
    for scene, files in by_scene.items():
        if {"analytic", "udm2"} <= files.keys():
            cache.deliveries.setdefault(scene, []).append(
                {
                    "order_id": order_id,
                    "analytic": str(files["analytic"].relative_to(cache.root)),
                    "udm2": str(files["udm2"].relative_to(cache.root)),
                }
            )


def _collect_orders(
    cache: PlanetCache,
    result: PrefetchResult,
    *,
    wait: bool,
    timeout_minutes: float,
    poll_seconds: float,
) -> None:
    """Wait for open orders (up to the timeout) and download those that finish."""
    deadline = time.monotonic() + (timeout_minutes * 60 if wait else 0)
    open_orders = [
        oid for oid, o in cache.orders.items() if o["state"] not in _ORDER_DONE | _ORDER_FAILED
    ]
    while open_orders:
        still_open = []
        for order_id in open_orders:
            order = planet_api.get_order(order_id)
            cache.orders[order_id]["state"] = order["state"]
            if order["state"] in _ORDER_DONE:
                paths = planet_api.download_order(order, cache.root / order_id)
                _record_delivery(cache, order_id, paths)
                logger.info(f"Downloaded Planet order {order_id}")
            elif order["state"] in _ORDER_FAILED:
                result.failed_orders.append(order_id)
                logger.warning(f"Planet order {order_id} {order['state']}")
            else:
                still_open.append(order_id)
            cache.save()
        open_orders = still_open
        if not open_orders or time.monotonic() >= deadline:
            break
        logger.debug(f"Waiting for {len(open_orders)} Planet order(s)...")
        time.sleep(poll_seconds)
    result.pending_orders.extend(open_orders)


def prefetch_scenes(
    windows: list[ChipWindow],
    cache_root: Path,
    *,
    cloud_cover_max: float,
    nodata_max: float,
    candidates_per_chip: int = CANDIDATES_PER_CHIP,
    harmonize: bool = False,
    wait: bool = True,
    timeout_minutes: float = 120.0,
    poll_seconds: float = 30.0,
) -> PrefetchResult:
    """Make sure each window's candidate scenes are delivered to ``cache_root``.

    Returns, per (chip, season), the features of its candidates that have been
    delivered; a chip whose orders are still processing gets fewer (or none) and can
    be selected again on a later run.
    """
    cache = PlanetCache(cache_root)
    result = PrefetchResult()
    groups: dict[str, list[ChipWindow]] = defaultdict(list)
    for window in windows:
        groups[_group_key(window)].append(window)

    wanted: dict[tuple[str, str], list[dict]] = {}
    for group, members in groups.items():
        features = _search_group(members, cloud_cover_max)
        for f in features:
            cache.features[f["id"]] = f
        needed: set[str] = set()
        for window in members:
            picks = _chip_candidates(window, features, nodata_max, candidates_per_chip)
            wanted[(window.chip_id, window.season)] = picks
            needed.update(f["id"] for f in picks)
        to_order = sorted(needed - cache.ordered_for(group))
        logger.info(
            f"{group}: {len(members)} chip(s), {len(features)} scene(s) found, "
            f"{len(needed)} needed, {len(to_order)} to order"
        )
        if to_order:
            _submit_orders(cache, group, to_order, _clip_aoi(members), harmonize=harmonize)
    cache.save()

    _collect_orders(
        cache, result, wait=wait, timeout_minutes=timeout_minutes, poll_seconds=poll_seconds
    )

    for key, picks in wanted.items():
        result.candidates[key] = [f for f in picks if cache.deliveries.get(f["id"])]
    return result
