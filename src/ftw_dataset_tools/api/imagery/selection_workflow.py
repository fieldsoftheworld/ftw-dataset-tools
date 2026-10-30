"""Image selection orchestration for STAC catalogs.

This module provides workflow functions for selecting imagery across all chips
in a catalog. It is used by the `select-images` and `create-dataset` commands and
the `ftwd run` pipeline, so selection output is identical across all three paths.

Chips are processed concurrently: a chip is several STAC searches' worth of
network wait, and no two chips touch the same files. Only the calling thread
reads the results, so the counters and the progress display never race.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import pystac

from ftw_dataset_tools.api.imagery import parquet_search
from ftw_dataset_tools.api.imagery.catalog_ops import (
    SelectionConflictError,
    clear_chip_selections,
    has_mode_selection,
    iter_chip_dirs,
    selection_conflict,
)
from ftw_dataset_tools.api.imagery.crop_calendar import ensure_crop_calendar_exists
from ftw_dataset_tools.api.imagery.mosaic_selection import (
    MosaicYearError,
    check_mosaic_year,
    select_mosaics_for_chip,
)
from ftw_dataset_tools.api.imagery.parallel import (
    DEFAULT_WORKERS,
    ParallelOutcome,
    run_in_parallel,
)
from ftw_dataset_tools.api.imagery.progress import ImageryProgressBar
from ftw_dataset_tools.api.imagery.scene_selection import select_scenes_for_chip
from ftw_dataset_tools.api.imagery.slots import (
    DEFAULT_IMAGERY_MODE,
    normalize_imagery_modes,
    parse_child_id,
)
from ftw_dataset_tools.api.imagery.stac_child_items import (
    create_child_items_from_selection,
    create_mosaic_child_items,
)
from ftw_dataset_tools.api.logging_config import capture_logs, get_logger

if TYPE_CHECKING:
    import logging
    from collections.abc import Sequence
    from pathlib import Path

    from ftw_dataset_tools.api.imagery.mosaic_selection import MosaicSelectionResult
    from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult

logger = get_logger(__name__)

__all__ = [
    "ChipSelectionJob",
    "MultiModeSelectionResult",
    "NoCloudFreeScenesError",
    "SelectionWorkflowResult",
    "chip_year",
    "find_chip_items",
    "resolve_selection_conflicts",
    "run_chip_selection",
    "select_imagery_for_catalog",
    "select_imagery_modes_for_catalog",
    "unavailable_mosaics_reason",
]

NO_YEAR_REASON = "No year provided and could not extract from chip ID or item properties"


class NoCloudFreeScenesError(ValueError):
    """A chip got no imagery (no cloud-free scene or complete mosaic year) and the run
    was asked to fail on that.
    """

    def __init__(self, chip_id: str, reason: str | None) -> None:
        super().__init__(f"No imagery for {chip_id}: {reason}")
        self.chip_id = chip_id


def chip_year(item: pystac.Item) -> int | None:
    """Year of a chip's crop cycle, from its id suffix (``_2024``) or its datetime."""
    match = re.search(r"_(\d{4})$", item.id)
    if match:
        return int(match.group(1))

    start = item.properties.get("start_datetime")
    if isinstance(start, str):
        try:
            return datetime.fromisoformat(start.replace("Z", "+00:00")).year
        except ValueError:
            pass

    return item.datetime.year if item.datetime else None


@dataclass
class ChipSelectionJob:
    """One chip's selection work: its item, where it lives, and the year to select for.

    ``logs`` collects the log records the selection produced so the caller can
    replay them in one block; emitting them from the worker would interleave the
    chips running alongside it.
    """

    item: pystac.Item
    item_path: Path
    year: int
    logs: list[logging.LogRecord] = field(default_factory=list)


def run_chip_selection(
    job: ChipSelectionJob,
    *,
    cloud_cover_chip: float,
    nodata_max: float,
    buffer_days: int,
    num_buffer_expansions: int,
    buffer_expansion_size: int,
    search_backend: str = "parquet",
    imagery_mode: str = DEFAULT_IMAGERY_MODE,
) -> SceneSelectionResult | MosaicSelectionResult:
    """Select imagery for one chip and write its child items.

    Safe to run on a worker thread: every chip queries the scene catalog on its
    own and writes only into its own directory.
    """
    with capture_logs(job.logs):
        if imagery_mode == "mosaics":
            mosaics = select_mosaics_for_chip(
                chip_id=job.item.id,
                bbox=tuple(job.item.bbox),
                year=job.year,
                nodata_max=nodata_max,
            )
            if mosaics.success:
                create_mosaic_child_items(job.item_path.parent, job.item, mosaics)
            return mosaics

        selection = select_scenes_for_chip(
            chip_id=job.item.id,
            bbox=tuple(job.item.bbox),
            year=job.year,
            cloud_cover_chip=cloud_cover_chip,
            nodata_max=nodata_max,
            buffer_days=buffer_days,
            num_buffer_expansions=num_buffer_expansions,
            buffer_expansion_size=buffer_expansion_size,
            search_backend=search_backend,
        )

        if selection.success:
            create_child_items_from_selection(
                chip_dir=job.item_path.parent,
                parent_item=job.item,
                result=selection,
                year=job.year,
                cloud_cover_chip=cloud_cover_chip,
                buffer_days=buffer_days,
                num_buffer_expansions=num_buffer_expansions,
                buffer_expansion_size=buffer_expansion_size,
            )

    return selection


@dataclass
class SelectionWorkflowResult:
    """Result of running image selection across a catalog."""

    successful: int = 0
    skipped: int = 0
    failed: int = 0
    skipped_details: list[dict] = field(default_factory=list)
    failed_details: list[dict] = field(default_factory=list)


def find_chip_items(
    catalog_dir: Path,
    unreadable: list[dict] | None = None,
) -> list[tuple[pystac.Item, Path]]:
    """Find all parent chip items in a catalog directory.

    Searches subdirectories for STAC item JSON files, excluding child S2 items
    (``{chip}_{slot}_s2``).

    Args:
        catalog_dir: Path to the collection directory (holding collection.json),
                     whose ``chips/<square>/<item_id>/`` subdirectories hold STAC
                     item files
        unreadable: Optional list that collects ``{"chip": ..., "error": ...}``
                    entries for chip files that could not be parsed. A chip that
                    cannot be read is otherwise invisible: it never gets imagery
                    and is counted nowhere, so pass this in to report it.

    Returns:
        List of (pystac.Item, item_path) tuples for each parent chip item found.
        Returns empty list if no items found.
    """
    chip_items = []

    for subdir in iter_chip_dirs(catalog_dir):
        for json_file in subdir.glob("*.json"):
            if parse_child_id(json_file.stem) is not None:
                continue
            try:
                item = pystac.Item.from_file(str(json_file))
            except Exception as e:
                if unreadable is not None:
                    unreadable.append({"chip": json_file.stem, "error": f"Unreadable chip: {e}"})
                continue
            chip_items.append((item, json_file))

    return chip_items


def resolve_selection_conflicts(
    items: list[pystac.Item], *, imagery_mode: str, year: int | None, force: bool
) -> int:
    """Stop before chips holding this mode's imagery for another year get mixed in.

    Another mode's selection on the same chip is no conflict: both modes coexist.
    Without ``force`` raises SelectionConflictError; with it, clears this mode's
    selection from those chips so they are selected afresh, leaving any other mode's
    imagery in place. Returns how many chips were cleared.
    """
    conflicts = [
        (item, reason)
        for item in items
        if (reason := selection_conflict(item, imagery_mode, year)) is not None
    ]
    if not conflicts:
        return 0
    if not force:
        item, reason = conflicts[0]
        wanted = f"mosaics for {year}" if imagery_mode == "mosaics" else "scenes"
        raise SelectionConflictError(
            f"{len(conflicts)} chip(s) already hold imagery from a different run "
            f"(e.g. {item.id} {reason}; this run selects {wanted})."
        )
    for item, _reason in conflicts:
        clear_chip_selections(item, modes=(imagery_mode,))
    return len(conflicts)


def select_imagery_for_catalog(
    catalog_dir: Path,
    year: int | None,
    cloud_cover_chip: float = 2.0,
    nodata_max: float = 0.0,
    buffer_days: int = 14,
    num_buffer_expansions: int = 3,
    buffer_expansion_size: int = 14,
    force: bool = False,
    on_missing: Literal["skip", "fail"] = "skip",
    workers: int = DEFAULT_WORKERS,
    search_backend: str = "parquet",
    imagery_mode: str = DEFAULT_IMAGERY_MODE,
    chip_items: list[tuple[pystac.Item, Path]] | None = None,
) -> SelectionWorkflowResult:
    """Select imagery for all chips in a catalog.

    This is the core orchestration function used by the `select-images` and
    `create-dataset` commands and the `ftwd run` pipeline.

    Args:
        catalog_dir: Path to the chips collection directory
        year: Calendar year for the crop cycle (the mosaic year in mosaic mode), or
              None to take each chip's year from its id or datetime (see `chip_year`)
        cloud_cover_chip: Maximum chip-level cloud cover percentage (0-100)
        nodata_max: Maximum nodata percentage (0-100). Default 0 rejects any nodata.
        buffer_days: Days to search around crop calendar dates
        num_buffer_expansions: Number of times to expand search window
        buffer_expansion_size: Days to add to buffer on each expansion
        force: If True, overwrite existing selections. If False, skip chips with scenes.
        on_missing: How to handle chips with no cloud-free scenes:
                    - "skip": Skip and record in skipped_details
                    - "fail": Raise exception
        workers: Number of chips to select for concurrently
        search_backend: "parquet" (the STAC-GeoParquet mirror, default) or
                        "earth-search" (the Earth Search STAC API)
        imagery_mode: "scenes" (planting/harvest) or "mosaics" (Q1-Q4)
        chip_items: (item, item_path) pairs to select for instead of every chip
                    found under `catalog_dir`

    Returns:
        SelectionWorkflowResult with success/skipped/failed counts and details

    Raises:
        NoCloudFreeScenesError: If on_missing="fail" and no cloud-free scenes found
        Exception: If on_missing="fail" and a chip's selection raised
        SelectionConflictError: If chips hold another mode's or year's imagery
            and ``force`` is False
    """
    result = SelectionWorkflowResult()

    if chip_items is None:
        # Chips whose JSON cannot be read are reported as failures rather than
        # silently dropped from the run.
        unreadable: list[dict] = []
        chip_items = find_chip_items(catalog_dir, unreadable=unreadable)
        result.failed += len(unreadable)
        result.failed_details.extend(unreadable)

    if not chip_items:
        return result

    cleared = resolve_selection_conflicts(
        [item for item, _path in chip_items], imagery_mode=imagery_mode, year=year, force=force
    )
    if cleared:
        logger.info(f"Cleared imagery from a different run: {cleared}")

    jobs: list[ChipSelectionJob] = []
    for item, item_path in chip_items:
        job_year = year if year is not None else chip_year(item)
        reason = _skip_reason(item, force=force, year=job_year, imagery_mode=imagery_mode)
        if reason is not None:
            result.skipped += 1
            result.skipped_details.append({"chip": item.id, "reason": reason})
            continue
        jobs.append(ChipSelectionJob(item=item, item_path=item_path, year=job_year))

    if not jobs:
        return result

    # Warm the crop calendar before fanning out. Every chip needs it, and the
    # first-time download must happen once rather than from every worker at once.
    # Mosaics need neither the crop calendar nor the scene catalog.
    index = None
    if imagery_mode != "mosaics":
        ensure_crop_calendar_exists()
        index = _build_scene_index(jobs, search_backend)

    with parquet_search.local_index(index):
        _run_selection(
            jobs,
            result=result,
            cloud_cover_chip=cloud_cover_chip,
            nodata_max=nodata_max,
            buffer_days=buffer_days,
            num_buffer_expansions=num_buffer_expansions,
            buffer_expansion_size=buffer_expansion_size,
            on_missing=on_missing,
            workers=workers,
            search_backend=search_backend,
            imagery_mode=imagery_mode,
        )

    return result


@dataclass
class MultiModeSelectionResult:
    """Selection results of every imagery mode run into one catalog.

    ``skipped_modes`` maps a mode that did not run to the reason, e.g. mosaics for a
    year that has none. The count properties sum over the modes that ran, so the
    result reads like a single SelectionWorkflowResult where only totals matter.
    """

    by_mode: dict[str, SelectionWorkflowResult] = field(default_factory=dict)
    skipped_modes: dict[str, str] = field(default_factory=dict)

    @property
    def successful(self) -> int:
        return sum(r.successful for r in self.by_mode.values())

    @property
    def skipped(self) -> int:
        return sum(r.skipped for r in self.by_mode.values())

    @property
    def failed(self) -> int:
        return sum(r.failed for r in self.by_mode.values())

    @property
    def skipped_details(self) -> list[dict]:
        return [{**d, "mode": m} for m, r in self.by_mode.items() for d in r.skipped_details]

    @property
    def failed_details(self) -> list[dict]:
        return [{**d, "mode": m} for m, r in self.by_mode.items() for d in r.failed_details]


def unavailable_mosaics_reason(year: int) -> str | None:
    """Why mosaics cannot be selected for ``year``, or None when its mosaics exist."""
    try:
        check_mosaic_year(year)
    except MosaicYearError:
        return (
            f"Mosaics for {year} are not available (its quarterly index is incomplete), "
            "so the mosaics imagery mode was skipped."
        )
    return None


def select_imagery_modes_for_catalog(
    catalog_dir: Path,
    imagery_modes: Sequence[str],
    *,
    year: int | None,
    mosaic_year: int | None = None,
    **options: Any,
) -> MultiModeSelectionResult:
    """Select imagery for every chip in each of ``imagery_modes``, scenes first.

    Each mode is one :func:`select_imagery_for_catalog` run, and the modes write
    disjoint slots, links and properties, so a chip ends up holding both.

    Mosaics for a year that has none are skipped with a warning (and recorded in
    ``skipped_modes``) when another mode is configured, so the scenes still run; as
    the only mode they raise MosaicYearError, since the run could select nothing.

    Args:
        catalog_dir: Path to the chips collection directory
        imagery_modes: "scenes", "mosaics", or both
        year: Scene year, or None to take each chip's year from its id
        mosaic_year: Mosaic year; defaults to ``year``
        **options: Passed on to :func:`select_imagery_for_catalog`

    Raises:
        MosaicYearError: If mosaics are the only mode and ``mosaic_year`` has none
        SelectionConflictError: As :func:`select_imagery_for_catalog`, per mode
    """
    modes = normalize_imagery_modes(list(imagery_modes))
    result = MultiModeSelectionResult()
    for mode in modes:
        mode_year = year
        if mode == "mosaics" and mosaic_year is not None:
            mode_year = mosaic_year
        if mode == "mosaics" and mode_year is not None:
            reason = unavailable_mosaics_reason(mode_year)
            if reason is not None:
                if len(modes) == 1:
                    raise MosaicYearError(mode_year)
                logger.warning(reason)
                result.skipped_modes[mode] = reason
                continue
        if len(modes) > 1:
            logger.info(f"Selecting {mode}...")
        result.by_mode[mode] = select_imagery_for_catalog(
            catalog_dir=catalog_dir, year=mode_year, imagery_mode=mode, **options
        )
    return result


def _skip_reason(
    item: pystac.Item, *, force: bool, year: int | None, imagery_mode: str = DEFAULT_IMAGERY_MODE
) -> str | None:
    """Return why this chip needs no selection in ``imagery_mode``, or None if it does."""
    if not item.bbox:
        return "No bbox in item"
    if not force and has_mode_selection(item, imagery_mode):
        return "Already has imagery selections"
    if year is None:
        return NO_YEAR_REASON
    return None


def _build_scene_index(
    jobs: list[ChipSelectionJob],
    search_backend: str,
) -> Path | None:
    """Pull the run's slice of the scene catalog local before fanning out.

    Every chip searches the same few MGRS tiles, so the whole run's candidate
    scenes are a file of a few hundred kilobytes. Fetching it once turns each
    chip's searches from remote reads of a multi-gigabyte part file into local
    ones. Returns ``None`` when there is nothing to index or the build fails -
    both leave the workers querying the mirror directly, which still works.
    """
    if search_backend != "parquet":
        return None

    boxes = [job.item.bbox for job in jobs if job.item.bbox]
    if not boxes:
        return None
    bbox = (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )

    try:
        # Chips can carry different years, so the index covers each one's reach.
        years = {y for job in jobs for y in parquet_search.index_years(job.year)}
        return parquet_search.build_local_index(bbox, sorted(years))
    except Exception as e:
        # An unbuildable index is a performance problem, not a correctness one.
        logger.debug(f"Scene index unavailable ({e}); querying the catalog directly.")
        return None


def _run_selection(
    jobs: list[ChipSelectionJob],
    *,
    result: SelectionWorkflowResult,
    cloud_cover_chip: float,
    nodata_max: float,
    buffer_days: int,
    num_buffer_expansions: int,
    buffer_expansion_size: int,
    on_missing: Literal["skip", "fail"],
    workers: int,
    search_backend: str = "parquet",
    imagery_mode: str = DEFAULT_IMAGERY_MODE,
) -> None:
    """Select scenes for every chip on a thread pool, recording outcomes as they finish.

    Only this function touches the counters and the progress bar, so a chip's log
    lines stay in one block instead of interleaving with the rest of the pool.
    """

    def work(job: ChipSelectionJob) -> SceneSelectionResult | MosaicSelectionResult:
        return run_chip_selection(
            job,
            cloud_cover_chip=cloud_cover_chip,
            nodata_max=nodata_max,
            buffer_days=buffer_days,
            num_buffer_expansions=num_buffer_expansions,
            buffer_expansion_size=buffer_expansion_size,
            search_backend=search_backend,
            imagery_mode=imagery_mode,
        )

    with ImageryProgressBar(total=len(jobs), leave=False) as progress:

        def apply(outcome: ParallelOutcome[ChipSelectionJob, SceneSelectionResult]) -> None:
            _record_chip(outcome, result=result, progress=progress, on_missing=on_missing)

        run_in_parallel(jobs, work=work, apply=apply, workers=workers)

        # Per-chip exceptions are swallowed below to keep the run going; without
        # this the operator only ever sees a count.
        progress.report_failures(result.failed_details)


def _record_chip(
    outcome: ParallelOutcome[ChipSelectionJob, SceneSelectionResult],
    *,
    result: SelectionWorkflowResult,
    progress: ImageryProgressBar,
    on_missing: Literal["skip", "fail"],
) -> None:
    """Fold one chip's outcome into the counters and the progress bar (calling thread only)."""
    job = outcome.task
    progress.start_chip(job.item.id)
    progress.show(job.logs)

    if outcome.error is not None:
        if on_missing == "fail":
            raise outcome.error
        result.failed += 1
        result.failed_details.append({"chip": job.item.id, "error": str(outcome.error)})
        progress.mark_failed(str(outcome.error))
        return

    selection = outcome.value
    if selection is None:  # pragma: no cover - a worker returns a result or raises
        return

    if selection.success:
        result.successful += 1
        progress.mark_success(selection)
        return

    if on_missing == "fail":
        raise NoCloudFreeScenesError(job.item.id, selection.skipped_reason)

    result.skipped += 1
    result.skipped_details.append(
        {
            "chip": job.item.id,
            "reason": selection.skipped_reason or "Unknown reason",
            "candidates_checked": selection.candidates_checked,
        }
    )
    progress.mark_skipped(selection.skipped_reason or "No cloud-free scenes")
