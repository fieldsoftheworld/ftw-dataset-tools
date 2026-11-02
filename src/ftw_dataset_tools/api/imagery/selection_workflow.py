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
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import pystac

from ftw_dataset_tools.api.imagery import parquet_search
from ftw_dataset_tools.api.imagery.catalog_ops import (
    SelectionConflictError,
    clear_chip_selections,
    has_mode_selection,
    iter_chip_dirs,
    mosaic_requested_year,
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
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from ftw_dataset_tools.api.imagery.mosaic_selection import MosaicSelectionResult
    from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult

logger = get_logger(__name__)

__all__ = [
    "ChipSelectionJob",
    "MultiModeSelectionResult",
    "NoCloudFreeScenesError",
    "SelectionWorkflowResult",
    "SkipCode",
    "find_chip_items",
    "require_mosaic_years",
    "resolve_selection_conflicts",
    "run_chip_selection",
    "select_imagery_modes_for_catalog",
    "selection_year",
    "unavailable_mosaics_reason",
]


class NoCloudFreeScenesError(ValueError):
    """A chip got no imagery (no cloud-free scene or complete mosaic year) and the run
    was asked to fail on that.
    """

    def __init__(self, chip_id: str, reason: str | None) -> None:
        super().__init__(f"No imagery for {chip_id}: {reason}")
        self.chip_id = chip_id


class SkipCode(StrEnum):
    """Why a chip got no selection in a mode; the ``code`` of every skipped detail."""

    NO_BBOX = "no_bbox"
    ALREADY_SELECTED = "already_selected"
    NO_YEAR = "no_year"
    NO_MOSAICS_FOR_YEAR = "no_mosaics_for_year"
    NO_IMAGERY = "no_imagery"


NO_YEAR_REASON = "No year provided and could not extract from chip ID or item properties"


def selection_year(item: pystac.Item, imagery_mode: str, year: int | None = None) -> int | None:
    """The year a chip is selected for in ``imagery_mode``, used by every check.

    Read from stable inputs only, never the temporal properties, which selection
    itself rewrites: the run's ``year``; else the year this mode's selection on the
    chip recorded (``ftw:calendar_year`` for scenes, the mosaic requested year for
    mosaics), so a run without a year never moves an existing selection; else the
    chip id's ``_YYYY`` suffix. With none of those the chip gets None and is skipped.
    """
    if year is not None:
        return year
    if imagery_mode == "scenes":
        recorded = item.properties.get("ftw:calendar_year")
    else:
        recorded = mosaic_requested_year(item)
    if recorded is not None:
        return recorded
    match = re.search(r"_(\d{4})$", item.id)
    return int(match.group(1)) if match else None


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
                footprint=job.item.geometry,
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
            footprint=job.item.geometry,
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
    items: list[pystac.Item],
    *,
    imagery_modes: str | Sequence[str],
    year: int | None,
    force: bool,
) -> int:
    """Stop before chips holding a mode's imagery for another year get mixed in.

    Every mode is checked before any chip is cleared, so a conflict in one mode
    stops the run before work in another. Another mode's selection on the same chip
    is no conflict: both modes coexist. Without ``force`` raises
    SelectionConflictError; with it, clears the conflicting mode's selection from
    those chips so they are selected afresh, leaving any other mode's imagery in
    place. Returns how many chip selections were cleared.
    """
    conflicts = [
        (item, mode, reason)
        for mode in normalize_imagery_modes(imagery_modes)
        for item in items
        if (reason := selection_conflict(item, mode, selection_year(item, mode, year)))
    ]
    if not conflicts:
        return 0
    if not force:
        item, mode, reason = conflicts[0]
        wanted = f"{mode} for {selection_year(item, mode, year)}"
        raise SelectionConflictError(
            f"{len(conflicts)} chip(s) already hold imagery from a different run "
            f"(e.g. {item.id} {reason}; this run selects {wanted})."
        )
    for item, mode, _reason in conflicts:
        clear_chip_selections(item, modes=(mode,))
    return len(conflicts)


def _select_mode(
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

    One mode's pass of :func:`select_imagery_modes_for_catalog`, which the commands
    and the pipeline call. It does not check selection conflicts; that function
    resolves them once, for every mode, before any pass starts.

    Args:
        catalog_dir: Path to the chips collection directory
        year: Year for every chip, or None for each chip's own (see `selection_year`)
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

    jobs: list[ChipSelectionJob] = []
    for item, item_path in chip_items:
        job_year = selection_year(item, imagery_mode, year)
        skip = _skip_reason(item, force=force, year=job_year, imagery_mode=imagery_mode)
        if skip is not None:
            result.skipped += 1
            result.skipped_details.append({"chip": item.id, "code": skip[0], "reason": skip[1]})
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

    ``unreadable`` lists chips whose JSON could not be parsed;
    they are read once for the whole run, so they are counted once, not per mode.
    The count properties sum over the modes that ran plus the unreadable chips, so
    the result reads like a single SelectionWorkflowResult where only totals matter.
    """

    by_mode: dict[str, SelectionWorkflowResult] = field(default_factory=dict)
    unreadable: list[dict] = field(default_factory=list)

    @property
    def successful(self) -> int:
        return sum(r.successful for r in self.by_mode.values())

    @property
    def skipped(self) -> int:
        return sum(r.skipped for r in self.by_mode.values())

    @property
    def failed(self) -> int:
        return len(self.unreadable) + sum(r.failed for r in self.by_mode.values())

    @property
    def skipped_details(self) -> list[dict]:
        return [{**d, "mode": m} for m, r in self.by_mode.items() for d in r.skipped_details]

    @property
    def failed_details(self) -> list[dict]:
        per_mode = [{**d, "mode": m} for m, r in self.by_mode.items() for d in r.failed_details]
        return [*self.unreadable, *per_mode]


def unavailable_mosaics_reason(year: int) -> str | None:
    """Why mosaics cannot be selected for ``year``, or None when its mosaics exist."""
    try:
        check_mosaic_year(year)
    except MosaicYearError:
        return f"Mosaics for {year} are not available (its quarterly index is incomplete)"
    return None


def require_mosaic_years(years: Iterable[int], imagery_modes: Sequence[str]) -> list[int]:
    """The one rule for a year with no mosaics; returns those of ``years`` without them.

    With mosaics as the only mode that is an error (MosaicYearError), raised before
    any work. Alongside scenes the chips of such a year skip their mosaics, each
    with a recorded reason, and still get scenes.
    """
    if "mosaics" not in imagery_modes:
        return []
    missing = [y for y in sorted(set(years)) if unavailable_mosaics_reason(y) is not None]
    if missing and tuple(imagery_modes) == ("mosaics",):
        raise MosaicYearError(missing[0])
    return missing


def select_imagery_modes_for_catalog(
    catalog_dir: Path,
    imagery_modes: str | Sequence[str],
    *,
    year: int | None,
    force: bool = False,
    chip_items: list[tuple[pystac.Item, Path]] | None = None,
    **options: Any,
) -> MultiModeSelectionResult:
    """Select imagery for every chip in each of ``imagery_modes``, scenes first.

    Each mode is one :func:`_select_mode` pass over the same chip items,
    read once, and the modes write disjoint slots, links and properties, so a chip
    ends up holding both, for the one year :func:`selection_year` gives it.

    Everything that can stop the run is checked for every mode before any pass:
    mosaic availability (:func:`require_mosaic_years`) and selection conflicts,
    which are resolved here and only here.

    Args:
        catalog_dir: Path to the chips collection directory
        imagery_modes: "scenes", "mosaics", or both
        year: Year for every chip, or None for each chip's own (see `selection_year`)
        force: Replace a conflicting or existing selection in the mode being run
        chip_items: (item, item_path) pairs to select for instead of every chip
                    found under ``catalog_dir``
        **options: Passed on to :func:`_select_mode`

    Raises:
        MosaicYearError: If mosaics are the only mode and a chip's year has none
        SelectionConflictError: If any mode conflicts and ``force`` is False
    """
    modes = normalize_imagery_modes(imagery_modes)
    result = MultiModeSelectionResult()
    if chip_items is None:
        chip_items = find_chip_items(catalog_dir, unreadable=result.unreadable)
    items = [item for item, _path in chip_items]

    mosaic_years = (selection_year(item, "mosaics", year) for item in items)
    for missing in require_mosaic_years(mosaic_years, modes):
        logger.warning(f"Mosaics for {missing} are not available; its chips get scenes only.")

    cleared = resolve_selection_conflicts(items, imagery_modes=modes, year=year, force=force)
    if cleared:
        logger.info(f"Cleared imagery from a different run: {cleared}")

    for mode in modes:
        if len(modes) > 1:
            logger.info(f"Selecting {mode}...")
        result.by_mode[mode] = _select_mode(
            catalog_dir=catalog_dir,
            year=year,
            imagery_mode=mode,
            force=force,
            chip_items=chip_items,
            **options,
        )
    return result


def _skip_reason(
    item: pystac.Item, *, force: bool, year: int | None, imagery_mode: str = DEFAULT_IMAGERY_MODE
) -> tuple[SkipCode, str] | None:
    """Return why this chip needs no selection in ``imagery_mode``, or None if it does."""
    if not item.bbox:
        return SkipCode.NO_BBOX, "No bbox in item"
    if not force and has_mode_selection(item, imagery_mode):
        return SkipCode.ALREADY_SELECTED, "Already has imagery selections"
    if year is None:
        return SkipCode.NO_YEAR, NO_YEAR_REASON
    no_mosaics = unavailable_mosaics_reason(year) if imagery_mode == "mosaics" else None
    return None if no_mosaics is None else (SkipCode.NO_MOSAICS_FOR_YEAR, no_mosaics)


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
            "code": SkipCode.NO_IMAGERY,
            "reason": selection.skipped_reason or "Unknown reason",
            "candidates_checked": selection.candidates_checked,
        }
    )
    progress.mark_skipped(selection.skipped_reason or "No cloud-free scenes")
