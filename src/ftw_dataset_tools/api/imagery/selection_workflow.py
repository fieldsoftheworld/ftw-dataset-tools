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
from typing import TYPE_CHECKING, Literal

import pystac

from ftw_dataset_tools.api.imagery.catalog_ops import has_existing_scenes, iter_chip_dirs
from ftw_dataset_tools.api.imagery.crop_calendar import ensure_crop_calendar_exists
from ftw_dataset_tools.api.imagery.parallel import (
    DEFAULT_WORKERS,
    ParallelOutcome,
    run_in_parallel,
)
from ftw_dataset_tools.api.imagery.progress import ImageryProgressBar
from ftw_dataset_tools.api.imagery.scene_selection import select_scenes_for_chip
from ftw_dataset_tools.api.imagery.stac_child_items import create_child_items_from_selection

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult

__all__ = [
    "ChipSelectionJob",
    "NoCloudFreeScenesError",
    "SelectionWorkflowResult",
    "chip_year",
    "find_chip_items",
    "run_chip_selection",
    "select_imagery_for_catalog",
]

NO_YEAR_REASON = "No year provided and could not extract from chip ID or item properties"


class NoCloudFreeScenesError(ValueError):
    """A chip has no cloud-free scenes and the run was asked to fail on that."""

    def __init__(self, chip_id: str, reason: str | None) -> None:
        super().__init__(f"No cloud-free scenes for {chip_id}: {reason}")
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

    ``logs`` collects the progress lines the selection produced so the caller can
    replay them in one block; printing them from the worker would interleave the
    chips running alongside it.
    """

    item: pystac.Item
    item_path: Path
    year: int
    logs: list[str] = field(default_factory=list)


def run_chip_selection(
    job: ChipSelectionJob,
    *,
    cloud_cover_chip: float,
    nodata_max: float,
    buffer_days: int,
    num_buffer_expansions: int,
    buffer_expansion_size: int,
    search_backend: str = "parquet",
) -> SceneSelectionResult:
    """Select scenes for one chip and write its child items.

    Safe to run on a worker thread: every chip queries the scene catalog on its
    own and writes only into its own directory.
    """
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
        on_progress=job.logs.append,
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
    (those ending in _planting_s2 or _harvest_s2).

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
            # Skip child items (they have _planting_s2 or _harvest_s2 suffix)
            if "_planting_s2" in json_file.name or "_harvest_s2" in json_file.name:
                continue
            try:
                item = pystac.Item.from_file(str(json_file))
            except Exception as e:
                if unreadable is not None:
                    unreadable.append({"chip": json_file.stem, "error": f"Unreadable chip: {e}"})
                continue
            chip_items.append((item, json_file))

    return chip_items


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
    verbose: bool = False,
    workers: int = DEFAULT_WORKERS,
    search_backend: str = "parquet",
    chip_items: list[tuple[pystac.Item, Path]] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> SelectionWorkflowResult:
    """Select imagery for all chips in a catalog.

    This is the core orchestration function used by the `select-images` and
    `create-dataset` commands and the `ftwd run` pipeline.

    Args:
        catalog_dir: Path to the chips collection directory
        year: Calendar year for the crop cycle, or None to take each chip's
              year from its id or datetime (see `chip_year`)
        cloud_cover_chip: Maximum chip-level cloud cover percentage (0-100)
        nodata_max: Maximum nodata percentage (0-100). Default 0 rejects any nodata.
        buffer_days: Days to search around crop calendar dates
        num_buffer_expansions: Number of times to expand search window
        buffer_expansion_size: Days to add to buffer on each expansion
        force: If True, overwrite existing selections. If False, skip chips with scenes.
        on_missing: How to handle chips with no cloud-free scenes:
                    - "skip": Skip and record in skipped_details
                    - "fail": Raise exception
        verbose: If True, show detailed STAC query information
        workers: Number of chips to select for concurrently
        search_backend: "parquet" (the STAC-GeoParquet mirror, default) or
                        "earth-search" (the Earth Search STAC API)
        chip_items: (item, item_path) pairs to select for instead of every chip
                    found under `catalog_dir`
        on_progress: Optional callback for crop calendar download messages

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

    jobs: list[ChipSelectionJob] = []
    for item, item_path in chip_items:
        job_year = year if year is not None else chip_year(item)
        reason = _skip_reason(item, force=force, year=job_year)
        if reason is not None:
            result.skipped += 1
            result.skipped_details.append({"chip": item.id, "reason": reason})
            continue
        jobs.append(ChipSelectionJob(item=item, item_path=item_path, year=job_year))

    if not jobs:
        return result

    # Warm the crop calendar before fanning out. Every chip needs it, and the
    # first-time download must happen once rather than from every worker at once.
    ensure_crop_calendar_exists(on_progress=on_progress)

    _run_selection(
        jobs,
        result=result,
        cloud_cover_chip=cloud_cover_chip,
        nodata_max=nodata_max,
        buffer_days=buffer_days,
        num_buffer_expansions=num_buffer_expansions,
        buffer_expansion_size=buffer_expansion_size,
        on_missing=on_missing,
        verbose=verbose,
        workers=workers,
        search_backend=search_backend,
    )

    return result


def _skip_reason(item: pystac.Item, *, force: bool, year: int | None) -> str | None:
    """Return why this chip needs no selection, or None if it does."""
    if not item.bbox:
        return "No bbox in item"
    if not force and has_existing_scenes(item):
        return "Already has imagery selections"
    if year is None:
        return NO_YEAR_REASON
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
    verbose: bool,
    workers: int,
    search_backend: str = "parquet",
) -> None:
    """Select scenes for every chip on a thread pool, recording outcomes as they finish.

    Only this function touches the counters and the progress bar, so a chip's log
    lines stay in one block instead of interleaving with the rest of the pool.
    """

    def work(job: ChipSelectionJob) -> SceneSelectionResult:
        return run_chip_selection(
            job,
            cloud_cover_chip=cloud_cover_chip,
            nodata_max=nodata_max,
            buffer_days=buffer_days,
            num_buffer_expansions=num_buffer_expansions,
            buffer_expansion_size=buffer_expansion_size,
            search_backend=search_backend,
        )

    with ImageryProgressBar(total=len(jobs), leave=False, verbose=verbose) as progress:

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
    for message in job.logs:
        progress.on_progress(message)

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
