"""Convert a catalog's JPEG chip previews to WebP.

Previews are written as WebP, but a catalog built before that switch still has
``.jpg`` files on disk and ``.jpg`` hrefs in its items. This re-renders each preview
from the imagery it was made from, repoints the chip item and its season children at
the WebP, and only then removes the superseded ``.jpg``.

Rendering uses source imagery instead of transcoding the legacy JPEG.
Overlay composition still re-encodes its WebP base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

import pystac
from PIL import Image
from tqdm import tqdm

from ftw_dataset_tools.api.assets import add_file_info
from ftw_dataset_tools.api.imagery.parallel import DEFAULT_WORKERS, run_in_parallel
from ftw_dataset_tools.api.imagery.preview_workflow import (
    PreviewTask,
    build_preview_task,
    render_preview,
)
from ftw_dataset_tools.api.imagery.selection_workflow import find_chip_items
from ftw_dataset_tools.api.imagery.stac_child_items import SEASONS, attach_thumbnail_to_parent
from ftw_dataset_tools.api.imagery.thumbnails import (
    LEGACY_PREVIEW_SUFFIXES,
    PREVIEW_MEDIA_TYPE,
    PREVIEW_SUFFIX,
    REFERENCE_MASK_SUFFIX,
    generate_overlay_thumbnail,
    generate_thumbnail,
)
from ftw_dataset_tools.api.stac_items import write_item

if TYPE_CHECKING:
    from collections.abc import Callable

    from ftw_dataset_tools.api.imagery.parallel import ParallelOutcome

__all__ = [
    "ConversionResult",
    "conversion_summary_line",
    "convert_previews_for_catalog",
]


@dataclass
class ConversionResult:
    """Counts and details for a conversion run."""

    chips_converted: int = 0
    previews_written: int = 0
    legacy_removed: int = 0
    skipped: int = 0
    failed: int = 0
    skipped_details: list[dict] = field(default_factory=list)
    failed_details: list[dict] = field(default_factory=list)


@dataclass
class _ConversionPlan:
    item: pystac.Item
    item_path: Path
    children: list[tuple[pystac.Item, Path]]
    legacy: list[Path]
    seasons: tuple[str, ...]
    overlay: bool
    remote_task: PreviewTask | None
    reusable: tuple[Path, ...]

    @property
    def outputs(self) -> tuple[Path, ...]:
        chip_dir = self.item_path.parent
        stems = [f"{self.item.id}_{season}_image_s2" for season in self.seasons]
        if self.overlay:
            stems.append(f"{self.item.id}_overlay")
        return tuple(chip_dir / f"{stem}{PREVIEW_SUFFIX}" for stem in stems)


def legacy_previews(chip_dir: Path, item_id: str) -> list[Path]:
    """Every JPEG preview this chip still has on disk.

    Args:
        chip_dir: Directory holding the chip's files
        item_id: Chip item id

    Returns:
        Paths to the chip's legacy previews, empty if it has none
    """
    stems = [f"{item_id}_overlay", *(f"{item_id}_{season}_image_s2" for season in SEASONS)]
    return [
        path
        for stem in stems
        for ext in LEGACY_PREVIEW_SUFFIXES
        if (path := chip_dir / f"{stem}{ext}").exists()
    ]


def _read_children(chip_dir: Path, item_id: str) -> list[tuple[pystac.Item, Path]]:
    children = []
    for season in SEASONS:
        path = chip_dir / f"{item_id}_{season}_s2.json"
        if path.exists():
            try:
                children.append((pystac.Item.from_file(str(path)), path))
            except Exception as err:
                raise OSError(f"Unreadable child {path.name}: {err}") from err
    return children


def _local_asset_path(asset: pystac.Asset, chip_dir: Path) -> Path | None:
    href = asset.href
    if Path(href).is_absolute():
        return Path(href).resolve()
    parsed = urlsplit(href)
    if parsed.scheme == "file":
        local = f"//{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path
        return Path(url2pathname(local)).resolve()
    if parsed.scheme or parsed.netloc:
        return None
    return (chip_dir / unquote(parsed.path)).resolve()


def _referenced_legacy(items: list[pystac.Item], chip_dir: Path, item_id: str) -> set[Path]:
    stems = [f"{item_id}_overlay", *(f"{item_id}_{s}_image_s2" for s in SEASONS)]
    expected = {
        (chip_dir / f"{stem}{ext}").resolve() for stem in stems for ext in LEGACY_PREVIEW_SUFFIXES
    }
    return {
        path
        for item in items
        if (asset := item.assets.get("thumbnail")) is not None
        if (path := _local_asset_path(asset, chip_dir)) in expected
    }


def _valid_webp(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            return image.format == "WEBP"
    except (OSError, ValueError):
        return False


def _plan_conversion(item: pystac.Item, item_path: Path) -> _ConversionPlan | str:
    chip_dir = item_path.parent
    children = _read_children(chip_dir, item.id)
    legacy = legacy_previews(chip_dir, item.id)
    referenced = _referenced_legacy([item, *(child for child, _ in children)], chip_dir, item.id)
    if not legacy and not referenced:
        return "No JPEG previews"

    seasons = tuple(
        season for season in SEASONS if (chip_dir / f"{item.id}_{season}_image_s2.tif").is_file()
    )
    remote_task = None
    overlay = (chip_dir / f"{item.id}{REFERENCE_MASK_SUFFIX}").is_file()
    if overlay and "planting" not in seasons:
        task = build_preview_task(item, item_path)
        remote_task = task if isinstance(task, PreviewTask) else None
        overlay = remote_task is not None

    plan = _ConversionPlan(item, item_path, children, legacy, seasons, overlay, remote_task, ())
    stems = [f"{item.id}_overlay", *(f"{item.id}_{s}_image_s2" for s in SEASONS)]
    existing = tuple(chip_dir / f"{stem}{PREVIEW_SUFFIX}" for stem in stems)
    plan.reusable = tuple(
        path for path in existing if path not in plan.outputs and _valid_webp(path)
    )
    if not plan.outputs and not plan.reusable:
        return "No imagery to re-render the preview from"
    for path in existing:
        if path.exists() and path not in (*plan.outputs, *plan.reusable):
            raise OSError(f"Invalid WebP without a render source: {path.name}")
    return plan


def render_chip_previews(plan: _ConversionPlan) -> tuple[Path, ...]:
    """Render the outputs selected by the same planner used for dry-run."""
    chip_dir = plan.item_path.parent
    for season in plan.seasons:
        stem = f"{plan.item.id}_{season}_image_s2"
        generate_thumbnail(chip_dir / f"{stem}.tif", chip_dir / f"{stem}{PREVIEW_SUFFIX}")
    if plan.overlay:
        if plan.remote_task is not None:
            render_preview(plan.remote_task)
        else:
            generate_overlay_thumbnail(
                chip_dir / f"{plan.item.id}_planting_image_s2{PREVIEW_SUFFIX}",
                chip_dir / f"{plan.item.id}{REFERENCE_MASK_SUFFIX}",
                chip_dir / f"{plan.item.id}_overlay{PREVIEW_SUFFIX}",
            )
    return plan.outputs


def _repoint_child_items(plan: _ConversionPlan) -> None:
    for child, child_path in plan.children:
        preview_path = _child_preview_path(child_path)
        if preview_path not in (*plan.outputs, *plan.reusable):
            continue
        thumbnail = child.assets.get("thumbnail")
        if thumbnail is None:
            continue
        checksums = "file:checksum" in thumbnail.extra_fields
        thumbnail.href = f"./{preview_path.name}"
        thumbnail.media_type = PREVIEW_MEDIA_TYPE
        thumbnail.title = "WebP preview"
        add_file_info(thumbnail, preview_path, checksum=checksums)
        write_item(child, child_path)


def _child_preview_path(child_path: Path) -> Path:
    return child_path.with_name(child_path.stem.removesuffix("_s2") + "_image_s2" + PREVIEW_SUFFIX)


def _removable_previews(plan: _ConversionPlan) -> list[Path]:
    available = (*plan.outputs, *plan.reusable)
    chip_dir = plan.item_path.parent
    referenced = set()
    for owner, path in [(plan.item, plan.item_path), *plan.children]:
        if path == plan.item_path:
            repointed = any(
                chip_dir / f"{plan.item.id}{stem}{PREVIEW_SUFFIX}" in available
                for stem in ("_overlay", "_planting_image_s2")
            )
        else:
            repointed = _child_preview_path(path) in available
        referenced.update(
            _local_asset_path(asset, chip_dir)
            for key, asset in owner.assets.items()
            if key != "thumbnail" or not repointed
        )
    return [
        path
        for path in plan.legacy
        if path.with_suffix(PREVIEW_SUFFIX) in available and path.resolve() not in referenced
    ]


def commit_chip_conversion(plan: _ConversionPlan) -> int:
    """Update every readable item before removing its superseded JPEGs."""
    item, item_path = plan.item, plan.item_path
    chip_dir = item_path.parent
    thumbnail = item.assets.get("thumbnail")
    checksums = thumbnail is not None and "file:checksum" in thumbnail.extra_fields
    attach_thumbnail_to_parent(item, chip_dir, checksums=checksums)
    write_item(item, item_path)
    _repoint_child_items(plan)

    removed = 0
    for path in _removable_previews(plan):
        path.unlink()
        removed += 1
    return removed


def convert_previews_for_catalog(
    catalog_dir: Path,
    *,
    dry_run: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    show_progress_bar: bool = True,
    workers: int = DEFAULT_WORKERS,
) -> ConversionResult:
    """Convert every JPEG chip preview in a catalog to WebP.

    Chips that already have only WebP previews are skipped, so a run is resumable
    and re-running over a converted catalog does nothing.

    Args:
        catalog_dir: The collection directory holding ``collection.json``.
        dry_run: Report what would be converted without writing or deleting.
        on_progress: Optional ``(done, total)`` callback.
        show_progress_bar: Show a tqdm bar.
        workers: Number of chips to render concurrently.

    Returns:
        ConversionResult with counts and per-chip details.
    """
    result = ConversionResult()

    unreadable: list[dict] = []
    chip_items = find_chip_items(catalog_dir, unreadable=unreadable)
    result.failed += len(unreadable)
    result.failed_details.extend(unreadable)
    if not chip_items:
        return result

    pending = _pending_conversions(chip_items, result, dry_run=dry_run)
    if dry_run or not pending:
        return result

    progress_bar = (
        tqdm(total=len(pending), desc="Converting previews", unit="chip", leave=False)
        if show_progress_bar
        else None
    )
    done = 0

    def advance() -> None:
        nonlocal done
        done += 1
        if progress_bar:
            progress_bar.update(1)
        if on_progress:
            on_progress(done, len(pending))

    def apply(outcome: ParallelOutcome[_ConversionPlan, tuple[Path, ...]]) -> None:
        plan = outcome.task
        item = plan.item
        if outcome.error is not None:
            result.failed += 1
            result.failed_details.append({"chip": item.id, "error": str(outcome.error)})
            advance()
            return
        if not outcome.value and not plan.reusable:
            result.skipped += 1
            result.skipped_details.append(
                {"chip": item.id, "reason": "No imagery to re-render the preview from"}
            )
            advance()
            return
        try:
            removed = commit_chip_conversion(plan)
        except (OSError, pystac.STACError) as err:
            result.failed += 1
            result.failed_details.append({"chip": item.id, "error": str(err)})
            advance()
            return
        result.chips_converted += 1
        result.previews_written += len(outcome.value or ())
        result.legacy_removed += removed
        advance()

    try:
        run_in_parallel(pending, render_chip_previews, apply, workers=workers)
    finally:
        if progress_bar:
            progress_bar.close()

    return result


def _pending_conversions(
    chip_items: list[tuple[pystac.Item, Path]],
    result: ConversionResult,
    *,
    dry_run: bool,
) -> list[_ConversionPlan]:
    """Plan source-backed work and report chips with nothing convertible."""
    pending = []
    for item, item_path in chip_items:
        try:
            plan = _plan_conversion(item, item_path)
        except OSError as err:
            result.failed += 1
            result.failed_details.append({"chip": item.id, "error": str(err)})
            continue
        if isinstance(plan, str):
            result.skipped += 1
            result.skipped_details.append({"chip": item.id, "reason": plan})
            continue
        if dry_run:
            result.chips_converted += 1
            result.previews_written += len(plan.outputs)
            result.legacy_removed += len(_removable_previews(plan))
            continue
        pending.append(plan)
    return pending


def conversion_summary_line(result: ConversionResult) -> str:
    """One-line summary for the run report."""
    return (
        f"{result.chips_converted} chips converted, "
        f"{result.previews_written} previews written, "
        f"{result.legacy_removed} JPEGs removed, "
        f"{result.skipped} skipped, {result.failed} failed"
    )
