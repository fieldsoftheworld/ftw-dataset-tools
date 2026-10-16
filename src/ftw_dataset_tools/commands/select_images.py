"""CLI command for selecting satellite imagery from STAC catalogs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import click
import pystac
from tqdm import tqdm

from ftw_dataset_tools.api.config import default_selection_workers
from ftw_dataset_tools.api.imagery import (
    clear_chip_selections,
    find_chip_items,
    find_collection_dir,
    get_imagery_stats,
    has_existing_scenes,
    select_imagery_for_catalog,
)
from ftw_dataset_tools.api.imagery.parallel import MAX_WORKERS
from ftw_dataset_tools.api.imagery.selection_workflow import (
    NoCloudFreeScenesError,
    SelectionWorkflowResult,
)
from ftw_dataset_tools.api.stac_items import copy_catalog

ChipEntries = list[tuple[pystac.Item, Path]]


def _load_single_chip(chip_path: Path, output_dir: Path | None) -> tuple[ChipEntries, Path]:
    if not chip_path.exists():
        raise click.ClickException(f"Chip file not found: {chip_path}")

    if output_dir is not None:
        raise click.ClickException("--output-dir is not supported in single chip mode")

    item = pystac.Item.from_file(str(chip_path))
    # The item lives at <collection_dir>/chips/<mgrs_square>/<item_id>/<item_id>.json
    try:
        catalog_dir = find_collection_dir(chip_path.resolve().parents[3])
    except (FileNotFoundError, IndexError) as e:
        raise click.ClickException(
            f"Could not locate collection.json above chip file: {chip_path}"
        ) from e

    click.echo(f"Single chip: {item.id}")
    return [(item, chip_path)], catalog_dir


def _copy_catalog_to(catalog_dir: Path, output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise click.ClickException(f"Output directory already exists: {output_dir}")
    click.echo(f"Copying catalog to: {output_dir}")
    try:
        copy_catalog(catalog_dir, output_dir)
    except Exception as e:
        raise click.ClickException(f"Failed to copy catalog: {e}") from e
    return output_dir


def _load_catalog_chips(dataset_dir: Path, output_dir: Path | None) -> tuple[ChipEntries, Path]:
    # Dataset directory - collection.json lives directly in it
    try:
        catalog_dir = find_collection_dir(dataset_dir)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e

    click.echo(f"Catalog: {catalog_dir}")

    # If output_dir specified, copy catalog before processing
    if output_dir is not None:
        catalog_dir = _copy_catalog_to(catalog_dir, output_dir)

    # Find all chip items (parent items, not child S2 items). Chips whose JSON
    # cannot be read are reported rather than silently dropped from the run.
    unreadable: list[dict] = []
    chip_entries = find_chip_items(catalog_dir, unreadable)

    for detail in unreadable:
        click.echo(
            click.style(f"  {detail['chip']}: {detail['error']}", fg="red"),
            err=True,
        )

    if not chip_entries:
        raise click.ClickException("No chip items found in catalog")
    return chip_entries, catalog_dir


def _load_chip_entries(input_path: Path, output_dir: Path | None) -> tuple[ChipEntries, Path]:
    """Chip items (with their paths) to work on and the catalog directory they belong to."""
    if input_path.suffix == ".json":
        return _load_single_chip(input_path, output_dir)
    return _load_catalog_chips(input_path, output_dir)


def _echo_stats(chip_items: list[pystac.Item], catalog_dir: Path) -> None:
    stats = get_imagery_stats(chip_items)
    click.echo(f"\nImagery Selection Statistics for {catalog_dir}")
    click.echo("=" * 50)
    click.echo(f"Total chips: {stats.total}")
    click.echo(f"With imagery: {stats.with_imagery}")
    click.echo(f"Without imagery: {stats.without_imagery}")

    if stats.planting_cloud_cover_max is not None:
        click.echo(
            f"\nPlanting cloud cover: max {stats.planting_cloud_cover_max:.1f}%, "
            f"avg {stats.planting_cloud_cover_avg:.1f}%"
        )

    if stats.harvest_cloud_cover_max is not None:
        click.echo(
            f"Harvest cloud cover: max {stats.harvest_cloud_cover_max:.1f}%, "
            f"avg {stats.harvest_cloud_cover_avg:.1f}%"
        )


def _clear_all_selections(chip_items: list[pystac.Item]) -> None:
    stats = get_imagery_stats(chip_items)

    if stats.with_imagery == 0:
        click.echo("No chips have imagery selections to clear.")
        return

    click.echo(click.style("\nWARNING: This will permanently delete:", fg="red", bold=True))
    click.echo(f"  - {stats.with_imagery} planting scene STAC items")
    click.echo(f"  - {stats.with_imagery} harvest scene STAC items")
    click.echo("  - Any downloaded GeoTIFF imagery files")
    click.echo("  - Imagery links from parent chip items")
    click.echo("")

    if not click.confirm("Are you sure you want to proceed?"):
        click.echo("Aborted.")
        return

    total_stac = 0
    total_tifs = 0
    chips_cleared = 0

    with tqdm(total=len(chip_items), desc="Clearing selections", unit="chip") as pbar:
        for item in chip_items:
            if has_existing_scenes(item):
                result = clear_chip_selections(item)
                total_stac += result.stac_items_deleted
                total_tifs += result.geotiffs_deleted
                chips_cleared += 1
            pbar.update(1)

    click.echo("")
    click.echo(click.style("Cleared imagery selections:", fg="green"))
    click.echo(f"  Chips processed: {chips_cleared}")
    click.echo(f"  STAC items deleted: {total_stac}")
    click.echo(f"  GeoTIFF files deleted: {total_tifs}")


def _echo_settings(
    year: int | None,
    cloud_cover_chip: float,
    buffer_days: int,
    buffer_expansion_size: int,
    num_buffer_expansions: int,
    verbose: bool,
) -> None:
    if year:
        click.echo(f"Year: {year} (from --year option)")
    else:
        click.echo("Year: (extracted from chip IDs)")
    click.echo(f"Cloud cover chip threshold: {cloud_cover_chip}%")
    click.echo(
        f"Buffer: {buffer_days} days (expand by {buffer_expansion_size}d x{num_buffer_expansions})"
    )
    if verbose:
        click.echo("Verbose mode: ON")


def _echo_first_five(heading: str, entries: list[dict]) -> None:
    if not entries:
        return
    click.echo(click.style(f"\n{len(entries)} {heading}:", fg="yellow"))
    for entry in entries[:5]:
        click.echo(f"  - {entry['chip']}: {entry['reason']}")
    if len(entries) > 5:
        click.echo(f"  ... and {len(entries) - 5} more")


def _echo_summary(chip_items: list[pystac.Item], result: SelectionWorkflowResult) -> None:
    skipped = result.skipped_details
    already_has = [s for s in skipped if s["reason"] == "Already has imagery selections"]
    no_scenes = [s for s in skipped if "No cloud-free" in s.get("reason", "")]
    other_skipped = [s for s in skipped if s not in already_has and s not in no_scenes]

    final_stats = get_imagery_stats(chip_items)

    click.echo("\n" + "=" * 50)
    click.echo("Summary:")
    click.echo(f"  Newly selected: {result.successful}")
    click.echo(f"  Already had imagery: {len(already_has)}")
    click.echo(f"  No cloud-free scenes: {len(no_scenes)}")
    if other_skipped:
        click.echo(f"  Other skipped: {len(other_skipped)}")
    click.echo(f"  Failed: {len(result.failed_details)}")
    click.echo("")
    click.echo(f"  Total with imagery: {final_stats.with_imagery}/{final_stats.total}")
    click.echo(f"  Still without imagery: {final_stats.without_imagery}")

    _echo_first_five("chips without cloud-free scenes", no_scenes)
    _echo_first_five("chips skipped (other)", other_skipped)


def _write_report(
    report_path: Path,
    chip_count: int,
    result: SelectionWorkflowResult,
    parameters: dict,
) -> None:
    report = {
        "total_processed": chip_count,
        "successful": result.successful,
        "skipped": result.skipped_details,
        "failed": result.failed_details,
        "parameters": parameters,
    }
    report_path.write_text(json.dumps(report, indent=2))
    click.echo(f"\nReport written to: {report_path}")


@click.command("select-images")
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--year",
    type=int,
    default=None,
    help="Calendar year for the crop cycle. If not provided, extracted from chip IDs.",
)
@click.option(
    "--cloud-cover-chip",
    type=click.FloatRange(0.0, 100.0),
    default=2.0,
    show_default=True,
    help="Maximum chip-level cloud cover percentage (0-100).",
)
@click.option(
    "--nodata-max",
    type=click.FloatRange(0.0, 100.0),
    default=0.0,
    show_default=True,
    help="Maximum nodata percentage (0-100). Default 0 rejects any nodata.",
)
@click.option(
    "--buffer-days",
    type=int,
    default=14,
    show_default=True,
    help="Days to search around crop calendar dates.",
)
@click.option(
    "--on-missing",
    type=click.Choice(["skip", "fail"]),
    default="skip",
    show_default=True,
    help="How to handle chips with no cloud-free scenes.",
)
@click.option(
    "--num-buffer-expansions",
    type=int,
    default=3,
    show_default=True,
    help="Number of times to expand search window for chips without cloud-free scenes.",
)
@click.option(
    "--buffer-expansion-size",
    type=int,
    default=14,
    show_default=True,
    help="Days to add to search window on each expansion.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing imagery selections (by default, chips with scenes are skipped).",
)
@click.option(
    "--search-backend",
    type=click.Choice(["parquet", "earth-search"]),
    default="parquet",
    show_default=True,
    help="Scene search backend: the Sentinel-2 STAC-GeoParquet mirror (no API, "
    "no rate limit) or the Earth Search STAC API.",
)
@click.option(
    "--workers",
    type=click.IntRange(1, MAX_WORKERS),
    default=None,
    help="Chips to select for concurrently. Defaults to 16 for the parquet "
    "backend and 4 for earth-search (which rate-bans aggressive clients).",
)
@click.option(
    "--output-report",
    type=click.Path(),
    default=None,
    help="Path for JSON report of skipped/failed chips.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Show detailed STAC query information and results.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for complete STAC catalog copy. If not specified, modifies catalog in place.",
)
@click.option(
    "--show-stats",
    is_flag=True,
    default=False,
    help="Show imagery selection statistics without processing.",
)
@click.option(
    "--clear-selections",
    is_flag=True,
    default=False,
    help="Remove all imagery selections (STAC items, GeoTIFFs, and links).",
)
def select_images_cmd(
    input_path: str,
    year: int | None,
    cloud_cover_chip: float,
    nodata_max: float,
    buffer_days: int,
    on_missing: Literal["skip", "fail"],
    num_buffer_expansions: int,
    buffer_expansion_size: int,
    force: bool,
    search_backend: Literal["parquet", "earth-search"],
    workers: int | None,
    output_report: str | None,
    verbose: bool,
    output_dir: Path | None,
    show_stats: bool,
    clear_selections: bool,
) -> None:
    """Select optimal Sentinel-2 imagery for chips.

    Finds cloud-free Sentinel-2 scenes for each chip based on crop calendar
    dates (planting and harvest), searching the Sentinel-2 STAC-GeoParquet
    mirror by default (--search-backend earth-search queries the Earth Search
    API instead). Creates child STAC items with remote asset links.

    By default, chips that already have imagery selections are skipped.
    Use --force to overwrite existing selections.

    \b
    INPUT_PATH: One of:
                - Dataset directory (containing collection.json directly)
                - Single chip JSON file for testing

    \b
    Examples:
        ftwd select-images ./my-dataset
        ftwd select-images ./my-dataset/chips/33UXP/ftw-34UFF1628_2024/ftw-34UFF1628_2024.json -v
        ftwd select-images ./my-dataset --year 2023 --cloud-cover-chip 5
        ftwd select-images ./my-dataset --force  # Overwrite existing selections
    """
    if workers is None:
        workers = default_selection_workers(search_backend)

    chip_entries, catalog_dir = _load_chip_entries(Path(input_path), output_dir)
    chip_items = [item for item, _item_path in chip_entries]

    if show_stats:
        _echo_stats(chip_items, catalog_dir)
        return

    if clear_selections:
        _clear_all_selections(chip_items)
        return

    _echo_settings(
        year, cloud_cover_chip, buffer_days, buffer_expansion_size, num_buffer_expansions, verbose
    )
    click.echo(f"\nFound {len(chip_items)} total chips")

    try:
        result = select_imagery_for_catalog(
            catalog_dir,
            year=year,
            cloud_cover_chip=cloud_cover_chip,
            nodata_max=nodata_max,
            buffer_days=buffer_days,
            num_buffer_expansions=num_buffer_expansions,
            buffer_expansion_size=buffer_expansion_size,
            force=force,
            on_missing=on_missing,
            verbose=verbose,
            workers=workers,
            search_backend=search_backend,
            chip_items=chip_entries,
            on_progress=lambda msg: click.echo(f"  {msg}"),
        )
    except NoCloudFreeScenesError as e:
        raise click.ClickException(str(e)) from e

    _echo_summary(chip_items, result)

    if output_report:
        parameters = {
            "year": year if year else "extracted_from_chip_ids",
            "cloud_cover_chip": cloud_cover_chip,
            "buffer_days": buffer_days,
            "num_buffer_expansions": num_buffer_expansions,
            "buffer_expansion_size": buffer_expansion_size,
        }
        _write_report(Path(output_report), len(chip_items), result, parameters)

    if result.successful:
        click.echo(click.style("\nDone!", fg="green"))
    else:
        click.echo(click.style("\nNo chips successfully processed.", fg="yellow"))


# Alias for registration
select_images = select_images_cmd
