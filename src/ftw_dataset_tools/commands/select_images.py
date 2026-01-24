"""CLI command for selecting satellite imagery from STAC catalogs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import click
import pystac
from tqdm import tqdm

from ftw_dataset_tools.api.imagery import (
    ImageryProgressBar,
    clear_chip_selections,
    create_child_items_from_selection,
    get_imagery_stats,
    has_existing_scenes,
    select_scenes_for_chip,
)
from ftw_dataset_tools.api.stac_items import copy_catalog


def _extract_year_from_chip_id(chip_id: str) -> int | None:
    """Extract year from chip ID (e.g., 'ftw-34UFF1628_2024' -> 2024)."""
    match = re.search(r"_(\d{4})$", chip_id)
    if match:
        return int(match.group(1))
    return None


def _extract_year_from_item(item: pystac.Item) -> int | None:
    """Extract year from item's temporal properties."""
    # Try start_datetime first
    start_dt = item.properties.get("start_datetime")
    if start_dt:
        try:
            from datetime import datetime

            if isinstance(start_dt, str):
                # Parse ISO format datetime
                dt = datetime.fromisoformat(start_dt.replace("Z", "+00:00"))
                return dt.year
        except (ValueError, TypeError):
            pass

    # Try datetime property
    if item.datetime:
        return item.datetime.year

    return None


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
    output_report: str | None,
    verbose: bool,
    output_dir: Path | None,
    show_stats: bool,
    clear_selections: bool,
) -> None:
    """Select optimal Sentinel-2 imagery for chips.

    Queries EarthSearch STAC catalog to find cloud-free Sentinel-2 scenes for each
    chip based on crop calendar dates (planting and harvest). Creates child STAC
    items with remote asset links.

    By default, chips that already have imagery selections are skipped.
    Use --force to overwrite existing selections.

    \b
    INPUT_PATH: One of:
                - Dataset directory (containing *-chips/ subdirectory with collection.json)
                - Chips collection directory (containing collection.json directly)
                - Single chip JSON file for testing

    \b
    Examples:
        ftwd select-images ./my-dataset              # Dataset directory
        ftwd select-images ./my-dataset-chips        # Chips collection directory
        ftwd select-images ./chips/ftw-34UFF1628_2024/ftw-34UFF1628_2024.json -v
        ftwd select-images ./chips --year 2023 --cloud-cover-chip 5
        ftwd select-images ./chips --force  # Overwrite existing selections
    """
    input_path_obj = Path(input_path)

    # Determine if input is a single chip JSON or a catalog directory
    single_chip_mode = input_path_obj.suffix == ".json"

    if single_chip_mode:
        # Single chip JSON file
        if not input_path_obj.exists():
            raise click.ClickException(f"Chip file not found: {input_path}")

        if output_dir is not None:
            raise click.ClickException("--output-dir is not supported in single chip mode")

        item = pystac.Item.from_file(str(input_path_obj))
        chip_items = [item]
        # Catalog dir is parent of the chip directory (grandparent of the JSON file)
        catalog_dir = input_path_obj.parent.parent

        click.echo(f"Single chip: {item.id}")
    else:
        # Catalog directory - check for collection.json here or in a chips subdirectory
        collection_file = input_path_obj / "collection.json"

        if collection_file.exists():
            # collection.json directly in input path (chips directory)
            catalog_dir = input_path_obj
        else:
            # Look for *-chips subdirectory with collection.json (dataset directory)
            chips_dirs = list(input_path_obj.glob("*-chips"))
            chips_dir_with_collection = None
            for chips_dir in chips_dirs:
                if (chips_dir / "collection.json").exists():
                    chips_dir_with_collection = chips_dir
                    break

            if chips_dir_with_collection:
                catalog_dir = chips_dir_with_collection
                collection_file = catalog_dir / "collection.json"
            else:
                raise click.ClickException(
                    f"No collection.json found in {input_path} or in any *-chips subdirectory"
                )

        click.echo(f"Catalog: {catalog_dir}")

        # If output_dir specified, copy catalog before processing
        if output_dir is not None:
            output_dir = output_dir.resolve()
            if output_dir.exists():
                raise click.ClickException(f"Output directory already exists: {output_dir}")
            click.echo(f"Copying catalog to: {output_dir}")
            try:
                copy_catalog(catalog_dir, output_dir)
            except Exception as e:
                raise click.ClickException(f"Failed to copy catalog: {e}") from e
            catalog_dir = output_dir
            collection_file = catalog_dir / "collection.json"

        # Load collection to find items
        collection = pystac.Collection.from_file(str(collection_file))

        # Find all chip items (parent items, not child S2 items)
        chip_items = []
        for item_link in collection.get_item_links():
            item_path = catalog_dir / item_link.href
            if item_path.exists():
                item = pystac.Item.from_file(str(item_path))
                # Skip child items (they have _planting_s2 or _harvest_s2 suffix)
                if not item.id.endswith("_planting_s2") and not item.id.endswith("_harvest_s2"):
                    chip_items.append(item)

        if not chip_items:
            raise click.ClickException("No chip items found in catalog")

    # Handle --show-stats mode
    if show_stats:
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

        return

    # Handle --clear-selections mode
    if clear_selections:
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

        # Clear selections
        total_stac = 0
        total_tifs = 0
        chips_cleared = 0

        with tqdm(total=len(chip_items), desc="Clearing selections", unit="chip") as pbar:
            for item in chip_items:
                if has_existing_scenes(item):
                    result = clear_chip_selections(catalog_dir, item)
                    total_stac += result.stac_items_deleted
                    total_tifs += result.geotiffs_deleted
                    chips_cleared += 1
                pbar.update(1)

        click.echo("")
        click.echo(click.style("Cleared imagery selections:", fg="green"))
        click.echo(f"  Chips processed: {chips_cleared}")
        click.echo(f"  STAC items deleted: {total_stac}")
        click.echo(f"  GeoTIFF files deleted: {total_tifs}")
        return

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

    click.echo(f"\nFound {len(chip_items)} total chips")

    # Pre-scan to categorize chips and filter to only those needing processing
    chips_to_process: list[tuple[pystac.Item, int]] = []  # (item, year)
    skipped: list[dict] = []
    already_has_count = 0

    for item in chip_items:
        chip_id = item.id
        bbox = tuple(item.bbox) if item.bbox else None

        if bbox is None:
            skipped.append({"chip": chip_id, "reason": "No bbox in item"})
            continue

        # Skip chips that already have scene selections (unless --force)
        if not force and has_existing_scenes(item):
            skipped.append({"chip": chip_id, "reason": "Already has imagery selections"})
            already_has_count += 1
            continue

        # Determine the year for this chip
        chip_year = year
        if chip_year is None:
            chip_year = _extract_year_from_chip_id(chip_id)
        if chip_year is None:
            chip_year = _extract_year_from_item(item)
        if chip_year is None:
            reason = "No year provided and could not extract from chip ID or item properties"
            skipped.append({"chip": chip_id, "reason": reason})
            continue

        chips_to_process.append((item, chip_year))

    # Report pre-scan results
    if already_has_count > 0:
        click.echo(f"  Already have imagery: {already_has_count}")
    pre_skipped = len(skipped) - already_has_count
    if pre_skipped > 0:
        click.echo(f"  Skipped (no bbox/year): {pre_skipped}")
    click.echo(f"  To process: {len(chips_to_process)}")

    if not chips_to_process:
        click.echo("\nNo chips need processing.")
        return

    # Track results for detailed reporting
    successful: list[str] = []
    failed: list[dict] = []

    # Process only the chips that need work
    with ImageryProgressBar(total=len(chips_to_process), leave=True, verbose=verbose) as progress:
        for item, chip_year in chips_to_process:
            chip_id = item.id
            progress.start_chip(chip_id)
            bbox = tuple(item.bbox)  # Already validated in pre-scan

            try:
                result = select_scenes_for_chip(
                    chip_id=chip_id,
                    bbox=bbox,
                    year=chip_year,
                    cloud_cover_chip=cloud_cover_chip,
                    nodata_max=nodata_max,
                    buffer_days=buffer_days,
                    num_buffer_expansions=num_buffer_expansions,
                    buffer_expansion_size=buffer_expansion_size,
                    on_progress=progress.on_progress,
                )

                if result.success:
                    # Create child STAC items for planting and harvest
                    create_child_items_from_selection(
                        chip_dir=catalog_dir / item.id,
                        parent_item=item,
                        result=result,
                        year=chip_year,
                        cloud_cover_chip=cloud_cover_chip,
                        buffer_days=buffer_days,
                        num_buffer_expansions=num_buffer_expansions,
                        buffer_expansion_size=buffer_expansion_size,
                    )
                    successful.append(chip_id)
                    progress.mark_success(result)
                else:
                    if on_missing == "fail":
                        raise click.ClickException(
                            f"No cloud-free scenes for {chip_id}: {result.skipped_reason}"
                        )
                    skipped.append(
                        {
                            "chip": chip_id,
                            "reason": result.skipped_reason,
                            "candidates_checked": result.candidates_checked,
                        }
                    )
                    progress.mark_skipped(result.skipped_reason or "Unknown reason")

            except click.ClickException:
                raise
            except Exception as e:
                if on_missing == "fail":
                    raise
                failed.append({"chip": chip_id, "error": str(e)})
                progress.mark_failed(str(e))

    # Categorize skipped items
    already_has = [s for s in skipped if s["reason"] == "Already has imagery selections"]
    no_scenes = [s for s in skipped if "No cloud-free" in s.get("reason", "")]
    other_skipped = [s for s in skipped if s not in already_has and s not in no_scenes]

    # Get final stats
    final_stats = get_imagery_stats(chip_items)

    # Print summary
    click.echo("\n" + "=" * 50)
    click.echo("Summary:")
    click.echo(f"  Newly selected: {len(successful)}")
    click.echo(f"  Already had imagery: {len(already_has)}")
    click.echo(f"  No cloud-free scenes: {len(no_scenes)}")
    if other_skipped:
        click.echo(f"  Other skipped: {len(other_skipped)}")
    click.echo(f"  Failed: {len(failed)}")
    click.echo("")
    click.echo(f"  Total with imagery: {final_stats.with_imagery}/{final_stats.total}")
    click.echo(f"  Still without imagery: {final_stats.without_imagery}")

    if no_scenes:
        click.echo(click.style(f"\n{len(no_scenes)} chips without cloud-free scenes:", fg="yellow"))
        for s in no_scenes[:5]:
            click.echo(f"  - {s['chip']}: {s['reason']}")
        if len(no_scenes) > 5:
            click.echo(f"  ... and {len(no_scenes) - 5} more")

    if other_skipped:
        click.echo(click.style(f"\n{len(other_skipped)} chips skipped (other):", fg="yellow"))
        for s in other_skipped[:5]:
            click.echo(f"  - {s['chip']}: {s['reason']}")
        if len(other_skipped) > 5:
            click.echo(f"  ... and {len(other_skipped) - 5} more")

    if failed:
        click.echo(click.style(f"\n{len(failed)} chips failed:", fg="red"))
        for f in failed[:5]:
            click.echo(f"  - {f['chip']}: {f['error']}")
        if len(failed) > 5:
            click.echo(f"  ... and {len(failed) - 5} more")

    # Write report if requested
    if output_report:
        report = {
            "total_processed": len(chip_items),
            "successful": len(successful),
            "skipped": skipped,
            "failed": failed,
            "parameters": {
                "year": year if year else "extracted_from_chip_ids",
                "cloud_cover_chip": cloud_cover_chip,
                "buffer_days": buffer_days,
                "num_buffer_expansions": num_buffer_expansions,
                "buffer_expansion_size": buffer_expansion_size,
            },
        }
        report_path = Path(output_report)
        report_path.write_text(json.dumps(report, indent=2))
        click.echo(f"\nReport written to: {report_path}")

    if successful:
        click.echo(click.style("\nDone!", fg="green"))
    else:
        click.echo(click.style("\nNo chips successfully processed.", fg="yellow"))


# Alias for registration
select_images = select_images_cmd
