"""CLI command for selecting satellite imagery from STAC catalogs."""

from __future__ import annotations

import json
import re
from datetime import UTC
from pathlib import Path
from typing import Literal

import click
import pystac
from tqdm import tqdm

from ftw_dataset_tools.api.imagery import (
    ImageryProgressBar,
    SceneSelectionResult,
    select_scenes_for_chip,
)


def _extract_year_from_chip_id(chip_id: str) -> int | None:
    """Extract year from chip ID (e.g., 'ftw-34UFF1628_2024' -> 2024)."""
    match = re.search(r"_(\d{4})$", chip_id)
    if match:
        return int(match.group(1))
    return None


def _has_existing_scenes(item: pystac.Item) -> bool:
    """Check if item already has planting and harvest scene links."""
    has_planting = any(link.rel == "ftw:planting" for link in item.links)
    has_harvest = any(link.rel == "ftw:harvest" for link in item.links)
    return has_planting and has_harvest


def _get_imagery_stats(chip_items: list[pystac.Item]) -> dict:
    """Gather imagery selection statistics from chip items."""
    stats = {
        "total": len(chip_items),
        "with_imagery": 0,
        "without_imagery": 0,
        "planting_cloud_covers": [],
        "harvest_cloud_covers": [],
    }

    for item in chip_items:
        if _has_existing_scenes(item):
            stats["with_imagery"] += 1
            # Get cloud cover values
            planting_cc = item.properties.get("ftw:planting_cloud_cover")
            harvest_cc = item.properties.get("ftw:harvest_cloud_cover")
            if planting_cc is not None:
                stats["planting_cloud_covers"].append(planting_cc)
            if harvest_cc is not None:
                stats["harvest_cloud_covers"].append(harvest_cc)
        else:
            stats["without_imagery"] += 1

    return stats


def _clear_chip_selections(catalog_dir: Path, item: pystac.Item) -> dict:
    """Clear imagery selections for a single chip.

    Returns dict with counts of deleted items.
    """
    from datetime import datetime

    chip_dir = catalog_dir / item.id
    deleted = {"stac_items": 0, "geotiffs": 0}

    # Delete planting and harvest child STAC items and their GeoTIFFs
    for season in ["planting", "harvest"]:
        child_json = chip_dir / f"{item.id}_{season}_s2.json"
        if child_json.exists():
            child_json.unlink()
            deleted["stac_items"] += 1

        # Delete associated GeoTIFFs (e.g., ftw-xxx_planting_image_s2.tif)
        for tif in chip_dir.glob(f"{item.id}_{season}_*.tif"):
            tif.unlink()
            deleted["geotiffs"] += 1

    # Remove ftw:planting and ftw:harvest links from parent item
    item.links = [link for link in item.links if link.rel not in ("ftw:planting", "ftw:harvest")]

    # Extract calendar year before removing properties (needed to restore datetime)
    calendar_year = item.properties.get("ftw:calendar_year")

    # Remove ftw: properties related to imagery selection
    props_to_remove = [
        "ftw:calendar_year",
        "ftw:planting_day",
        "ftw:harvest_day",
        "ftw:stac_host",
        "ftw:cloud_cover_scene_threshold",
        "ftw:buffer_days",
        "ftw:pixel_check",
        "ftw:num_buffer_expansions",
        "ftw:buffer_expansion_size",
        "ftw:planting_buffer_used",
        "ftw:harvest_buffer_used",
        "ftw:expansions_performed",
        "ftw:planting_cloud_cover",
        "ftw:harvest_cloud_cover",
    ]
    for prop in props_to_remove:
        item.properties.pop(prop, None)

    # Remove planting_image and harvest_image assets if they exist
    item.assets.pop("planting_image", None)
    item.assets.pop("harvest_image", None)

    # Restore datetime - STAC requires either datetime or both start/end_datetime
    # Remove the selection-set temporal range and restore a single datetime
    item.properties.pop("start_datetime", None)
    item.properties.pop("end_datetime", None)

    # Set datetime to Jan 1 of the calendar year if known, otherwise current date
    if calendar_year:
        restored_dt = datetime(calendar_year, 1, 1, 0, 0, 0, tzinfo=UTC)
    else:
        restored_dt = datetime.now(UTC)
    # Set both the Item attribute and the properties dict
    item.datetime = restored_dt
    item.properties["datetime"] = restored_dt.isoformat()

    # Always save since we've modified the item (removed properties, restored datetime)
    parent_path = chip_dir / f"{item.id}.json"
    if item.get_self_href() is None:
        item.set_self_href(str(parent_path))
    item.save_object(dest_href=str(parent_path))

    return deleted


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
    type=click.Choice(["skip", "fail", "best-available"]),
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
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from previous run using progress file.",
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
    on_missing: Literal["skip", "fail", "best-available"],
    num_buffer_expansions: int,
    buffer_expansion_size: int,
    resume: bool,  # noqa: ARG001 - planned feature
    force: bool,
    output_report: str | None,
    verbose: bool,
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
        ftwd select-images ./chips --year 2023 --cloud-cover-scene 5
        ftwd select-images ./chips --force  # Overwrite existing selections
    """
    input_path_obj = Path(input_path)

    # Determine if input is a single chip JSON or a catalog directory
    single_chip_mode = input_path_obj.suffix == ".json"

    if single_chip_mode:
        # Single chip JSON file
        if not input_path_obj.exists():
            raise click.ClickException(f"Chip file not found: {input_path}")

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
        stats = _get_imagery_stats(chip_items)
        click.echo(f"\nImagery Selection Statistics for {catalog_dir}")
        click.echo("=" * 50)
        click.echo(f"Total chips: {stats['total']}")
        click.echo(f"With imagery: {stats['with_imagery']}")
        click.echo(f"Without imagery: {stats['without_imagery']}")

        if stats["planting_cloud_covers"]:
            p_max = max(stats["planting_cloud_covers"])
            p_avg = sum(stats["planting_cloud_covers"]) / len(stats["planting_cloud_covers"])
            click.echo(f"\nPlanting cloud cover: max {p_max:.1f}%, avg {p_avg:.1f}%")

        if stats["harvest_cloud_covers"]:
            h_max = max(stats["harvest_cloud_covers"])
            h_avg = sum(stats["harvest_cloud_covers"]) / len(stats["harvest_cloud_covers"])
            click.echo(f"Harvest cloud cover: max {h_max:.1f}%, avg {h_avg:.1f}%")

        return

    # Handle --clear-selections mode
    if clear_selections:
        stats = _get_imagery_stats(chip_items)

        if stats["with_imagery"] == 0:
            click.echo("No chips have imagery selections to clear.")
            return

        click.echo(click.style("\nWARNING: This will permanently delete:", fg="red", bold=True))
        click.echo(f"  - {stats['with_imagery']} planting scene STAC items")
        click.echo(f"  - {stats['with_imagery']} harvest scene STAC items")
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
                if _has_existing_scenes(item):
                    deleted = _clear_chip_selections(catalog_dir, item)
                    total_stac += deleted["stac_items"]
                    total_tifs += deleted["geotiffs"]
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
        if not force and _has_existing_scenes(item):
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
                    _create_child_items(
                        catalog_dir=catalog_dir,
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
    final_stats = _get_imagery_stats(chip_items)

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
    click.echo(f"  Total with imagery: {final_stats['with_imagery']}/{final_stats['total']}")
    click.echo(f"  Still without imagery: {final_stats['without_imagery']}")

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


def _create_child_items(
    catalog_dir: Path,
    parent_item: pystac.Item,
    result: SceneSelectionResult,
    year: int,
    cloud_cover_chip: float,
    buffer_days: int,
    num_buffer_expansions: int,
    buffer_expansion_size: int,
) -> None:
    """Create child STAC items for planting and harvest scenes."""
    chip_dir = catalog_dir / parent_item.id

    # Ensure parent item has self_href set (required for saving with relative links)
    parent_path = chip_dir / f"{parent_item.id}.json"
    if parent_item.get_self_href() is None:
        parent_item.set_self_href(str(parent_path))

    # Update parent item with FTW properties
    parent_item.properties["ftw:calendar_year"] = year
    parent_item.properties["ftw:planting_day"] = result.crop_calendar.planting_day
    parent_item.properties["ftw:harvest_day"] = result.crop_calendar.harvest_day
    parent_item.properties["ftw:stac_host"] = "earthsearch"  # Always earthsearch
    parent_item.properties["ftw:cloud_cover_chip_threshold"] = cloud_cover_chip
    parent_item.properties["ftw:buffer_days"] = buffer_days
    parent_item.properties["ftw:num_buffer_expansions"] = num_buffer_expansions
    parent_item.properties["ftw:buffer_expansion_size"] = buffer_expansion_size
    # Track actual buffer used for each season
    parent_item.properties["ftw:planting_buffer_used"] = result.planting_buffer_used
    parent_item.properties["ftw:harvest_buffer_used"] = result.harvest_buffer_used
    parent_item.properties["ftw:expansions_performed"] = result.expansions_performed

    # Set temporal extent to the full calendar year
    # This represents the crop cycle year, not just the scene acquisition dates
    from datetime import datetime

    parent_item.properties["start_datetime"] = datetime(year, 1, 1, 0, 0, 0, tzinfo=UTC).isoformat()
    parent_item.properties["end_datetime"] = datetime(
        year, 12, 31, 23, 59, 59, tzinfo=UTC
    ).isoformat()

    # Add cloud cover from child scenes to parent
    if result.planting_scene:
        parent_item.properties["ftw:planting_cloud_cover"] = round(
            result.planting_scene.cloud_cover, 2
        )
    if result.harvest_scene:
        parent_item.properties["ftw:harvest_cloud_cover"] = round(
            result.harvest_scene.cloud_cover, 2
        )

    # Remove any existing planting/harvest links before adding new ones
    parent_item.links = [
        link
        for link in parent_item.links
        if link.rel not in ("ftw:planting", "ftw:harvest", "derived")
    ]

    # Add links from parent to child items
    if result.planting_scene:
        planting_child_id = f"{parent_item.id}_planting_s2"
        parent_item.add_link(
            pystac.Link(
                rel="ftw:planting",
                target=f"./{planting_child_id}.json",
                media_type="application/json",
                title="Planting season Sentinel-2 imagery",
            )
        )

    if result.harvest_scene:
        harvest_child_id = f"{parent_item.id}_harvest_s2"
        parent_item.add_link(
            pystac.Link(
                rel="ftw:harvest",
                target=f"./{harvest_child_id}.json",
                media_type="application/json",
                title="Harvest season Sentinel-2 imagery",
            )
        )

    # Save updated parent (parent_path already set at function start)
    parent_item.save_object(dest_href=str(parent_path))

    # Create planting child item
    if result.planting_scene:
        _create_season_child_item(
            chip_dir=chip_dir,
            parent_item=parent_item,
            scene=result.planting_scene,
            season="planting",
            year=year,
        )

    # Create harvest child item
    if result.harvest_scene:
        _create_season_child_item(
            chip_dir=chip_dir,
            parent_item=parent_item,
            scene=result.harvest_scene,
            season="harvest",
            year=year,
        )


def _create_season_child_item(
    chip_dir: Path,
    parent_item: pystac.Item,
    scene,
    season: Literal["planting", "harvest"],
    year: int,
) -> None:
    """Create a child STAC item for a season (planting or harvest)."""
    child_id = f"{parent_item.id}_{season}_s2"

    # Create child item
    child_path = chip_dir / f"{child_id}.json"
    child_item = pystac.Item(
        id=child_id,
        geometry=parent_item.geometry,
        bbox=parent_item.bbox,
        datetime=scene.datetime,
        properties={
            "ftw:season": season,
            "ftw:source": "sentinel-2",
            "ftw:calendar_year": year,
        },
    )

    # Set self_href before adding links (required for relative link resolution)
    child_item.set_self_href(str(child_path))

    # Copy relevant band assets from source scene
    bands_to_copy = ["red", "green", "blue", "nir", "scl", "visual"]
    for band in bands_to_copy:
        if band in scene.item.assets:
            child_item.assets[band] = scene.item.assets[band].clone()

    # Add cloud probability asset if available
    if "cloud" in scene.item.assets:
        child_item.assets["cloud_probability"] = scene.item.assets["cloud"].clone()

    # Add links
    child_item.add_link(
        pystac.Link(
            rel="ftw:parent_chip",
            target=f"./{parent_item.id}.json",
            media_type="application/json",
        )
    )

    if scene.stac_url:
        child_item.add_link(
            pystac.Link(
                rel="via",
                target=scene.stac_url,
                media_type="application/json",
            )
        )

    # Always include eo:cloud_cover, rounded to 2 decimal places
    child_item.properties["eo:cloud_cover"] = round(scene.cloud_cover, 2)

    # Save child item (child_path already set at function start)
    child_item.save_object(dest_href=str(child_path))


# Alias for registration
select_images = select_images_cmd
