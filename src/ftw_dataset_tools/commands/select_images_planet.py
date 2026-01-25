"""CLI command for selecting Planet satellite imagery from STAC catalogs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import click
import pystac

from ftw_dataset_tools.api.imagery import (
    ImageryProgressBar,
)
from ftw_dataset_tools.api.imagery.planet_client import (
    DEFAULT_BUFFER_DAYS,
    DEFAULT_NUM_ITERATIONS,
    PlanetClient,
)
from ftw_dataset_tools.api.imagery.planet_selection import (
    DEFAULT_CLOUD_COVER_CHIP,
    PlanetSelectionResult,
    generate_thumbnail,
    select_planet_scenes_for_chip,
)

if TYPE_CHECKING:
    from ftw_dataset_tools.api.imagery.planet_selection import PlanetScene


def _extract_year_from_chip_id(chip_id: str) -> int | None:
    """Extract year from chip ID (e.g., 'ftw-34UFF1628_2024' -> 2024)."""
    match = re.search(r"_(\d{4})$", chip_id)
    if match:
        return int(match.group(1))
    return None


def _extract_year_from_item(item: pystac.Item) -> int | None:
    """Extract year from item's temporal properties."""
    start_dt = item.properties.get("start_datetime")
    if start_dt:
        try:
            from datetime import datetime

            if isinstance(start_dt, str):
                dt = datetime.fromisoformat(start_dt.replace("Z", "+00:00"))
                return dt.year
        except (ValueError, TypeError):
            pass

    if item.datetime:
        return item.datetime.year

    return None


def _has_existing_planet_scenes(item: pystac.Item) -> bool:
    """Check if item already has Planet imagery selections."""
    return any(link.rel in ("ftw:planting_planet", "ftw:harvest_planet") for link in item.links)


def _create_planet_child_items(
    client: PlanetClient,
    chip_dir: Path,
    parent_item: pystac.Item,
    result: PlanetSelectionResult,
    year: int,
    cloud_cover_chip: float,
    buffer_days: int,
    num_iterations: int,
) -> None:
    """Create child STAC items for Planet planting and harvest scenes.

    Updates the parent item with Planet properties and creates child items
    with thumbnails and proper links.

    Args:
        client: Authenticated Planet client
        chip_dir: Directory containing the chip STAC items
        parent_item: Parent chip STAC item to update
        result: Planet selection result containing planting and harvest scenes
        year: Calendar year for the crop cycle
        cloud_cover_chip: Cloud cover threshold used for selection
        buffer_days: Initial buffer days used for selection
        num_iterations: Number of iterations configured
    """
    # Ensure parent item has self_href set
    parent_path = chip_dir / f"{parent_item.id}.json"
    if parent_item.get_self_href() is None:
        parent_item.set_self_href(str(parent_path))

    # Update parent item with Planet-specific properties
    parent_item.properties["ftw:planet_calendar_year"] = year
    parent_item.properties["ftw:planet_cloud_cover_chip_threshold"] = cloud_cover_chip
    parent_item.properties["ftw:planet_buffer_days"] = buffer_days
    parent_item.properties["ftw:planet_num_iterations"] = num_iterations
    parent_item.properties["ftw:planet_iterations_used"] = result.iterations_used
    parent_item.properties["ftw:planet_planting_buffer_used"] = result.planting_buffer_used
    parent_item.properties["ftw:planet_harvest_buffer_used"] = result.harvest_buffer_used

    # Add cloud cover from child scenes to parent
    if result.planting_scene:
        parent_item.properties["ftw:planet_planting_cloud_cover"] = round(
            result.planting_scene.cloud_cover, 2
        )
    if result.harvest_scene:
        parent_item.properties["ftw:planet_harvest_cloud_cover"] = round(
            result.harvest_scene.cloud_cover, 2
        )

    # Remove any existing Planet planting/harvest links before adding new ones
    parent_item.links = [
        link
        for link in parent_item.links
        if link.rel not in ("ftw:planting_planet", "ftw:harvest_planet")
    ]

    # Add links from parent to child items
    if result.planting_scene:
        planting_child_id = f"{parent_item.id}_planting_planet"
        parent_item.add_link(
            pystac.Link(
                rel="ftw:planting_planet",
                target=f"./{planting_child_id}.json",
                media_type="application/json",
                title="Planting season Planet imagery",
            )
        )

    if result.harvest_scene:
        harvest_child_id = f"{parent_item.id}_harvest_planet"
        parent_item.add_link(
            pystac.Link(
                rel="ftw:harvest_planet",
                target=f"./{harvest_child_id}.json",
                media_type="application/json",
                title="Harvest season Planet imagery",
            )
        )

    # Save updated parent
    parent_item.save_object(dest_href=str(parent_path))

    # Create planting child item
    if result.planting_scene:
        _create_planet_season_child_item(
            client=client,
            chip_dir=chip_dir,
            parent_item=parent_item,
            scene=result.planting_scene,
            season="planting",
            year=year,
        )

    # Create harvest child item
    if result.harvest_scene:
        _create_planet_season_child_item(
            client=client,
            chip_dir=chip_dir,
            parent_item=parent_item,
            scene=result.harvest_scene,
            season="harvest",
            year=year,
        )


def _create_planet_season_child_item(
    client: PlanetClient,
    chip_dir: Path,
    parent_item: pystac.Item,
    scene: PlanetScene,
    season: Literal["planting", "harvest"],
    year: int,
) -> None:
    """Create a child STAC item for a Planet season (planting or harvest).

    Args:
        client: Authenticated Planet client
        chip_dir: Directory containing the chip STAC items
        parent_item: Parent chip STAC item
        scene: Selected Planet scene for this season
        season: Season identifier
        year: Calendar year for the crop cycle
    """
    child_id = f"{parent_item.id}_{season}_planet"
    child_path = chip_dir / f"{child_id}.json"

    # Generate thumbnail clipped to chip bounds
    thumb_path = chip_dir / f"{child_id}_thumb.png"
    chip_bbox = tuple(parent_item.bbox) if parent_item.bbox else None
    thumb_result = generate_thumbnail(client, scene.id, thumb_path, bbox=chip_bbox)

    # Create child item
    child_item = pystac.Item(
        id=child_id,
        geometry=parent_item.geometry,
        bbox=parent_item.bbox,
        datetime=scene.datetime,
        properties={
            "ftw:season": season,
            "ftw:source": "planet",
            "ftw:calendar_year": year,
            "ftw:scene_id": scene.id,
            "ftw:collection": "PSScene",
            "ftw:clear_coverage": round(scene.clear_coverage, 2),
            "ftw:cloud_cover": round(scene.cloud_cover, 2),
            "eo:cloud_cover": round(scene.cloud_cover, 2),
        },
    )

    # Set self_href before adding links
    child_item.set_self_href(str(child_path))

    # Add thumbnail asset if generated
    if thumb_result:
        child_item.assets["thumbnail"] = pystac.Asset(
            href=f"./{child_id}_thumb.png",
            media_type="image/png",
            roles=["thumbnail"],
        )

    # Add links
    child_item.add_link(
        pystac.Link(
            rel="ftw:parent_chip",
            target=f"./{parent_item.id}.json",
            media_type="application/json",
        )
    )

    # Add link to original Planet STAC item
    if scene.stac_url:
        child_item.add_link(
            pystac.Link(
                rel="derived_from",
                target=scene.stac_url,
                media_type="application/geo+json",
            )
        )
    else:
        # Construct URL from scene ID
        planet_stac_url = f"https://api.planet.com/x/data/collections/PSScene/items/{scene.id}"
        child_item.add_link(
            pystac.Link(
                rel="derived_from",
                target=planet_stac_url,
                media_type="application/geo+json",
            )
        )

    # Save child item
    child_item.save_object(dest_href=str(child_path))


@click.command("select-images-planet")
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--year",
    type=int,
    default=None,
    help="Calendar year for the crop cycle. If not provided, extracted from chip IDs.",
)
@click.option(
    "--buffer-days",
    type=int,
    default=DEFAULT_BUFFER_DAYS,
    show_default=True,
    help="Initial buffer days around crop calendar dates.",
)
@click.option(
    "--num-iterations",
    type=int,
    default=DEFAULT_NUM_ITERATIONS,
    show_default=True,
    help="Number of buffer expansion iterations.",
)
@click.option(
    "--cloud-cover-chip",
    type=click.FloatRange(0.0, 100.0),
    default=DEFAULT_CLOUD_COVER_CHIP,
    show_default=True,
    help="Maximum chip-level cloud cover percentage (0-100).",
)
@click.option(
    "--on-missing",
    type=click.Choice(["skip", "fail"]),
    default="skip",
    show_default=True,
    help="How to handle chips with no cloud-free scenes.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing Planet imagery selections.",
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
def select_images_planet_cmd(
    input_path: str,
    year: int | None,
    buffer_days: int,
    num_iterations: int,
    cloud_cover_chip: float,
    on_missing: Literal["skip", "fail"],
    force: bool,
    output_report: str | None,
    verbose: bool,
) -> None:
    """Select optimal Planet imagery for chips.

    Queries Planet STAC API to find cloud-free PlanetScope scenes for each
    chip based on crop calendar dates (planting and harvest). Creates child STAC
    items with thumbnails and metadata for the download command.

    Requires PL_API_KEY environment variable to be set.

    \b
    INPUT_PATH: One of:
                - Dataset directory (containing *-chips/ subdirectory with collection.json)
                - Chips collection directory (containing collection.json directly)
                - Single chip JSON file for testing

    \b
    Examples:
        ftwd select-images-planet ./my-dataset --year 2024
        ftwd select-images-planet ./my-dataset-chips --year 2024
        ftwd select-images-planet ./chips --year 2024 --buffer-days 21 --num-iterations 5
        ftwd select-images-planet ./chips --force  # Overwrite existing selections
    """
    input_path_obj = Path(input_path)

    # Validate Planet API key before processing
    try:
        client = PlanetClient()
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    try:
        client.validate_auth()
    except Exception as e:
        raise click.ClickException(f"Planet API authentication failed: {e}") from e

    click.echo("Planet API: authenticated")

    # Determine if input is a single chip JSON or a catalog directory
    single_chip_mode = input_path_obj.suffix == ".json"

    if single_chip_mode:
        if not input_path_obj.exists():
            raise click.ClickException(f"Chip file not found: {input_path}")

        item = pystac.Item.from_file(str(input_path_obj))
        chip_items = [item]
        catalog_dir = input_path_obj.parent.parent

        click.echo(f"Single chip: {item.id}")
    else:
        # Catalog directory
        collection_file = input_path_obj / "collection.json"

        if collection_file.exists():
            catalog_dir = input_path_obj
        else:
            # Look for *-chips subdirectory
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

        # Find all chip items (parent items, not child items)
        chip_items = []
        for item_link in collection.get_item_links():
            item_path = catalog_dir / item_link.href
            if item_path.exists():
                item = pystac.Item.from_file(str(item_path))
                # Skip child items (they have _planting/_harvest suffix)
                if (
                    not item.id.endswith("_planting_s2")
                    and not item.id.endswith("_harvest_s2")
                    and not item.id.endswith("_planting_planet")
                    and not item.id.endswith("_harvest_planet")
                ):
                    chip_items.append(item)

        if not chip_items:
            raise click.ClickException("No chip items found in catalog")

    click.echo(f"Year: {year if year else '(extracted from chip IDs)'}")
    click.echo(f"Cloud cover chip threshold: {cloud_cover_chip}%")
    click.echo(f"Buffer: {buffer_days} days, {num_iterations} iterations")
    if verbose:
        click.echo("Verbose mode: ON")

    click.echo(f"\nFound {len(chip_items)} total chips")

    # Pre-scan to categorize chips
    chips_to_process: list[tuple[pystac.Item, int]] = []
    skipped: list[dict] = []
    already_has_count = 0

    for item in chip_items:
        chip_id = item.id
        bbox = tuple(item.bbox) if item.bbox else None

        if bbox is None:
            skipped.append({"chip": chip_id, "reason": "No bbox in item"})
            continue

        # Skip chips that already have Planet scene selections (unless --force)
        if not force and _has_existing_planet_scenes(item):
            skipped.append({"chip": chip_id, "reason": "Already has Planet imagery selections"})
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
        click.echo(f"  Already have Planet imagery: {already_has_count}")
    pre_skipped = len(skipped) - already_has_count
    if pre_skipped > 0:
        click.echo(f"  Skipped (no bbox/year): {pre_skipped}")
    click.echo(f"  To process: {len(chips_to_process)}")

    if not chips_to_process:
        click.echo("\nNo chips need processing.")
        return

    # Track results
    successful: list[str] = []
    failed: list[dict] = []

    # Process chips
    with ImageryProgressBar(total=len(chips_to_process), leave=True, verbose=verbose) as progress:
        for item, chip_year in chips_to_process:
            chip_id = item.id
            progress.start_chip(chip_id)
            bbox = tuple(item.bbox)

            try:
                result = select_planet_scenes_for_chip(
                    client=client,
                    chip_id=chip_id,
                    bbox=bbox,
                    year=chip_year,
                    buffer_days=buffer_days,
                    num_iterations=num_iterations,
                    cloud_cover_chip=cloud_cover_chip,
                    on_progress=progress.on_progress,
                )

                if result.success:
                    # Create child STAC items for planting and harvest
                    _create_planet_child_items(
                        client=client,
                        chip_dir=catalog_dir / item.id,
                        parent_item=item,
                        result=result,
                        year=chip_year,
                        cloud_cover_chip=cloud_cover_chip,
                        buffer_days=buffer_days,
                        num_iterations=num_iterations,
                    )
                    successful.append(chip_id)
                    # Adapt to ImageryProgressBar's expected SceneSelectionResult
                    progress.mark_success_planet(result)
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
    already_has = [s for s in skipped if "Already has" in s.get("reason", "")]
    no_scenes = [s for s in skipped if "No cloud-free" in s.get("reason", "")]
    other_skipped = [s for s in skipped if s not in already_has and s not in no_scenes]

    # Print summary
    click.echo("\n" + "=" * 50)
    click.echo("Summary:")
    click.echo(f"  Newly selected: {len(successful)}")
    click.echo(f"  Already had Planet imagery: {len(already_has)}")
    click.echo(f"  No cloud-free scenes: {len(no_scenes)}")
    if other_skipped:
        click.echo(f"  Other skipped: {len(other_skipped)}")
    click.echo(f"  Failed: {len(failed)}")

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
                "num_iterations": num_iterations,
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
select_images_planet = select_images_planet_cmd
