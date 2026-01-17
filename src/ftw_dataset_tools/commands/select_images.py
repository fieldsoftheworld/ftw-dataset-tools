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
    SceneSelectionResult,
    select_scenes_for_chip,
)


def _extract_year_from_chip_id(chip_id: str) -> int | None:
    """Extract year from chip ID (e.g., 'ftw-34UFF1628_2024' -> 2024)."""
    match = re.search(r"_(\d{4})$", chip_id)
    if match:
        return int(match.group(1))
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
    "--stac-host",
    type=click.Choice(["earthsearch", "mspc"]),
    default="earthsearch",
    show_default=True,
    help="STAC host to query for imagery.",
)
@click.option(
    "--cloud-cover-scene",
    type=int,
    default=30,
    show_default=True,
    help="Maximum scene-level cloud cover percentage.",
)
@click.option(
    "--buffer-days",
    type=int,
    default=14,
    show_default=True,
    help="Days to search around crop calendar dates.",
)
@click.option(
    "--pixel-check",
    is_flag=True,
    default=False,
    help="Enable pixel-level cloud filtering using cloud mask COGs.",
)
@click.option(
    "--cloud-cover-pixel",
    type=float,
    default=0.0,
    show_default=True,
    help="Maximum pixel-level cloud cover when --pixel-check is enabled.",
)
@click.option(
    "--s2-collection",
    type=click.Choice(["c1", "l2a"]),
    default="c1",
    show_default=True,
    help="Sentinel-2 collection (earthsearch only). c1=Collection-1, l2a=L2A.",
)
@click.option(
    "--on-missing",
    type=click.Choice(["skip", "fail", "best-available"]),
    default="skip",
    show_default=True,
    help="How to handle chips with no cloud-free scenes.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from previous run using progress file.",
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
def select_images_cmd(
    input_path: str,
    year: int | None,
    stac_host: Literal["earthsearch", "mspc"],
    cloud_cover_scene: int,
    buffer_days: int,
    pixel_check: bool,
    cloud_cover_pixel: float,
    s2_collection: str,
    on_missing: Literal["skip", "fail", "best-available"],
    resume: bool,  # noqa: ARG001 - planned feature
    output_report: str | None,
    verbose: bool,
) -> None:
    """Select optimal Sentinel-2 imagery for chips.

    Queries STAC catalogs to find cloud-free Sentinel-2 scenes for each chip
    based on crop calendar dates (planting and harvest). Creates child STAC items
    with remote asset links.

    \b
    INPUT_PATH: Either a STAC catalog directory (containing collection.json)
                or a single chip JSON file for testing.

    \b
    Examples:
        ftwd select-images ./my-dataset-chips
        ftwd select-images ./chips/ftw-34UFF1628_2024/ftw-34UFF1628_2024.json -v
        ftwd select-images ./chips --year 2023 --stac-host mspc
        ftwd select-images ./chips --cloud-cover-scene 5 --pixel-check
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
        # Catalog directory
        catalog_dir = input_path_obj
        collection_file = catalog_dir / "collection.json"

        if not collection_file.exists():
            raise click.ClickException(f"No collection.json found in {input_path}")

        click.echo(f"Catalog: {input_path}")

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

    if year:
        click.echo(f"Year: {year} (from --year option)")
    else:
        click.echo("Year: (extracted from chip IDs)")
    click.echo(f"STAC host: {stac_host}")
    click.echo(f"Cloud cover threshold: {cloud_cover_scene}%")
    if pixel_check:
        click.echo(f"Pixel-level check enabled (threshold: {cloud_cover_pixel}%)")
    if verbose:
        click.echo("Verbose mode: ON")

    click.echo(f"\nFound {len(chip_items)} chips to process")

    # Track results
    successful: list[str] = []
    skipped: list[dict] = []
    failed: list[dict] = []

    # Progress callback - only output if verbose
    def on_progress(msg: str) -> None:
        if verbose:
            click.echo(f"  {msg}")

    # Helper to format progress bar status
    def _format_status(
        ok: int, skip: int, fail: int, last_result: SceneSelectionResult | None
    ) -> dict:
        status = {"ok": ok, "skip": skip, "fail": fail}
        if last_result and last_result.success:
            p_cc = last_result.planting_scene.cloud_cover if last_result.planting_scene else 0
            h_cc = last_result.harvest_scene.cloud_cover if last_result.harvest_scene else 0
            status["last"] = f"P:{p_cc:.0f}%/H:{h_cc:.0f}%"
        return status

    # Process each chip
    with tqdm(total=len(chip_items), desc="Selecting imagery", unit="chip") as pbar:
        last_result: SceneSelectionResult | None = None
        pbar.set_postfix(_format_status(0, 0, 0, None))

        for item in chip_items:
            chip_id = item.id
            bbox = tuple(item.bbox) if item.bbox else None

            if bbox is None:
                skipped.append({"chip": chip_id, "reason": "No bbox in item"})
                pbar.set_postfix(
                    _format_status(len(successful), len(skipped), len(failed), last_result)
                )
                pbar.update(1)
                continue

            # Determine the year for this chip
            chip_year = year
            if chip_year is None:
                chip_year = _extract_year_from_chip_id(chip_id)
                if chip_year is None:
                    skipped.append(
                        {
                            "chip": chip_id,
                            "reason": "No year provided and could not extract from chip ID",
                        }
                    )
                    pbar.set_postfix(
                        _format_status(len(successful), len(skipped), len(failed), last_result)
                    )
                    pbar.update(1)
                    continue

            try:
                result = select_scenes_for_chip(
                    chip_id=chip_id,
                    bbox=bbox,
                    year=chip_year,
                    stac_host=stac_host,
                    cloud_cover_scene=cloud_cover_scene,
                    buffer_days=buffer_days,
                    pixel_check=pixel_check,
                    cloud_cover_pixel=cloud_cover_pixel,
                    s2_collection=s2_collection,
                    on_progress=on_progress,
                )

                if result.success:
                    # Create child STAC items for planting and harvest
                    _create_child_items(
                        catalog_dir=catalog_dir,
                        parent_item=item,
                        result=result,
                        year=chip_year,
                        stac_host=stac_host,
                        cloud_cover_scene=cloud_cover_scene,
                        buffer_days=buffer_days,
                        pixel_check=pixel_check,
                    )
                    successful.append(chip_id)
                    last_result = result
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

            except Exception as e:
                if on_missing == "fail":
                    raise
                failed.append({"chip": chip_id, "error": str(e)})

            pbar.set_postfix(
                _format_status(len(successful), len(skipped), len(failed), last_result)
            )
            pbar.update(1)

    # Print summary
    click.echo("\n" + "=" * 50)
    click.echo("Summary:")
    click.echo(f"  Successful: {len(successful)}")
    click.echo(f"  Skipped: {len(skipped)}")
    click.echo(f"  Failed: {len(failed)}")

    if skipped:
        click.echo(click.style(f"\n{len(skipped)} chips skipped:", fg="yellow"))
        for s in skipped[:5]:
            click.echo(f"  - {s['chip']}: {s['reason']}")
        if len(skipped) > 5:
            click.echo(f"  ... and {len(skipped) - 5} more")

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
                "stac_host": stac_host,
                "cloud_cover_scene": cloud_cover_scene,
                "buffer_days": buffer_days,
                "pixel_check": pixel_check,
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
    stac_host: str,
    cloud_cover_scene: int,
    buffer_days: int,
    pixel_check: bool,
) -> None:
    """Create child STAC items for planting and harvest scenes."""
    chip_dir = catalog_dir / parent_item.id

    # Update parent item with FTW properties
    parent_item.properties["ftw:calendar_year"] = year
    parent_item.properties["ftw:planting_day"] = result.crop_calendar.planting_day
    parent_item.properties["ftw:harvest_day"] = result.crop_calendar.harvest_day
    parent_item.properties["ftw:stac_host"] = stac_host
    parent_item.properties["ftw:cloud_cover_scene_threshold"] = cloud_cover_scene
    parent_item.properties["ftw:buffer_days"] = buffer_days
    parent_item.properties["ftw:pixel_check"] = pixel_check

    # Set temporal extent from selected scenes
    if result.planting_scene and result.harvest_scene:
        parent_item.properties["start_datetime"] = result.planting_scene.datetime.isoformat()
        parent_item.properties["end_datetime"] = result.harvest_scene.datetime.isoformat()

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

    # Save updated parent
    parent_path = chip_dir / f"{parent_item.id}.json"
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

    # Save child item
    child_path = chip_dir / f"{child_id}.json"
    child_item.save_object(dest_href=str(child_path))


# Alias for registration
select_images = select_images_cmd
