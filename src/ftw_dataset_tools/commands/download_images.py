"""CLI command for downloading satellite imagery from selected STAC items."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import click
import pystac
from tqdm import tqdm

from ftw_dataset_tools.api.imagery import (
    download_and_clip_scene,
)
from ftw_dataset_tools.api.imagery.scene_selection import SelectedScene


@click.command("download-images")
@click.argument("catalog_path", type=click.Path(exists=True))
@click.option(
    "--bands",
    default="red,green,blue,nir",
    show_default=True,
    help="Comma-separated list of bands to download.",
)
@click.option(
    "--resolution",
    type=float,
    default=10.0,
    show_default=True,
    help="Target resolution in meters.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from previous run, skipping already downloaded images.",
)
@click.option(
    "--output-report",
    type=click.Path(),
    default=None,
    help="Path for JSON report of download results.",
)
def download_images_cmd(
    catalog_path: str,
    bands: str,
    resolution: float,
    resume: bool,
    output_report: str | None,
) -> None:
    """Download and clip satellite imagery for selected scenes.

    Reads child STAC items (created by select-images) with remote asset links
    and downloads/clips the imagery to the chip's bounding box. Adds a "clipped"
    asset to each child item pointing to the local file.

    \b
    CATALOG_PATH: Path to the STAC catalog directory (containing collection.json)

    \b
    Examples:
        ftwd download-images ./my-dataset-chips
        ftwd download-images ./chips --bands red,green,blue,nir,scl
        ftwd download-images ./chips --resolution 10 --resume
    """
    catalog_dir = Path(catalog_path)
    collection_file = catalog_dir / "collection.json"

    if not collection_file.exists():
        raise click.ClickException(f"No collection.json found in {catalog_path}")

    band_list = [b.strip() for b in bands.split(",")]

    click.echo(f"Catalog: {catalog_path}")
    click.echo(f"Bands: {band_list}")
    click.echo(f"Resolution: {resolution}m")

    # Load collection to find items
    collection = pystac.Collection.from_file(str(collection_file))

    # Find all child S2 items (planting and harvest)
    child_items = []
    for item_link in collection.get_item_links():
        item_path = catalog_dir / item_link.href
        if item_path.exists():
            item = pystac.Item.from_file(str(item_path))
            # Only process child items (they have _planting_s2 or _harvest_s2 suffix)
            if item.id.endswith("_planting_s2") or item.id.endswith("_harvest_s2"):
                child_items.append((item, item_path))

    # Also search in subdirectories for child items
    for subdir in catalog_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("."):
            for json_file in subdir.glob("*_s2.json"):
                try:
                    item = pystac.Item.from_file(str(json_file))
                    is_s2_item = item.id.endswith("_planting_s2") or item.id.endswith("_harvest_s2")
                    not_duplicate = not any(i.id == item.id for i, _ in child_items)
                    if is_s2_item and not_duplicate:
                        child_items.append((item, json_file))
                except Exception:
                    pass  # Skip invalid JSON files

    if not child_items:
        raise click.ClickException(
            "No S2 child items found. Run 'select-images' first to create them."
        )

    click.echo(f"\nFound {len(child_items)} S2 items to download")

    # Track results
    successful: list[str] = []
    skipped: list[dict] = []
    failed: list[dict] = []

    # Progress callback
    def on_progress(msg: str) -> None:
        pass  # Suppress individual band progress for cleaner output

    # Process each child item
    with tqdm(total=len(child_items), desc="Downloading imagery", unit="scene") as pbar:
        for item, item_path in child_items:
            bbox = tuple(item.bbox) if item.bbox else None

            if bbox is None:
                skipped.append({"item": item.id, "reason": "No bbox in item"})
                pbar.update(1)
                continue

            # Check if already downloaded (resume mode)
            if resume and "clipped" in item.assets:
                clipped_href = item.assets["clipped"].href
                clipped_path = item_path.parent / clipped_href
                if clipped_path.exists():
                    skipped.append({"item": item.id, "reason": "Already downloaded"})
                    pbar.update(1)
                    continue

            # Determine season from item ID
            if item.id.endswith("_planting_s2"):
                season: Literal["planting", "harvest"] = "planting"
            else:
                season = "harvest"

            # Construct output filename
            # e.g., ftw-34UFF1628_2024_planting_image_s2.tif
            base_id = item.id.replace("_planting_s2", "").replace("_harvest_s2", "")
            output_filename = f"{base_id}_{season}_image_s2.tif"
            output_path = item_path.parent / output_filename

            try:
                # Create a minimal SelectedScene from the STAC item
                scene = SelectedScene(
                    item=item,
                    season=season,
                    cloud_cover=item.properties.get("eo:cloud_cover", 0.0),
                    datetime=item.datetime,
                    stac_url=item.get_self_href() or "",
                )

                result = download_and_clip_scene(
                    scene=scene,
                    bbox=bbox,
                    output_path=output_path,
                    bands=band_list,
                    resolution=resolution,
                    on_progress=on_progress,
                )

                if result.success:
                    # Update item with clipped asset
                    item.assets["clipped"] = pystac.Asset(
                        href=f"./{output_filename}",
                        media_type="image/tiff; application=geotiff; profile=cloud-optimized",
                        title=f"Clipped {len(band_list)}-band image ({','.join(band_list)})",
                        roles=["data"],
                    )
                    item.save_object(str(item_path))
                    successful.append(item.id)
                else:
                    failed.append({"item": item.id, "error": result.error})

            except Exception as e:
                failed.append({"item": item.id, "error": str(e)})

            pbar.update(1)

    # Print summary
    click.echo("\n" + "=" * 50)
    click.echo("Summary:")
    click.echo(f"  Downloaded: {len(successful)}")
    click.echo(f"  Skipped: {len(skipped)}")
    click.echo(f"  Failed: {len(failed)}")

    if skipped:
        # Count skipped by reason
        already_downloaded = sum(1 for s in skipped if s["reason"] == "Already downloaded")
        other_skipped = len(skipped) - already_downloaded
        if already_downloaded:
            click.echo(click.style(f"\n{already_downloaded} items already downloaded", fg="cyan"))
        if other_skipped:
            click.echo(click.style(f"{other_skipped} items skipped for other reasons", fg="yellow"))

    if failed:
        click.echo(click.style(f"\n{len(failed)} items failed:", fg="red"))
        for f in failed[:5]:
            click.echo(f"  - {f['item']}: {f['error']}")
        if len(failed) > 5:
            click.echo(f"  ... and {len(failed) - 5} more")

    # Write report if requested
    if output_report:
        report = {
            "total_processed": len(child_items),
            "successful": len(successful),
            "skipped": skipped,
            "failed": failed,
            "parameters": {
                "bands": band_list,
                "resolution": resolution,
            },
        }
        report_path = Path(output_report)
        report_path.write_text(json.dumps(report, indent=2))
        click.echo(f"\nReport written to: {report_path}")

    if successful:
        click.echo(click.style("\nDone!", fg="green"))
    elif skipped and not failed:
        click.echo(click.style("\nAll items were already downloaded.", fg="cyan"))
    else:
        click.echo(click.style("\nNo items successfully downloaded.", fg="yellow"))


# Alias for registration
download_images = download_images_cmd
