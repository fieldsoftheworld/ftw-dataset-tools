"""CLI command for downloading satellite imagery from selected STAC items."""

from __future__ import annotations

import contextlib
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
from ftw_dataset_tools.api.imagery.thumbnails import (
    ThumbnailError,
    generate_thumbnail,
    has_rgb_bands,
)
from ftw_dataset_tools.api.stac_items import STACSaveError, update_parent_item

# All valid Sentinel-2 bands from EarthSearch
VALID_BANDS: tuple[str, ...] = (
    # Visible bands
    "coastal",  # B01 - Coastal aerosol (60m)
    "blue",  # B02 - Blue (10m)
    "green",  # B03 - Green (10m)
    "red",  # B04 - Red (10m)
    # Red edge bands
    "rededge1",  # B05 - Vegetation red edge 1 (20m)
    "rededge2",  # B06 - Vegetation red edge 2 (20m)
    "rededge3",  # B07 - Vegetation red edge 3 (20m)
    # NIR bands
    "nir",  # B08 - NIR (10m)
    "nir08",  # B8A - NIR narrow (20m)
    "nir09",  # B09 - Water vapour (60m)
    # SWIR bands
    "swir16",  # B11 - SWIR 1.6μm (20m)
    "swir22",  # B12 - SWIR 2.2μm (20m)
    # Atmospheric
    "aot",  # Aerosol Optical Thickness
    "wvp",  # Water Vapour
    # Classification/masks
    "scl",  # Scene Classification Layer
    "cloud",  # Cloud probability
    "snow",  # Snow probability
    # Composite
    "visual",  # True color RGB composite
)


@click.command("download-images")
@click.argument("catalog_path", type=click.Path(exists=True))
@click.option(
    "--bands",
    type=click.Choice(VALID_BANDS, case_sensitive=False),
    multiple=True,
    default=("red", "green", "blue", "nir"),
    show_default=True,
    help="Sentinel-2 bands to download. Can be specified multiple times.",
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
@click.option(
    "--keep-remote-refs",
    is_flag=True,
    default=False,
    help="Keep remote asset references instead of replacing with local paths.",
)
@click.option(
    "--preview/--no-preview",
    default=True,
    show_default=True,
    help="Generate JPEG preview thumbnails for downloaded images.",
)
def download_images_cmd(
    catalog_path: str,
    bands: tuple[str, ...],
    resolution: float,
    resume: bool,
    output_report: str | None,
    keep_remote_refs: bool,
    preview: bool,
) -> None:
    """Download and clip satellite imagery for selected scenes.

    Reads child STAC items (created by select-images) with remote asset links
    and downloads/clips the imagery to the chip's bounding box.

    By default, updates STAC items to point to local files:
    - Child items: replaces band assets with local "image" asset
    - Parent chip items: adds planting_image/harvest_image assets

    Use --keep-remote-refs to keep original remote references and add a separate
    "clipped" asset for the local file.

    \b
    CATALOG_PATH: Path to the chips collection or dataset directory

    \b
    Examples:
        ftwd download-images ./my-dataset-chips
        ftwd download-images ./my-dataset              # Also works with dataset dir
        ftwd download-images ./chips --bands red,green,blue,nir,scl
        ftwd download-images ./chips --keep-remote-refs  # Keep remote asset refs
    """
    input_path = Path(catalog_path)
    collection_file = input_path / "collection.json"

    if collection_file.exists():
        catalog_dir = input_path
    else:
        # Look for *-chips subdirectory with collection.json (dataset directory)
        chips_dirs = list(input_path.glob("*-chips"))
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
                f"No collection.json found in {catalog_path} or in any *-chips subdirectory"
            )

    band_list = list(bands)

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
                    local_asset = pystac.Asset(
                        href=f"./{output_filename}",
                        media_type="image/tiff; application=geotiff; profile=cloud-optimized",
                        title=f"Clipped {len(band_list)}-band image ({','.join(band_list)})",
                        roles=["data"],
                    )

                    if keep_remote_refs:
                        # Keep remote assets, add local as "clipped"
                        item.assets["clipped"] = local_asset
                    else:
                        # Replace remote band assets with single local "image" asset
                        # Remove the downloaded band assets (keep others like scl, cloud)
                        for band in band_list:
                            item.assets.pop(band, None)
                        item.assets["image"] = local_asset

                    # Generate thumbnail if requested and RGB bands are available
                    thumbnail_path = None
                    if preview and has_rgb_bands(band_list):
                        try:
                            thumbnail_filename = output_filename.replace(".tif", ".jpg")
                            thumbnail_path = output_path.parent / thumbnail_filename
                            generate_thumbnail(output_path, thumbnail_path)
                            item.assets["thumbnail"] = pystac.Asset(
                                href=f"./{thumbnail_filename}",
                                media_type=pystac.MediaType.JPEG,
                                title="JPEG preview",
                                roles=["thumbnail"],
                            )
                        except ThumbnailError:
                            # Thumbnail generation failed, but don't fail the whole download
                            thumbnail_path = None

                    try:
                        item.save_object(str(item_path))
                    except Exception as e:
                        # Clean up downloaded TIF and thumbnail if save fails
                        if output_path.exists():
                            output_path.unlink()
                        if thumbnail_path and thumbnail_path.exists():
                            thumbnail_path.unlink()
                        raise STACSaveError(f"Failed to save child item {item.id}: {e}") from e

                    # Update parent chip item with asset reference
                    parent_item_path = item_path.parent / f"{base_id}.json"
                    if parent_item_path.exists():
                        parent_item = pystac.Item.from_file(str(parent_item_path))
                        # Pass thumbnail filename for planting season
                        thumb_for_parent = None
                        if thumbnail_path and season == "planting":
                            thumb_for_parent = thumbnail_path.name
                        # Suppress errors - child item was saved successfully
                        with contextlib.suppress(STACSaveError):
                            update_parent_item(
                                parent_item=parent_item,
                                parent_path=parent_item_path,
                                season=season,
                                output_filename=output_filename,
                                band_list=band_list,
                                thumbnail_filename=thumb_for_parent,
                            )

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
