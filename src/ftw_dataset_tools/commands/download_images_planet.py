"""CLI command for downloading Planet satellite imagery."""

from __future__ import annotations

from pathlib import Path

import click
import pystac
from tqdm import tqdm

from ftw_dataset_tools.api.imagery.planet_client import (
    VALID_BANDS,
    PlanetClient,
)
from ftw_dataset_tools.api.imagery.planet_download import (
    DEFAULT_ASSET_TYPE,
    activate_asset,
    download_and_clip_planet_scene,
)


def _find_planet_child_items(catalog_dir: Path) -> list[tuple[Path, pystac.Item, str]]:
    """Find all Planet child items in a catalog directory.

    Returns:
        List of (item_path, item, season) tuples
    """
    results = []

    # Look for collection.json
    collection_file = catalog_dir / "collection.json"
    if not collection_file.exists():
        return results

    collection = pystac.Collection.from_file(str(collection_file))

    # Find all chip directories
    for item_link in collection.get_item_links():
        item_path = catalog_dir / item_link.href
        if not item_path.exists():
            continue

        item = pystac.Item.from_file(str(item_path))

        # Skip if it's a child item itself
        if "_planting_" in item.id or "_harvest_" in item.id:
            continue

        chip_dir = item_path.parent

        # Look for Planet child items
        for planet_child_name in [
            f"{item.id}_planting_planet.json",
            f"{item.id}_harvest_planet.json",
        ]:
            planet_child_path = chip_dir / planet_child_name
            if planet_child_path.exists():
                child_item = pystac.Item.from_file(str(planet_child_path))
                season = "planting" if "_planting_" in planet_child_name else "harvest"
                results.append((planet_child_path, child_item, season))

    return results


def _has_downloaded_image(item: pystac.Item) -> bool:
    """Check if item already has a downloaded local image asset."""
    image_asset = item.assets.get("image")
    if not image_asset:
        return False
    # Check if it's a local path (not a remote URL)
    return not image_asset.href.startswith("http")


@click.command("download-images-planet")
@click.argument("catalog_path", type=click.Path(exists=True))
@click.option(
    "--bands",
    type=click.Choice(VALID_BANDS),
    multiple=True,
    default=["red", "green", "blue", "nir"],
    show_default=True,
    help="Bands to download (can specify multiple).",
)
@click.option(
    "--asset-type",
    type=click.Choice(
        ["ortho_analytic_4b", "ortho_analytic_4b_sr", "ortho_analytic_8b", "ortho_visual"]
    ),
    default=DEFAULT_ASSET_TYPE,
    show_default=True,
    help="Planet asset type to download.",
)
@click.option(
    "--resolution",
    type=float,
    default=3.0,
    show_default=True,
    help="Target resolution in meters.",
)
@click.option(
    "--activate-only",
    is_flag=True,
    default=False,
    help="Only activate assets, don't download (for batch workflows).",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Skip already downloaded items.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Show detailed progress information.",
)
def download_images_planet_cmd(
    catalog_path: str,
    bands: tuple[str, ...],
    asset_type: str,
    resolution: float,
    activate_only: bool,
    resume: bool,
    verbose: bool,
) -> None:
    """Download and clip Planet imagery for selected scenes.

    Reads scene information from STAC items created by select-images-planet,
    activates assets (Planet requires activation before download), downloads
    the imagery, and clips to chip bounds.

    Use --activate-only to queue all activations without waiting (useful for
    batch workflows where you want to activate many assets in parallel, then
    download later).

    Requires PL_API_KEY environment variable to be set.

    \b
    CATALOG_PATH: Path to dataset directory or chips collection directory

    \b
    Examples:
        # Download with default bands
        ftwd download-images-planet ./my-dataset

        # Download with specific bands
        ftwd download-images-planet ./dataset --bands red --bands green --bands blue --bands nir

        # Activate all assets first (queue activations, exit immediately)
        ftwd download-images-planet ./dataset --activate-only

        # Then download later (assets already active)
        ftwd download-images-planet ./dataset --resume
    """
    catalog_path_obj = Path(catalog_path)

    # Validate Planet API key
    try:
        client = PlanetClient()
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    try:
        client.validate_auth()
    except Exception as e:
        raise click.ClickException(f"Planet API authentication failed: {e}") from e

    click.echo("Planet API: authenticated")

    # Find catalog directory
    collection_file = catalog_path_obj / "collection.json"
    if collection_file.exists():
        catalog_dir = catalog_path_obj
    else:
        # Look for *-chips subdirectory
        chips_dirs = list(catalog_path_obj.glob("*-chips"))
        chips_dir_with_collection = None
        for chips_dir in chips_dirs:
            if (chips_dir / "collection.json").exists():
                chips_dir_with_collection = chips_dir
                break

        if chips_dir_with_collection:
            catalog_dir = chips_dir_with_collection
        else:
            raise click.ClickException(
                f"No collection.json found in {catalog_path} or in any *-chips subdirectory"
            )

    click.echo(f"Catalog: {catalog_dir}")

    # Find Planet child items
    child_items = _find_planet_child_items(catalog_dir)

    if not child_items:
        raise click.ClickException("No Planet child items found. Run select-images-planet first.")

    click.echo(f"Found {len(child_items)} Planet scene items")

    # Filter out already downloaded if --resume
    if resume:
        items_to_process = [
            (path, item, season)
            for path, item, season in child_items
            if not _has_downloaded_image(item)
        ]
        already_downloaded = len(child_items) - len(items_to_process)
        if already_downloaded > 0:
            click.echo(f"  Already downloaded: {already_downloaded}")
            click.echo(f"  To process: {len(items_to_process)}")
    else:
        items_to_process = child_items

    if not items_to_process:
        click.echo("All items already downloaded. Nothing to do.")
        return

    band_list = list(bands)
    click.echo(f"Asset type: {asset_type}")
    click.echo(f"Bands: {band_list}")
    click.echo(f"Resolution: {resolution}m")

    if activate_only:
        # Activate-only mode: queue all activations and exit
        click.echo("\nActivate-only mode: queuing activations...")

        activated = 0
        failed = 0

        with tqdm(total=len(items_to_process), desc="Activating", unit="item") as pbar:
            for _item_path, item, _season in items_to_process:
                scene_id = item.properties.get("ftw:scene_id")
                if not scene_id:
                    pbar.update(1)
                    failed += 1
                    continue

                try:
                    activate_asset(
                        client=client,
                        item_id=scene_id,
                        asset_type=asset_type,
                    )
                    activated += 1
                except Exception as e:
                    if verbose:
                        click.echo(f"\n  Failed to activate {scene_id}: {e}")
                    failed += 1

                pbar.update(1)

        click.echo(f"\nActivation requests sent: {activated}")
        if failed > 0:
            click.echo(click.style(f"Failed: {failed}", fg="red"))

        click.echo("\nAssets are now activating in the background.")
        click.echo("Run without --activate-only to download when ready.")
        return

    # Full download mode
    click.echo(f"\nProcessing {len(items_to_process)} items...")

    successful = 0
    failed_items: list[tuple[str, str]] = []

    with tqdm(total=len(items_to_process), desc="Downloading", unit="item") as pbar:
        for item_path, item, season in items_to_process:
            scene_id = item.properties.get("ftw:scene_id")
            if not scene_id:
                pbar.set_description(f"Skipping {item.id[:20]}... (no scene_id)")
                pbar.update(1)
                continue

            # Get bbox from item
            if not item.bbox:
                pbar.set_description(f"Skipping {item.id[:20]}... (no bbox)")
                pbar.update(1)
                continue

            bbox = tuple(item.bbox)
            chip_dir = item_path.parent
            output_filename = f"{item.id}.tif"
            output_path = chip_dir / output_filename

            pbar.set_description(f"Downloading {scene_id[:20]}...")

            def log_progress(msg: str) -> None:
                if verbose:
                    tqdm.write(f"  {msg}")

            result = download_and_clip_planet_scene(
                client=client,
                item_id=scene_id,
                bbox=bbox,
                output_path=output_path,
                asset_type=asset_type,
                bands=band_list if band_list else None,
                resolution=resolution,
                season=season,
                on_progress=log_progress if verbose else None,
            )

            if result.success:
                # Update STAC item with local asset
                _update_item_with_local_asset(item, item_path, output_filename, band_list)
                successful += 1
            else:
                failed_items.append((item.id, result.error or "Unknown error"))

            pbar.update(1)

    # Print summary
    click.echo("\n" + "=" * 50)
    click.echo("Summary:")
    click.echo(f"  Downloaded: {successful}")
    click.echo(f"  Failed: {len(failed_items)}")

    if failed_items:
        click.echo(click.style(f"\n{len(failed_items)} items failed:", fg="red"))
        for item_id, error in failed_items[:5]:
            click.echo(f"  - {item_id}: {error}")
        if len(failed_items) > 5:
            click.echo(f"  ... and {len(failed_items) - 5} more")

    if successful > 0:
        click.echo(click.style("\nDone!", fg="green"))
    else:
        click.echo(click.style("\nNo items successfully downloaded.", fg="yellow"))


def _update_item_with_local_asset(
    item: pystac.Item,
    item_path: Path,
    output_filename: str,
    band_list: list[str],
) -> None:
    """Update STAC item with local image asset."""
    # Add local image asset
    item.assets["image"] = pystac.Asset(
        href=f"./{output_filename}",
        media_type="image/tiff; application=geotiff; profile=cloud-optimized",
        title=f"Clipped {len(band_list)}-band Planet image ({','.join(band_list)})",
        roles=["data"],
    )

    # Mark as downloaded
    item.properties["ftw:downloaded"] = True

    # Save updated item
    item.save_object(str(item_path))


# Alias for registration
download_images_planet = download_images_planet_cmd
