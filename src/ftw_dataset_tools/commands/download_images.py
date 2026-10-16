"""CLI command for downloading satellite imagery from selected STAC items."""

from __future__ import annotations

import json
from pathlib import Path

import click
from tqdm import tqdm

from ftw_dataset_tools.api.imagery import download_imagery_for_catalog, find_collection_dir
from ftw_dataset_tools.api.imagery.parallel import DEFAULT_WORKERS, MAX_WORKERS

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


def _echo_grid_log(message: str) -> None:
    if message.startswith("Grid:"):
        tqdm.write(f"  {message}")


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
    help=(
        "Target resolution in meters when no reference mask grid is found. "
        "If a co-located mask exists, its CRS/transform/size is used instead."
    ),
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help=(
        "Leave chips that already have local imagery alone. Use --no-resume to fetch "
        "every chip again, which is the only way to pick up a changed --bands or "
        "--resolution (the skip is on the local file existing, not on what is inside it). "
        "--no-resume needs remote band refs, so re-run select-images --force first."
    ),
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
    help="Generate WebP preview thumbnails for downloaded images.",
)
@click.option(
    "--workers",
    type=click.IntRange(1, MAX_WORKERS),
    default=DEFAULT_WORKERS,
    show_default=True,
    help="Scenes to download concurrently. STAC item writes stay serialized.",
)
def download_images_cmd(
    catalog_path: str,
    bands: tuple[str, ...],
    resolution: float,
    resume: bool,
    output_report: str | None,
    keep_remote_refs: bool,
    preview: bool,
    workers: int,
) -> None:
    """Download and clip satellite imagery for selected scenes.

    Reads child STAC items (created by select-images) with remote asset links
    and downloads/clips the imagery to the chip's bounding box.

    By default, updates STAC items to point to local files:
    - Child items: replaces band assets with local "image" asset
    - Parent chip items: adds planting_image/harvest_image assets

    Use --keep-remote-refs to keep original remote references and add a separate
    "clipped" asset for the local file.

    Chips that already have local imagery are left alone, so a second run picks up
    where the first stopped. Pass --no-resume to fetch every chip again.

    \b
    CATALOG_PATH: Path to the dataset directory (containing collection.json)

    \b
    Examples:
        ftwd download-images ./my-dataset
        ftwd download-images ./my-dataset --bands red,green,blue,nir,scl
        ftwd download-images ./my-dataset --keep-remote-refs  # Keep remote asset refs
        ftwd download-images ./my-dataset --no-resume  # Re-fetch every chip
    """
    input_path = Path(catalog_path)
    try:
        catalog_dir = find_collection_dir(input_path)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e

    band_list = list(bands)

    click.echo(f"Catalog: {catalog_path}")
    click.echo(f"Bands: {band_list}")
    click.echo(f"Resolution: {resolution}m")

    result = download_imagery_for_catalog(
        catalog_dir,
        bands=band_list,
        resolution=resolution,
        generate_thumbnails=preview,
        resume=resume,
        workers=workers,
        keep_remote_refs=keep_remote_refs,
        on_log=_echo_grid_log,
    )
    skipped = result.skipped_details
    failed = result.failed_details
    total = result.successful + result.skipped + result.failed

    if total == 0:
        raise click.ClickException(
            "No S2 child items found. Run 'select-images' first to create them."
        )

    # Print summary
    click.echo("\n" + "=" * 50)
    click.echo("Summary:")
    click.echo(f"  Downloaded: {result.successful}")
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
            "total_processed": total,
            "successful": result.successful,
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

    if result.successful:
        click.echo(click.style("\nDone!", fg="green"))
    elif skipped and not failed:
        click.echo(click.style("\nAll items were already downloaded.", fg="cyan"))
    else:
        click.echo(click.style("\nNo items successfully downloaded.", fg="yellow"))


# Alias for registration
download_images = download_images_cmd
