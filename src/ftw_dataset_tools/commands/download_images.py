"""CLI command for downloading satellite imagery from selected STAC items."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import click
import pystac
from tqdm import tqdm

from ftw_dataset_tools.api.imagery import (
    find_collection_dir,
    find_s2_child_items,
    process_downloaded_scene,
)
from ftw_dataset_tools.api.imagery.download_workflow import (
    DownloadTask,
    build_download_task,
    download_task_scene,
    skip_download_reason,
)
from ftw_dataset_tools.api.imagery.parallel import (
    DEFAULT_WORKERS,
    ParallelOutcome,
    run_in_parallel,
)
from ftw_dataset_tools.api.imagery.thumbnails import has_rgb_bands
from ftw_dataset_tools.api.stac_items import STACSaveError, write_item

if TYPE_CHECKING:
    from ftw_dataset_tools.api.imagery.image_download import DownloadResult

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


def _record_download(
    outcome: ParallelOutcome[DownloadTask, DownloadResult],
    *,
    successful: list[str],
    failed: list[dict],
    band_list: list[str],
    keep_remote_refs: bool,
    generate_thumbnails: bool,
) -> None:
    """Report and record one finished download. Runs on the main thread only.

    A chip's planting and harvest items share a parent item file, so the STAC
    updates below must never run concurrently with each other.
    """
    task = outcome.task
    for message in task.logs:
        if message.startswith("Grid:"):
            tqdm.write(f"  {message}")

    if outcome.error is not None:
        failed.append({"item": task.item.id, "error": str(outcome.error)})
        return

    result = outcome.value
    if result is None or not result.success:
        failed.append({"item": task.item.id, "error": result.error if result else "Unknown error"})
        return

    try:
        _update_stac_items(
            task,
            band_list=band_list,
            keep_remote_refs=keep_remote_refs,
            generate_thumbnails=generate_thumbnails,
        )
    except Exception as e:
        failed.append({"item": task.item.id, "error": str(e)})
        return

    successful.append(task.item.id)


def _update_stac_items(
    task: DownloadTask,
    *,
    band_list: list[str],
    keep_remote_refs: bool,
    generate_thumbnails: bool,
) -> None:
    """Point the STAC items at the downloaded file (or add it alongside the remote refs)."""
    if keep_remote_refs:
        # Legacy mode: keep remote assets, add local as "clipped"
        task.item.assets["clipped"] = pystac.Asset(
            href=f"./{task.output_filename}",
            media_type="image/tiff; application=geotiff; profile=cloud-optimized",
            title=f"Clipped {len(band_list)}-band image ({','.join(band_list)})",
            roles=["data"],
        )
        write_item(task.item, task.item_path)
        return

    try:
        process_downloaded_scene(
            item=task.item,
            item_path=task.item_path,
            output_path=task.output_path,
            output_filename=task.output_filename,
            band_list=band_list,
            season=task.season,
            base_id=task.base_id,
            generate_thumbnails=generate_thumbnails,
        )
    except Exception as e:
        # Clean up downloaded TIF if processing fails
        if task.output_path.exists():
            task.output_path.unlink()
        raise STACSaveError(f"Failed to process {task.item.id}: {e}") from e


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
@click.option(
    "--workers",
    type=int,
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

    \b
    CATALOG_PATH: Path to the dataset directory (containing collection.json)

    \b
    Examples:
        ftwd download-images ./my-dataset
        ftwd download-images ./my-dataset --bands red,green,blue,nir,scl
        ftwd download-images ./my-dataset --keep-remote-refs  # Keep remote asset refs
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

    # Find all child S2 items (planting and harvest). Items whose JSON cannot be
    # read are reported as failures rather than silently dropped from the run.
    unreadable: list[dict] = []
    child_items = find_s2_child_items(catalog_dir, unreadable=unreadable)

    if not child_items and not unreadable:
        raise click.ClickException(
            "No S2 child items found. Run 'select-images' first to create them."
        )

    click.echo(f"\nFound {len(child_items)} S2 items to download")

    # Track results
    successful: list[str] = []
    skipped: list[dict] = []
    failed: list[dict] = list(unreadable)

    # Download several scenes at once: the reads are network-bound, and each one
    # writes only its own GeoTIFF. The STAC writes below stay on this thread,
    # since a chip's two seasons update the same parent item.
    tasks: list[DownloadTask] = []
    for item, item_path in child_items:
        reason = skip_download_reason(item, item_path, resume=resume)
        if reason is not None:
            skipped.append({"item": item.id, "reason": reason})
            continue
        tasks.append(build_download_task(item, item_path))

    generate_thumbnails = preview and has_rgb_bands(band_list)

    with tqdm(total=len(child_items), desc="Downloading imagery", unit="scene") as pbar:
        pbar.update(len(skipped))

        def work(task: DownloadTask) -> DownloadResult:
            return download_task_scene(task, bands=band_list, resolution=resolution)

        def apply(outcome: ParallelOutcome[DownloadTask, DownloadResult]) -> None:
            _record_download(
                outcome,
                successful=successful,
                failed=failed,
                band_list=band_list,
                keep_remote_refs=keep_remote_refs,
                generate_thumbnails=generate_thumbnails,
            )
            pbar.update(1)

        run_in_parallel(tasks, work=work, apply=apply, workers=workers)

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
