"""CLI command for creating complete training datasets from field boundaries."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import click
import pystac
from tqdm import tqdm

from ftw_dataset_tools.api import dataset
from ftw_dataset_tools.api.imagery import (
    ImageryProgressBar,
    download_and_clip_scene,
    select_scenes_for_chip,
)
from ftw_dataset_tools.api.imagery.scene_selection import SelectedScene
from ftw_dataset_tools.api.stac import detect_datetime_column, get_year_from_datetime_column


@click.command("create-dataset")
@click.argument("fields_file", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory for all generated files. Defaults to {input_stem}-dataset/",
)
@click.option(
    "--field-dataset",
    default=None,
    help="Name for the dataset (used in output filenames). Defaults to input filename stem.",
)
@click.option(
    "--min-coverage",
    type=float,
    default=0.01,
    show_default=True,
    help="Minimum coverage percentage to include grids.",
)
@click.option(
    "--resolution",
    type=float,
    default=10.0,
    show_default=True,
    help="Pixel resolution in meters for masks.",
)
@click.option(
    "--workers",
    "num_workers",
    type=int,
    default=None,
    help="Number of parallel workers for mask creation (default: half of CPUs).",
)
@click.option(
    "--skip-reproject",
    is_flag=True,
    default=False,
    help="Fail if input is not EPSG:4326 instead of auto-reprojecting.",
)
@click.option(
    "--year",
    type=int,
    default=None,
    help="Year for temporal extent (required if fields lack determination_datetime column).",
)
@click.option(
    "--skip-images",
    is_flag=True,
    default=False,
    help="Skip image selection (by default, imagery is selected after mask creation).",
)
@click.option(
    "--download-images",
    is_flag=True,
    default=False,
    help="Download images after selection.",
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
    default=10,
    show_default=True,
    help="Maximum scene-level cloud cover percentage for STAC query filter.",
)
@click.option(
    "--buffer-days",
    type=int,
    default=14,
    show_default=True,
    help="Days to search around crop calendar dates.",
)
@click.option(
    "--pixel-check/--no-pixel-check",
    default=True,
    show_default=True,
    help="Enable pixel-level cloud cover analysis using SCL band.",
)
@click.option(
    "--cloud-cover-pixel",
    type=float,
    default=0.0,
    show_default=True,
    help="Maximum pixel-level cloud cover percentage (0.0 = require fully clear).",
)
@click.option(
    "--num-buffer-expansions",
    type=int,
    default=3,
    show_default=True,
    help="Number of times to expand date buffer if no cloud-free scenes found.",
)
@click.option(
    "--buffer-expansion-size",
    type=int,
    default=14,
    show_default=True,
    help="Days to add to buffer on each expansion.",
)
def create_dataset_cmd(
    fields_file: str,
    output_dir: str | None,
    field_dataset: str | None,
    min_coverage: float,
    resolution: float,
    num_workers: int | None,
    skip_reproject: bool,
    year: int | None,
    skip_images: bool,
    download_images: bool,
    stac_host: str,
    cloud_cover_scene: int,
    buffer_days: int,
    pixel_check: bool,
    cloud_cover_pixel: float,
    num_buffer_expansions: int,
    buffer_expansion_size: int,
) -> None:
    """Create a complete training dataset from a fields file.

    Takes a single fields file (GeoParquet with polygon geometries) and creates:

    \b
    - Chips file with field coverage statistics
    - Boundary lines file
    - All three mask types (instance, semantic_2class, semantic_3class)
    - STAC static catalog with items for each chip

    If the input file is not in EPSG:4326, it will be automatically reprojected.

    For temporal extent, uses determination_datetime from fiboa if present,
    otherwise requires --year to specify the year range.

    Output structure::

        {name}-dataset/
        ├── catalog.json
        ├── {name}-source/
        │   └── collection.json
        ├── {name}-chips/
        │   ├── collection.json
        │   ├── items.parquet
        │   └── {grid_id}/
        │       ├── {grid_id}.json
        │       ├── {grid_id}_instance.tif
        │       ├── {grid_id}_semantic_2_class.tif
        │       └── {grid_id}_semantic_3_class.tif
        ├── {name}_fields.parquet
        ├── {name}_chips.parquet
        └── {name}_boundary_lines.parquet

    \b
    FIELDS_FILE: GeoParquet file with field boundary polygons

    \b
    Examples:
        ftwd create-dataset austria_fields.parquet --year 2023
        ftwd create-dataset fields.parquet --field-dataset austria -o ./austria_dataset --year 2022
        ftwd create-dataset fields.parquet --min-coverage 1.0 --resolution 5.0 --year 2024
    """
    # Derive output directory from input filename if not specified
    if output_dir is None:
        input_stem = Path(fields_file).stem
        output_dir = f"{input_stem}-dataset"

    click.echo(click.style("Creating dataset from fields file", fg="cyan", bold=True))
    click.echo(f"Input: {fields_file}")
    click.echo(f"Output: {output_dir}")

    # Progress callback for general messages
    def on_progress(msg: str) -> None:
        if msg.startswith("Warning:"):
            click.echo(click.style(msg, fg="yellow"))
        elif "Error" in msg:
            click.echo(click.style(msg, fg="red"))
        elif "Reprojecting" in msg or "CRS" in msg:
            click.echo(click.style(msg, fg="cyan"))
        elif "complete" in msg.lower():
            click.echo(click.style(msg, fg="green"))
        else:
            click.echo(msg)

    # Track current mask type for progress display
    current_mask_info = {"type": "", "total": 0}

    # Progress callback for mask creation
    def on_mask_progress(current: int, total: int) -> None:
        percent = int(100 * current / total) if total > 0 else 0
        bar_width = 40
        filled = int(bar_width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        sys.stdout.write(f"\r  Creating masks: |{bar}| {current}/{total} ({percent}%)")
        sys.stdout.flush()

    def on_mask_start(total_grids: int, filtered_grids: int) -> None:
        current_mask_info["total"] = filtered_grids
        skipped = total_grids - filtered_grids
        if skipped > 0:
            click.echo(
                f"  Processing {filtered_grids:,} grids (skipping {skipped:,} below threshold)"
            )
        else:
            click.echo(f"  Processing {filtered_grids:,} grids")

    try:
        result = dataset.create_dataset(
            fields_file=fields_file,
            output_dir=output_dir,
            field_dataset=field_dataset,
            min_coverage=min_coverage,
            resolution=resolution,
            num_workers=num_workers,
            skip_reproject=skip_reproject,
            year=year,
            on_progress=on_progress,
            on_mask_progress=on_mask_progress,
            on_mask_start=on_mask_start,
        )

        # Finish any progress line
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Print summary
        click.echo("")
        click.echo(click.style("Dataset created successfully!", fg="green", bold=True))
        click.echo("")
        click.echo("Summary:")
        click.echo(f"  Dataset name: {result.field_dataset}")
        if result.was_reprojected:
            click.echo(f"  Reprojected from: {result.source_crs}")

        if result.chips_result:
            click.echo(f"  Grid cells: {result.chips_result.total_cells:,}")
            click.echo(f"  Cells with coverage: {result.chips_result.cells_with_coverage:,}")

        click.echo(f"  Total masks created: {result.total_masks_created:,}")

        click.echo("")
        click.echo("Output files:")
        click.echo(f"  Fields: {result.fields_file}")
        click.echo(f"  Chips: {result.chips_file}")
        click.echo(f"  Boundary lines: {result.boundary_lines_file}")
        click.echo("  Masks:")
        if result.chips_base_dir:
            click.echo(f"    Location: {result.chips_base_dir}/{{grid_id}}/")
        for mask_type, mask_result in result.masks_results.items():
            click.echo(f"    {mask_type}: {mask_result.total_created:,} files")

        if result.stac_result:
            click.echo("")
            click.echo("STAC Catalog:")
            click.echo(f"  Catalog: {result.stac_result.catalog_path}")
            click.echo(f"  Source collection: {result.stac_result.source_collection_path}")
            click.echo(f"  Chips collection: {result.stac_result.chips_collection_path}")
            click.echo(f"  Items: {result.stac_result.total_items:,}")
            click.echo(f"  Items parquet: {result.stac_result.items_parquet_path}")

        # Image selection (by default enabled, unless --skip-images is set)
        should_select_images = not skip_images or download_images
        if should_select_images:
            # Try to extract year from determination_datetime if not provided
            effective_year = year
            if effective_year is None:
                datetime_col = detect_datetime_column(fields_file)
                if datetime_col:
                    effective_year = get_year_from_datetime_column(fields_file, datetime_col)
                    if effective_year:
                        click.echo(f"  Year: {effective_year} (from {datetime_col})")

            if effective_year is None:
                raise click.ClickException(
                    "--year is required for image selection "
                    "(no determination_datetime column found). "
                    "Use --skip-images to skip image selection."
                )

            click.echo("")
            click.echo(click.style("Selecting imagery...", fg="cyan", bold=True))

            # Get chips collection path
            chips_collection_path = Path(result.stac_result.chips_collection_path)
            catalog_dir = chips_collection_path.parent

            # Run image selection with full parameters
            image_stats = _run_image_selection(
                catalog_dir=catalog_dir,
                year=effective_year,
                stac_host=stac_host,
                cloud_cover_scene=cloud_cover_scene,
                buffer_days=buffer_days,
                pixel_check=pixel_check,
                cloud_cover_pixel=cloud_cover_pixel,
                num_buffer_expansions=num_buffer_expansions,
                buffer_expansion_size=buffer_expansion_size,
            )

            click.echo(f"  Selected: {image_stats['successful']}")
            click.echo(f"  Skipped: {image_stats['skipped']}")
            if image_stats["failed"]:
                click.echo(click.style(f"  Failed: {image_stats['failed']}", fg="yellow"))

            # Download if requested
            if download_images:
                click.echo("")
                click.echo(click.style("Downloading imagery...", fg="cyan", bold=True))

                download_stats = _run_image_download(
                    catalog_dir=catalog_dir,
                    bands=["red", "green", "blue", "nir"],
                    resolution=resolution,
                )

                click.echo(f"  Downloaded: {download_stats['successful']}")
                if download_stats["skipped"]:
                    click.echo(f"  Skipped: {download_stats['skipped']}")
                if download_stats["failed"]:
                    click.echo(click.style(f"  Failed: {download_stats['failed']}", fg="yellow"))

    except KeyboardInterrupt:
        sys.stdout.write("\n")
        click.echo(click.style("Interrupted by user.", fg="yellow"))
        raise SystemExit(130) from None
    except FileNotFoundError as e:
        click.echo(click.style(f"\nError: {e}", fg="red"))
        raise SystemExit(1) from e
    except ValueError as e:
        click.echo(click.style(f"\nError: {e}", fg="red"))
        raise SystemExit(1) from e


def _run_image_selection(
    catalog_dir: Path,
    year: int,
    stac_host: str,
    cloud_cover_scene: int,
    buffer_days: int,
    pixel_check: bool = True,
    cloud_cover_pixel: float = 0.0,
    num_buffer_expansions: int = 3,
    buffer_expansion_size: int = 14,
) -> dict:
    """Run image selection for all chips in a catalog."""
    # Find all chip items (parent items, not child S2 items)
    chip_items = []
    for subdir in catalog_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("."):
            for json_file in subdir.glob("*.json"):
                # Skip child items
                if "_planting_s2" in json_file.name or "_harvest_s2" in json_file.name:
                    continue
                try:
                    item = pystac.Item.from_file(str(json_file))
                    chip_items.append((item, json_file))
                except Exception:
                    pass

    # Process chips with unified progress display
    with ImageryProgressBar(total=len(chip_items), leave=False, verbose=False) as progress:
        for item, item_path in chip_items:
            progress.start_chip(item.id)
            bbox = tuple(item.bbox) if item.bbox else None

            if bbox is None:
                progress.mark_skipped("No bbox in item")
                continue

            try:
                result = select_scenes_for_chip(
                    chip_id=item.id,
                    bbox=bbox,
                    year=year,
                    stac_host=stac_host,
                    cloud_cover_scene=cloud_cover_scene,
                    buffer_days=buffer_days,
                    pixel_check=pixel_check,
                    cloud_cover_pixel=cloud_cover_pixel,
                    num_buffer_expansions=num_buffer_expansions,
                    buffer_expansion_size=buffer_expansion_size,
                    on_progress=progress.on_progress,
                )

                if result.success:
                    # Create child STAC items
                    _create_child_items_inline(
                        chip_dir=item_path.parent,
                        parent_item=item,
                        result=result,
                        year=year,
                        stac_host=stac_host,
                        cloud_cover_scene=cloud_cover_scene,
                        buffer_days=buffer_days,
                        pixel_check=pixel_check,
                        num_buffer_expansions=num_buffer_expansions,
                        buffer_expansion_size=buffer_expansion_size,
                    )
                    progress.mark_success(result)
                else:
                    progress.mark_skipped(result.skipped_reason or "No cloud-free scenes")

            except Exception as e:
                progress.mark_failed(str(e))

    return progress.get_stats_dict()


def _create_child_items_inline(
    chip_dir: Path,
    parent_item: pystac.Item,
    result,
    year: int,
    stac_host: str,
    cloud_cover_scene: int,
    buffer_days: int,
    pixel_check: bool = True,
    num_buffer_expansions: int = 3,
    buffer_expansion_size: int = 14,
) -> None:
    """Create child STAC items for planting and harvest scenes (inline version)."""
    # Update parent item with FTW properties
    parent_item.properties["ftw:calendar_year"] = year
    parent_item.properties["ftw:planting_day"] = result.crop_calendar.planting_day
    parent_item.properties["ftw:harvest_day"] = result.crop_calendar.harvest_day
    parent_item.properties["ftw:stac_host"] = stac_host
    parent_item.properties["ftw:cloud_cover_scene_threshold"] = cloud_cover_scene
    parent_item.properties["ftw:buffer_days"] = buffer_days
    parent_item.properties["ftw:pixel_check"] = pixel_check
    parent_item.properties["ftw:num_buffer_expansions"] = num_buffer_expansions
    parent_item.properties["ftw:buffer_expansion_size"] = buffer_expansion_size

    # Set temporal extent from selected scenes
    if result.planting_scene and result.harvest_scene:
        parent_item.properties["start_datetime"] = result.planting_scene.datetime.isoformat()
        parent_item.properties["end_datetime"] = result.harvest_scene.datetime.isoformat()

    # Save updated parent
    parent_path = chip_dir / f"{parent_item.id}.json"
    parent_item.save_object(str(parent_path))

    # Create child items for each season
    for scene, season in [(result.planting_scene, "planting"), (result.harvest_scene, "harvest")]:
        if scene:
            child_id = f"{parent_item.id}_{season}_s2"
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

            # Copy relevant band assets
            for band in ["red", "green", "blue", "nir", "scl", "visual"]:
                if band in scene.item.assets:
                    child_item.assets[band] = scene.item.assets[band].clone()

            if "cloud" in scene.item.assets:
                child_item.assets["cloud_probability"] = scene.item.assets["cloud"].clone()

            # Add links
            child_item.add_link(
                pystac.Link(
                    rel="derived_from",
                    target=f"./{parent_item.id}.json",
                    media_type="application/json",
                )
            )

            if scene.stac_url:
                child_item.add_link(
                    pystac.Link(rel="via", target=scene.stac_url, media_type="application/json")
                )

            if scene.cloud_cover < 0.1:
                child_item.properties["eo:cloud_cover"] = scene.cloud_cover

            child_path = chip_dir / f"{child_id}.json"
            child_item.save_object(str(child_path))


def _run_image_download(
    catalog_dir: Path,
    bands: list[str],
    resolution: float,
) -> dict:
    """Run image download for all S2 child items in a catalog."""
    # Find all child S2 items
    child_items = []
    for subdir in catalog_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("."):
            for json_file in subdir.glob("*_s2.json"):
                try:
                    item = pystac.Item.from_file(str(json_file))
                    if item.id.endswith("_planting_s2") or item.id.endswith("_harvest_s2"):
                        child_items.append((item, json_file))
                except Exception:
                    pass

    successful = 0
    skipped = 0
    failed = 0

    with tqdm(
        total=len(child_items), desc="Downloading imagery", unit="scene", leave=False
    ) as pbar:
        for item, item_path in child_items:
            bbox = tuple(item.bbox) if item.bbox else None

            if bbox is None or "clipped" in item.assets:
                skipped += 1
                pbar.update(1)
                continue

            # Determine season from item ID
            if item.id.endswith("_planting_s2"):
                season: Literal["planting", "harvest"] = "planting"
            else:
                season = "harvest"

            # Construct output filename
            base_id = item.id.replace("_planting_s2", "").replace("_harvest_s2", "")
            output_filename = f"{base_id}_{season}_image_s2.tif"
            output_path = item_path.parent / output_filename

            try:
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
                    bands=bands,
                    resolution=resolution,
                )

                if result.success:
                    item.assets["clipped"] = pystac.Asset(
                        href=f"./{output_filename}",
                        media_type="image/tiff; application=geotiff; profile=cloud-optimized",
                        title=f"Clipped {len(bands)}-band image ({','.join(bands)})",
                        roles=["data"],
                    )
                    item.save_object(str(item_path))
                    successful += 1
                else:
                    failed += 1

            except Exception:
                failed += 1

            pbar.update(1)

    return {"successful": successful, "skipped": skipped, "failed": failed}


# Alias for registration
create_dataset = create_dataset_cmd
