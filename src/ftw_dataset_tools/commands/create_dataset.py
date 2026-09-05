"""CLI command for creating complete training datasets from field boundaries."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import click
import pystac
from tqdm import tqdm

from ftw_dataset_tools.api import crop_stats, dataset, splits
from ftw_dataset_tools.api.config import DEFAULT_MASK_TYPES, PMTILES_AUTO, VALID_MASK_TYPES
from ftw_dataset_tools.api.imagery import (
    download_and_clip_scene,
    iter_chip_dirs,
    process_downloaded_scene,
    select_imagery_for_catalog,
)
from ftw_dataset_tools.api.imagery.scene_selection import SelectedScene
from ftw_dataset_tools.api.imagery.thumbnails import has_rgb_bands
from ftw_dataset_tools.api.pipeline import docs_summary_line
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
    "--split-type",
    type=click.Choice(splits.SPLIT_TYPE_CHOICES),
    required=True,
    help=(
        "Dataset train/val/test split strategy. "
        "Use 'block3x3' for spatially coherent 3x3 blocks, or "
        "'random-uniform' for random chip assignment across the dataset, or "
        "'predefined' to use a split column from the input fields file. "
        f"Available choices: {splits.SPLIT_TYPE_CHOICES_STR}."
    ),
)
@click.option(
    "--split-percents",
    nargs=3,
    type=click.IntRange(0, 100),
    default=(80, 10, 10),
    show_default=True,
    metavar="TRAIN VAL TEST",
    help="Train/val/test split percentages (must sum to 100).",
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
    help=(
        "Pixel resolution in meters for masks; also used as imagery fallback "
        "when no reference mask grid is found."
    ),
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
@click.option(
    "--force-image-selection",
    is_flag=True,
    default=False,
    help=(
        "Re-select imagery for chips that already have selections. Note: re-running "
        "create-dataset regenerates the STAC catalog, which clears prior selections, "
        "so every chip is re-selected either way; to resume an interrupted selection "
        "on an existing catalog, use select-images instead."
    ),
)
@click.option(
    "--mask-types",
    type=str,
    default=",".join(DEFAULT_MASK_TYPES),
    show_default=True,
    help=(
        "Comma-separated list of mask types to generate. One or more of: "
        f"{', '.join(VALID_MASK_TYPES)}. The decode_* layers are derived from "
        "the 2-class mask and are off by default."
    ),
)
@click.option(
    "--presence-only",
    is_flag=True,
    default=False,
    help="Indicates labels are presence-only; background class value will be 3 instead of 0.",
)
@click.option(
    "--drop-border-chips",
    is_flag=True,
    default=False,
    help="Remove chips touching outer boundary (edges of convex hull). Useful when fields at boundary may have partial coverage.",
)
@click.option(
    "--class-filter",
    "class_filter",
    type=click.Path(exists=True),
    default=None,
    help=(
        "Path to a class filter YAML (column + include/exclude lists). Include "
        "classes count as field; all other classes are treated as background. "
        "Every distinct class value must be listed in include or exclude."
    ),
)
@click.option(
    "--checksums",
    is_flag=True,
    default=False,
    help="Add file:checksum (multihash sha256) to every STAC asset. Slow on large datasets.",
)
def create_dataset_cmd(
    fields_file: str,
    output_dir: str | None,
    field_dataset: str | None,
    split_type: str,
    split_percents: tuple[int, int, int],
    min_coverage: float,
    resolution: float,
    num_workers: int | None,
    skip_reproject: bool,
    year: int | None,
    skip_images: bool,
    download_images: bool,
    cloud_cover_chip: float,
    nodata_max: float,
    buffer_days: int,
    num_buffer_expansions: int,
    buffer_expansion_size: int,
    force_image_selection: bool,
    mask_types: str,
    presence_only: bool,
    drop_border_chips: bool,
    class_filter: str | None,
    checksums: bool,
) -> None:
    """Create a complete training dataset from a fields file.

    Takes a single fields file (GeoParquet with polygon geometries) and creates:

    \b
    - Chips file with field coverage statistics
    - Train/val/test split assignments
    - Boundary lines file
    - Mask types (instance, semantic_2_class, semantic_3_class) - configurable via --mask-types
      (decode_boundary and decode_distance are also available, off by default)
    - STAC static catalog with items for each chip

    If the input file is not in EPSG:4326, it will be automatically reprojected.

    For temporal extent, uses determination_datetime from fiboa if present,
    otherwise requires --year to specify the year range.

    Output structure::

        {name}-dataset/
        ├── collection.json
        ├── chips/
        │   └── {mgrs_square}/
        │       └── {grid_id}/
        │           ├── {grid_id}.json
        │           ├── {grid_id}_instance.tif
        │           ├── {grid_id}_semantic_2_class.tif
        │           └── {grid_id}_semantic_3_class.tif
        ├── {name}_fields.parquet
        ├── {name}_chips.parquet
        └── {name}_boundary_lines.parquet

    \b
    FIELDS_FILE: GeoParquet file with field boundary polygons

    \b
    Examples:
        ftwd create-dataset austria_fields.parquet --split-type random-uniform --year 2023
        ftwd create-dataset fields.parquet --split-type block3x3 --field-dataset austria -o ./austria_dataset --year 2022
        ftwd create-dataset fields.parquet --split-type random-uniform --min-coverage 1.0 --resolution 5.0 --year 2024
        ftwd create-dataset fields.parquet --split-type block3x3 --mask-types semantic_2_class,semantic_3_class --year 2023
        ftwd create-dataset fields.parquet --split-type block3x3 --presence-only --year 2023
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
        # Validate at CLI layer for immediate user feedback with proper Click error formatting
        try:
            validated_split_percents = splits.validate_split_percents(split_percents)
        except ValueError as err:
            raise click.BadParameter(str(err), param_hint="split-percents") from err

        # Parse and validate mask types
        mask_types_list = [mt.strip() for mt in mask_types.split(",")]
        for mask_type in mask_types_list:
            if mask_type not in VALID_MASK_TYPES:
                raise click.BadParameter(
                    f"Invalid mask type '{mask_type}'. "
                    f"Must be one of: {', '.join(VALID_MASK_TYPES)}",
                    param_hint="mask-types",
                )

        result = dataset.create_dataset(
            fields_file=fields_file,
            output_dir=output_dir,
            field_dataset=field_dataset,
            split_type=split_type,
            split_percents=validated_split_percents,
            min_coverage=min_coverage,
            resolution=resolution,
            num_workers=num_workers,
            skip_reproject=skip_reproject,
            year=year,
            mask_types=mask_types_list,
            presence_only=presence_only,
            drop_border_chips=drop_border_chips,
            class_filter=class_filter,
            checksums=checksums,
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
            click.echo(f"  {crop_stats.crop_stats_summary(result.crop_stats_result)}")

        if result.splits_result:
            click.echo(
                f"  Splits: {result.splits_result.train_count} train, {result.splits_result.val_count} val, {result.splits_result.test_count} test"
            )

        click.echo(f"  Total masks created: {result.total_masks_created:,}")

        click.echo("")
        click.echo("Output files:")
        click.echo(f"  Fields: {result.fields_file}")
        click.echo(f"  Chips: {result.chips_file}")
        click.echo(f"  Boundary lines: {result.boundary_lines_file}")
        click.echo("  Masks:")
        if result.chips_base_dir:
            click.echo(f"    Location: {result.chips_base_dir}/<mgrs>/{{grid_id}}/")
        for mask_type, mask_result in result.masks_results.items():
            click.echo(f"    {mask_type}: {mask_result.total_created:,} files")

        if result.stac_result:
            click.echo("")
            click.echo("STAC Catalog:")
            click.echo(f"  Collection: {result.stac_result.collection_path}")
            click.echo(f"  Sub-catalogs: {len(result.stac_result.subcatalog_paths):,}")
            click.echo(f"  Items: {result.stac_result.total_items:,}")
            click.echo(f"  Items parquet: {result.stac_result.items_parquet_path}")

        if result.docs_result is not None:
            click.echo("")
            # create-dataset has no flag to change stages.docs.pmtiles, so it always
            # runs in "auto" mode: tiles/styles when tippecanoe is available, skipped
            # (not an error) otherwise.
            click.echo(docs_summary_line(result.docs_result, PMTILES_AUTO))

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

            # The imagery workflow walks the collection directory's chip sub-catalogs.
            catalog_dir = Path(result.stac_result.collection_path).parent

            # Shared workflow, also used by `ftwd run`. It records
            # ftw:planting/ftw:harvest links on each parent chip, so a later
            # `select-images` run resumes instead of starting over. (Re-running
            # create-dataset itself regenerates the catalog and clears those
            # links before this point, so selection here always starts fresh.)
            selection = select_imagery_for_catalog(
                catalog_dir=catalog_dir,
                year=effective_year,
                cloud_cover_chip=cloud_cover_chip,
                nodata_max=nodata_max,
                buffer_days=buffer_days,
                num_buffer_expansions=num_buffer_expansions,
                buffer_expansion_size=buffer_expansion_size,
                force=force_image_selection,
            )

            click.echo(f"  Selected: {selection.successful}")
            click.echo(f"  Skipped: {selection.skipped}")
            if selection.failed:
                click.echo(click.style(f"  Failed: {selection.failed}", fg="yellow"))

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
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        click.echo(click.style(f"\nError: {e}", fg="red"))
        raise SystemExit(1) from e


def _run_image_download(
    catalog_dir: Path,
    bands: list[str],
    resolution: float,
) -> dict:
    """Run image download for all S2 child items in a catalog.

    Downloads imagery, generates thumbnails, and creates overlay thumbnails
    when semantic masks are available. Updates both child and parent STAC items.

    Uses shared process_downloaded_scene() for consistent behavior with
    the standalone download-images command.
    """
    # Find all child S2 items
    child_items = []
    for subdir in iter_chip_dirs(catalog_dir):
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
    band_list = list(bands)
    can_generate_thumbnail = has_rgb_bands(band_list)

    def on_download_progress(msg: str) -> None:
        if msg.startswith("Grid:"):
            tqdm.write(f"  {msg}")

    with tqdm(
        total=len(child_items), desc="Downloading imagery", unit="scene", leave=False
    ) as pbar:
        for item, item_path in child_items:
            bbox = tuple(item.bbox) if item.bbox else None

            if bbox is None or "clipped" in item.assets or "image" in item.assets:
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
                    bands=band_list,
                    resolution=resolution,
                    on_progress=on_download_progress,
                )

                if result.success:
                    # Use shared processing logic
                    process_downloaded_scene(
                        item=item,
                        item_path=item_path,
                        output_path=output_path,
                        output_filename=output_filename,
                        band_list=band_list,
                        season=season,
                        base_id=base_id,
                        generate_thumbnails=can_generate_thumbnail,
                    )
                    successful += 1
                else:
                    failed += 1

            except Exception:
                failed += 1

            pbar.update(1)

    return {"successful": successful, "skipped": skipped, "failed": failed}


# Alias for registration
create_dataset = create_dataset_cmd
