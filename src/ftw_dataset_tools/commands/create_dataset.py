"""CLI command for creating complete training datasets from field boundaries."""

import sys
from pathlib import Path

import click

from ftw_dataset_tools.api import dataset, splits


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
    help=f"Dataset train/val/test split strategy ({splits.SPLIT_TYPE_CHOICES_STR}).",
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

    \b
    Output structure:
        {name}-dataset/
        ├── catalog.json
        ├── source/
        │   └── collection.json
        ├── chips/
        │   ├── collection.json
        │   ├── items.parquet
        │   └── {grid_id}/
        ├── {dataset}_fields.parquet
        ├── {dataset}_chips.parquet
        ├── {dataset}_boundary_lines.parquet
        └── label_masks/
            ├── instance/
            ├── semantic_2class/
            └── semantic_3class/

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
        # Validate at CLI layer for immediate user feedback with proper Click error formatting
        try:
            validated_split_percents = splits.validate_split_percents(split_percents)
        except ValueError as err:
            raise click.BadParameter(str(err), param_hint="split-percents") from err

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

        if result.splits_result:
            click.echo(f"  Splits: {result.splits_result.train_count} train, {result.splits_result.val_count} val, {result.splits_result.test_count} test")

        click.echo(f"  Total masks created: {result.total_masks_created:,}")

        click.echo("")
        click.echo("Output files:")
        click.echo(f"  Fields: {result.fields_file}")
        click.echo(f"  Chips: {result.chips_file}")
        click.echo(f"  Boundary lines: {result.boundary_lines_file}")
        click.echo("  Masks:")
        for mask_type, mask_dir in result.mask_dirs.items():
            if mask_type in result.masks_results:
                count = result.masks_results[mask_type].total_created
                click.echo(f"    {mask_type}: {mask_dir} ({count:,} files)")

        if result.stac_result:
            click.echo("")
            click.echo("STAC Catalog:")
            click.echo(f"  Catalog: {result.stac_result.catalog_path}")
            click.echo(f"  Source collection: {result.stac_result.source_collection_path}")
            click.echo(f"  Chips collection: {result.stac_result.chips_collection_path}")
            click.echo(f"  Items: {result.stac_result.total_items:,}")
            click.echo(f"  Items parquet: {result.stac_result.items_parquet_path}")

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


# Alias for registration
create_dataset = create_dataset_cmd
