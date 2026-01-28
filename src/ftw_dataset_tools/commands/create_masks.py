"""CLI command for creating raster masks from vector boundaries."""

import sys

import click

from ftw_dataset_tools.api import masks
from ftw_dataset_tools.api.masks import MaskType


@click.command("create-masks")
@click.argument("chips_file", type=click.Path(exists=True))
@click.argument("boundaries_file", type=click.Path(exists=True))
@click.argument("boundary_lines_file", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(),
    default="./masks",
    show_default=True,
    help="Output directory for mask files.",
)
@click.option(
    "--field-dataset",
    required=True,
    help="Name of the field dataset (used in output filenames).",
)
@click.option(
    "--grid-id-col",
    default="id",
    show_default=True,
    help="Column name for grid cell ID.",
)
@click.option(
    "--mask-type",
    type=click.Choice(["instance", "semantic_2_class", "semantic_3_class"]),
    default="semantic_3_class",
    show_default=True,
    help="Type of mask to create.",
)
@click.option(
    "--coverage-col",
    default="field_coverage_pct",
    show_default=True,
    help="Column name for field coverage percentage (from create-chips).",
)
@click.option(
    "--min-coverage",
    type=float,
    default=0.01,
    show_default=True,
    help="Minimum coverage percentage to process (0.01 skips empty grids).",
)
@click.option(
    "--resolution",
    type=float,
    default=10.0,
    show_default=True,
    help="Pixel resolution in CRS units (e.g., meters).",
)
@click.option(
    "--workers",
    "num_workers",
    type=int,
    default=None,
    help="Number of parallel workers (default: half of CPUs, minimum 1).",
)
def create_masks_cmd(
    chips_file: str,
    boundaries_file: str,
    boundary_lines_file: str,
    output_dir: str,
    field_dataset: str,
    grid_id_col: str,
    mask_type: str,
    coverage_col: str | None,
    min_coverage: float,
    resolution: float,
    num_workers: int | None,
) -> None:
    """Create raster masks from vector boundaries for each grid cell.

    Takes a chips file (from create-chips), boundaries file (polygons),
    and boundary lines file to create raster masks for training data.

    Output files are Cloud Optimized GeoTIFFs (COGs) named:
    {field_dataset}_{grid_id}_{mask_type}.tif

    \b
    CHIPS_FILE: GeoParquet file with chip definitions (from create-chips)
    BOUNDARIES_FILE: GeoParquet file with field boundary polygons
    BOUNDARY_LINES_FILE: GeoParquet file with boundary lines

    \b
    Examples:
        ftwd create-masks chips.parquet fields.parquet boundary_lines_fields.parquet --field-dataset austria
        ftwd create-masks chips.parquet fields.parquet lines.parquet --field-dataset france --mask-type instance
        ftwd create-masks chips.parquet fields.parquet lines.parquet --field-dataset spain --coverage-col field_coverage_pct --min-coverage 1.0
    """
    click.echo(f"Creating {mask_type} masks for {field_dataset}")
    click.echo(f"Output: {output_dir}")

    # Convert mask type string to enum
    mask_type_enum = MaskType(mask_type)

    # Callback to show grid counts before processing
    def on_start(total_grids: int, filtered_grids: int) -> None:
        click.echo(f"Total grids in chips file: {total_grids:,}")
        skipped = total_grids - filtered_grids
        if skipped > 0:
            click.echo(
                f"Grids to process: {filtered_grids:,} "
                f"(skipping {skipped:,} with {coverage_col} < {min_coverage})"
            )
        else:
            click.echo(f"Grids to process: {filtered_grids:,}")

    # Simple progress tracking using carriage return (works well with multiprocessing)
    def on_progress(current: int, total: int) -> None:
        percent = int(100 * current / total) if total > 0 else 0
        bar_width = 40
        filled = int(bar_width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        sys.stdout.write(f"\rCreating masks: |{bar}| {current}/{total} ({percent}%)")
        sys.stdout.flush()

    try:
        result = masks.create_masks(
            chips_file=chips_file,
            boundaries_file=boundaries_file,
            boundary_lines_file=boundary_lines_file,
            output_dir=output_dir,
            field_dataset=field_dataset,
            grid_id_col=grid_id_col,
            mask_type=mask_type_enum,
            coverage_col=coverage_col,
            min_coverage=min_coverage,
            resolution=resolution,
            num_workers=num_workers,
            on_progress=on_progress,
            on_start=on_start,
        )

        # Finish progress line
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Print summary
        click.echo("Summary:")
        click.echo(f"  Field dataset: {result.field_dataset}")
        click.echo(f"  Masks created: {result.total_created}")
        click.echo(f"  Masks skipped: {result.total_skipped}")

        if result.masks_skipped:
            click.echo("\nSkipped grids:")
            for grid_id, reason in result.masks_skipped[:10]:  # Show first 10
                click.echo(f"  {grid_id}: {reason}")
            if len(result.masks_skipped) > 10:
                click.echo(f"  ... and {len(result.masks_skipped) - 10} more")

        click.echo(click.style("Done!", fg="green"))

    except KeyboardInterrupt:
        sys.stdout.write("\n")
        click.echo(click.style("Interrupted by user.", fg="yellow"))
        raise SystemExit(130) from None
    except FileNotFoundError as e:
        sys.stdout.write("\n")
        click.echo(click.style(f"Error: {e}", fg="red"))
        raise SystemExit(1) from e
    except ValueError as e:
        sys.stdout.write("\n")
        click.echo(click.style(f"Error: {e}", fg="red"))
        raise SystemExit(1) from e


# Alias for registration
create_masks = create_masks_cmd
