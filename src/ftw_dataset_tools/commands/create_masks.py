"""CLI command for creating raster masks from vector boundaries."""

import sys
from pathlib import Path

import click

from ftw_dataset_tools.api import masks, stac
from ftw_dataset_tools.api.config import VALID_MASK_TYPES
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
    help="Dataset root. Masks are written under {output-dir}/chips/{mgrs}/{item_id}/.",
)
@click.option(
    "--field-dataset",
    required=True,
    help="Name of the field dataset (used as the STAC collection id).",
)
@click.option(
    "--year",
    type=int,
    default=None,
    help="Year folded into item IDs and filenames, matching create-dataset.",
)
@click.option(
    "--grid-id-col",
    default="id",
    show_default=True,
    help="Column name for grid cell ID.",
)
@click.option(
    "--mask-type",
    type=click.Choice(list(VALID_MASK_TYPES)),
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
    help="Number of parallel workers (default: CPU count, capped at 8).",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="Reuse masks that are already on disk instead of recreating them.",
)
def create_masks_cmd(
    chips_file: str,
    boundaries_file: str,
    boundary_lines_file: str,
    output_dir: str,
    field_dataset: str,
    year: int | None,
    grid_id_col: str,
    mask_type: str,
    coverage_col: str | None,
    min_coverage: float,
    resolution: float,
    num_workers: int | None,
    skip_existing: bool,
) -> None:
    """Create raster masks from vector boundaries for each grid cell.

    Takes a chips file (from create-chips), boundaries file (polygons),
    and boundary lines file to create raster masks for training data.

    Masks are Cloud Optimized GeoTIFFs written into the same STAC catalog
    layout create-dataset produces, alongside a STAC collection describing them:

    \b
        {output-dir}/collection.json
        {output-dir}/chips/{mgrs_square}/{item_id}/{item_id}.json
        {output-dir}/chips/{mgrs_square}/{item_id}/{item_id}_{mask_type}.tif

    \b
    CHIPS_FILE: GeoParquet file with chip definitions (from create-chips)
    BOUNDARIES_FILE: GeoParquet file with field boundary polygons
    BOUNDARY_LINES_FILE: GeoParquet file with boundary lines

    \b
    Examples:
        ftwd create-masks chips.parquet fields.parquet boundary_lines_fields.parquet --field-dataset austria --year 2024
        ftwd create-masks chips.parquet fields.parquet lines.parquet --field-dataset france --mask-type instance --year 2024
        ftwd create-masks chips.parquet fields.parquet lines.parquet --field-dataset spain --min-coverage 1.0 --year 2024

    \b
    --year may be omitted when the boundaries file has a determination_datetime
    column, which the STAC collection's temporal extent is then read from.
    """
    # The STAC collection written at the end needs a temporal extent. Checked before
    # any rasterization so a missing --year fails in a second rather than after a
    # full mask run.
    if year is None and stac.detect_datetime_column(boundaries_file) is None:
        raise click.BadParameter(
            "Cannot determine the collection's temporal extent: "
            f"{boundaries_file} has no 'determination_datetime' column. "
            "Pass --year.",
            param_hint="--year",
        )

    click.echo(f"Creating {mask_type} masks for {field_dataset}")
    click.echo(f"Output: {Path(output_dir) / 'chips'}")

    # Convert mask type string to enum
    mask_type_enum = MaskType(mask_type)

    # Callback to show grid counts before processing
    def on_start(total_grids: int, filtered_grids: int, total_tasks: int) -> None:
        click.echo(f"Total grids in chips file: {total_grids:,}")
        skipped = total_grids - filtered_grids
        # total_tasks is what the progress bar counts to: one rasterization per
        # grid, less any that --skip-existing found already on disk.
        tasks = f" -> {total_tasks:,} rasterization tasks" if total_tasks != filtered_grids else ""
        if skipped > 0:
            click.echo(
                f"Grids to process: {filtered_grids:,} "
                f"(skipping {skipped:,} with {coverage_col} < {min_coverage}){tasks}"
            )
        else:
            click.echo(f"Grids to process: {filtered_grids:,}{tasks}")

    # Simple progress tracking using carriage return (works well with multiprocessing)
    def on_progress(current: int, total: int) -> None:
        percent = int(100 * current / total) if total > 0 else 0
        bar_width = 40
        filled = int(bar_width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        sys.stdout.write(f"\rCreating masks: |{bar}| {current}/{total} ({percent}%)")
        sys.stdout.flush()

    try:
        results = masks.create_masks(
            chips_file=chips_file,
            boundaries_file=boundaries_file,
            boundary_lines_file=boundary_lines_file,
            output_dir=output_dir,
            field_dataset=field_dataset,
            grid_id_col=grid_id_col,
            mask_types=[mask_type_enum],
            coverage_col=coverage_col,
            min_coverage=min_coverage,
            resolution=resolution,
            num_workers=num_workers,
            year=year,
            skip_existing=skip_existing,
            on_progress=on_progress,
            on_start=on_start,
        )
        # This command builds one mask type at a time.
        result = results[mask_type_enum]

        # Finish progress line
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Print summary
        click.echo("Summary:")
        click.echo(f"  Field dataset: {result.field_dataset}")
        click.echo(f"  Masks created: {result.total_created}")
        click.echo(f"  Masks skipped: {result.total_skipped}")
        # Shared with the pipeline so both report reused outputs and a pool that
        # had to be restarted the same way.
        for line in masks.mask_run_summary_lines([result]):
            click.echo(line)

        if result.masks_skipped:
            click.echo("\nSkipped grids:")
            for grid_id, reason in result.masks_skipped[:10]:  # Show first 10
                click.echo(f"  {grid_id}: {reason}")
            if len(result.masks_skipped) > 10:
                click.echo(f"  ... and {len(result.masks_skipped) - 10} more")

        # Without this the command leaves the catalog's directory shape with no
        # catalog in it, and select-images/download-images both require a
        # collection.json to run against the output.
        click.echo("\nGenerating STAC catalog...")
        stac_result = stac.generate_stac_catalog(
            output_dir=output_dir,
            field_dataset=field_dataset,
            fields_file=boundaries_file,
            chips_file=chips_file,
            boundary_lines_file=boundary_lines_file,
            year=year,
        )
        click.echo(
            f"  Created STAC collection with {stac_result.total_items} item(s) "
            f"in {len(stac_result.subcatalog_paths)} sub-catalog(s)"
        )

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
