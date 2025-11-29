"""CLI command for adding field coverage statistics to grid cells."""

import click
from tqdm import tqdm

from ftw_dataset_tools.api import field_stats
from ftw_dataset_tools.api.geo import CRSMismatchError


@click.command("add-field-stats")
@click.argument("grid_file", type=click.Path(exists=True))
@click.argument("fields_file", type=click.Path(exists=True))
@click.option(
    "--grid-geom-col",
    default=None,
    help="Column name for grid geometry (auto-detected from GeoParquet metadata if not specified).",
)
@click.option(
    "--fields-geom-col",
    default=None,
    help="Column name for fields geometry (auto-detected from GeoParquet metadata if not specified).",
)
@click.option(
    "--grid-bbox-col",
    default=None,
    help="Column name for grid bbox (auto-detected if not specified).",
)
@click.option(
    "--fields-bbox-col",
    default=None,
    help="Column name for fields bbox (auto-detected if not specified).",
)
@click.option(
    "-o",
    "--output",
    "output_file",
    type=click.Path(),
    default=None,
    help="Output file path. If not specified, updates input file in place.",
)
@click.option(
    "--coverage-col",
    default="field_coverage_pct",
    show_default=True,
    help="Name for the new coverage percentage column.",
)
@click.option(
    "--min-coverage",
    type=float,
    default=None,
    help="Exclude grid cells with coverage below this percentage (e.g., 0.01 to remove cells with 0%%).",
)
@click.option(
    "--reproject",
    "reproject_to_4326",
    is_flag=True,
    default=False,
    help="Reproject both inputs to EPSG:4326 if CRS don't match.",
)
def add_field_stats_cmd(
    grid_file: str,
    fields_file: str,
    grid_geom_col: str | None,
    fields_geom_col: str | None,
    grid_bbox_col: str | None,
    fields_bbox_col: str | None,
    output_file: str | None,
    coverage_col: str,
    min_coverage: float | None,
    reproject_to_4326: bool,
) -> None:
    """Add field coverage statistics to each grid cell.

    Calculates what percentage of each grid cell is covered by field boundary
    polygons using DuckDB's spatial extension.

    The grid and fields files must have the same CRS. If they don't match,
    use --reproject to automatically reproject both to EPSG:4326.

    \b
    GRID_FILE: Parquet file containing grid geometries (e.g., MGRS cells)
    FIELDS_FILE: Parquet file containing field boundary polygons

    \b
    Examples:
        ftwd add-field-stats grid.parquet fields.parquet
        ftwd add-field-stats grid.parquet fields.parquet -o output.parquet
        ftwd add-field-stats grid.parquet fields.parquet --reproject
        ftwd add-field-stats grid.parquet fields.parquet --coverage-col pct_fields
    """
    click.echo(f"Grid file: {grid_file}")
    click.echo(f"Fields file: {fields_file}")

    # Progress callback that prints messages
    def on_progress(msg: str) -> None:
        if msg.startswith("Warning:"):
            click.echo(click.style(msg, fg="yellow"))
        elif "CRS mismatch" in msg or "reprojecting" in msg.lower():
            click.echo(click.style(msg, fg="cyan"))
        elif "optimization" in msg.lower():
            if "disabled" in msg.lower():
                click.echo(click.style(msg, fg="yellow"))
            else:
                click.echo(click.style(msg, fg="green"))
        else:
            click.echo(msg)

    try:
        # Show progress bar during calculation
        with tqdm(total=100, desc="Processing", unit="%") as pbar:
            pbar.update(10)

            result = field_stats.add_field_stats(
                grid_file=grid_file,
                fields_file=fields_file,
                output_file=output_file,
                grid_geom_col=grid_geom_col,
                fields_geom_col=fields_geom_col,
                grid_bbox_col=grid_bbox_col,
                fields_bbox_col=fields_bbox_col,
                coverage_col=coverage_col,
                min_coverage=min_coverage,
                reproject_to_4326=reproject_to_4326,
                on_progress=on_progress,
            )

            pbar.update(90)

        # Print summary
        click.echo("\nSummary:")
        click.echo(f"  Total grid cells: {result.total_cells:,}")
        click.echo(
            f"  Cells with field coverage: {result.cells_with_coverage:,} "
            f"({result.coverage_percentage:.1f}%)"
        )
        click.echo(f"  Average coverage: {result.average_coverage}%")
        click.echo(f"  Maximum coverage: {result.max_coverage}%")
        click.echo(f"\nOutput written to: {result.output_path}")

        click.echo(click.style("Done!", fg="green"))

    except CRSMismatchError as e:
        click.echo(click.style(f"\nError: {e}", fg="red"))
        click.echo(
            click.style(
                "\nHint: Use --reproject to automatically reproject both files to EPSG:4326",
                fg="yellow",
            )
        )
        raise SystemExit(1) from e


# Alias for registration
add_field_stats = add_field_stats_cmd
