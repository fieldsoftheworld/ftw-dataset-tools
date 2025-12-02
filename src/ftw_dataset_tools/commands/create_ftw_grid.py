"""CLI command for creating hierarchical FTW grids."""

from pathlib import Path

import click
from tqdm import tqdm

from ftw_dataset_tools.api import ftw_grid


@click.command("create-ftw-grid")
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(),
    default=None,
    help="Output path. Required for folder input (hive-partitioned output). "
    "For file input, defaults to ftw_<gzd>.parquet.",
)
@click.option(
    "--km-size",
    type=int,
    default=2,
    show_default=True,
    help="Grid cell size in km. Must divide 100 evenly (1, 2, 4, 5, 10, 20, 25, 50, 100).",
)
def create_ftw_grid_cmd(
    input_path: str,
    output_path: str | None,
    km_size: int,
) -> None:
    """Create a hierarchical FTW grid from 1km MGRS cells.

    Groups 1km MGRS cells into larger grid cells based on the specified km_size
    and unions their geometries. Never crosses GZD or 100km square boundaries.

    INPUT_PATH can be either:

    \b
    - A single GeoParquet file (must contain exactly one GZD)
    - A folder containing partitioned parquet files (multiple GZDs supported)

    For folder input, --output is required and will create a hive-partitioned
    output folder with gzd= partitions.

    Required columns: GZD (or gzd), MGRS (or mgrs)

    The output includes:

    \b
    - gzd: Grid Zone Designator
    - mgrs_10km: 10km MGRS code from source
    - id: Unique FTW grid cell ID (e.g., "ftw-33UXPA0410")
    - geometry: Unioned polygon of child cells

    \b
    Examples:
        ftwd create-ftw-grid mgrs_1km.parquet
        ftwd create-ftw-grid mgrs_1km.parquet --km-size 4
        ftwd create-ftw-grid ./mgrs_partitioned/ -o ./ftw_output/
    """
    input_p = Path(input_path)
    is_folder = input_p.is_dir()

    # Validate output is provided for folder input
    if is_folder and output_path is None:
        raise click.UsageError("Output path (-o/--output) is required when input is a folder")

    click.echo(f"Input: {input_path} ({'folder' if is_folder else 'file'})")
    click.echo(f"Grid size: {km_size}x{km_size}km")

    # Progress bar for file processing (folder mode only)
    pbar = None

    def on_progress(msg: str) -> None:
        # Close progress bar before printing messages
        if pbar is not None:
            pbar.clear()
        if msg.startswith("Warning:"):
            click.echo(click.style(msg, fg="yellow"))
        else:
            click.echo(msg)
        if pbar is not None:
            pbar.refresh()

    def on_file_progress(_current: int, total: int, file_name: str) -> None:
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total, desc="Processing files", unit="file")
        pbar.set_postfix_str(file_name[:30] + "..." if len(file_name) > 30 else file_name)
        pbar.update(1)

    try:
        result = ftw_grid.create_ftw_grid(
            input_path=input_path,
            output_path=output_path,
            km_size=km_size,
            on_progress=on_progress,
            on_file_progress=on_file_progress if is_folder else None,
        )

        if pbar:
            pbar.close()

        click.echo("\nSummary:")
        if result.gzd:
            click.echo(f"  GZD: {result.gzd}")
        else:
            click.echo(f"  GZDs processed: {result.gzd_count}")
        click.echo(f"  Grid cell size: {result.km_size}x{result.km_size}km")
        click.echo(f"  Total cells: {result.total_cells:,}")
        click.echo(f"\nOutput written to: {result.output_path}")
        click.echo(click.style("Done!", fg="green"))

    except (ftw_grid.InvalidKmSizeError, ftw_grid.MultipleGZDError, ValueError) as e:
        if pbar:
            pbar.close()
        click.echo(click.style(f"\nError: {e}", fg="red"))
        raise SystemExit(1) from e


# Alias for registration
create_ftw_grid = create_ftw_grid_cmd
