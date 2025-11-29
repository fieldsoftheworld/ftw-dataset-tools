"""CLI command for fetching MGRS grid cells from cloud source."""

import click

from ftw_dataset_tools.api import grid


@click.command("get-grid")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    "output_file",
    type=click.Path(),
    default=None,
    help="Output file path. If not specified, creates <input>_grid.parquet.",
)
@click.option(
    "--precise",
    is_flag=True,
    default=False,
    help="Use geometry union for precise matching (slower but excludes grids in bbox gaps).",
)
@click.option(
    "--grid-source",
    default=grid.DEFAULT_GRID_SOURCE,
    show_default=True,
    help="URL/path to the grid source.",
)
def get_grid_cmd(
    input_file: str,
    output_file: str | None,
    precise: bool,
    grid_source: str,
) -> None:
    """Fetch MGRS grid cells that cover the input file's extent.

    Downloads grid cells from a cloud-based GeoParquet source that intersect
    with the bounding box of the input file. The input file must be in EPSG:4326.

    By default, uses bounding box matching (fast). Use --precise for geometry-based
    matching which excludes grids that fall in gaps/corners of the actual geometries.

    \b
    INPUT_FILE: GeoParquet file defining the area of interest (must be EPSG:4326)

    \b
    Examples:
        ftwd get-grid fields.parquet
        ftwd get-grid fields.parquet --precise
        ftwd get-grid fields.parquet -o custom_grid.parquet
    """
    click.echo(f"Input file: {input_file}")

    def on_progress(msg: str) -> None:
        click.echo(msg)

    try:
        result = grid.get_grid(
            input_file=input_file,
            output_file=output_file,
            grid_source=grid_source,
            precise=precise,
            on_progress=on_progress,
        )

        click.echo("\nSummary:")
        click.echo(f"  Grid cells: {result.grid_count:,}")
        xmin, ymin, xmax, ymax = result.bounds
        click.echo(f"  Bounds: [{xmin:.6f}, {ymin:.6f}, {xmax:.6f}, {ymax:.6f}]")
        click.echo(f"\nOutput written to: {result.output_path}")
        click.echo(click.style("Done!", fg="green"))

    except grid.CRSError as e:
        click.echo(click.style(f"\nError: {e}", fg="red"))
        raise SystemExit(1) from e


# Alias for registration
get_grid = get_grid_cmd
