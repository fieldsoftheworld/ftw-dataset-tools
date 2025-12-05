"""CLI command for extracting polygon boundaries as lines."""

import click
from tqdm import tqdm

from ftw_dataset_tools.api import boundaries


@click.command("create-boundaries")
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory. If not specified, writes to same directory as input files.",
)
@click.option(
    "--prefix",
    "output_prefix",
    default="boundary_lines_",
    show_default=True,
    help="Prefix for output filenames.",
)
@click.option(
    "--geom-col",
    default=None,
    help="Geometry column name (auto-detected from GeoParquet metadata if not specified).",
)
def create_boundaries_cmd(
    input_path: str,
    output_dir: str | None,
    output_prefix: str,
    geom_col: str | None,
) -> None:
    """Convert polygon geometries to boundary lines using ST_Boundary.

    Takes an input file or directory of parquet files containing polygons
    and creates new parquet files with the polygon boundaries as lines.

    Output files are named: <prefix><original_name>.parquet

    \b
    INPUT_PATH: Parquet file or directory containing parquet files

    \b
    Examples:
        ftwd create-boundaries fields.parquet
        ftwd create-boundaries ./data/
        ftwd create-boundaries fields.parquet -o ./output/
        ftwd create-boundaries fields.parquet --prefix lines_
    """
    click.echo(f"Input: {input_path}")
    if output_dir:
        click.echo(f"Output directory: {output_dir}")

    # Progress callback
    def on_progress(msg: str) -> None:
        if "Warning:" in msg:
            click.echo(click.style(msg, fg="yellow"))
        elif msg.startswith("Error") or "Error:" in msg:
            click.echo(click.style(msg, fg="red"))
        else:
            click.echo(msg)

    try:
        with tqdm(total=100, desc="Processing", unit="%") as pbar:
            pbar.update(10)

            result = boundaries.create_boundaries(
                input_path=input_path,
                output_dir=output_dir,
                output_prefix=output_prefix,
                geom_col=geom_col,
                on_progress=on_progress,
            )

            pbar.update(90)

        # Print summary
        click.echo("\nSummary:")
        click.echo(f"  Files processed: {result.total_processed}")
        click.echo(f"  Files skipped: {result.total_skipped}")
        click.echo(f"  Total features: {result.total_features:,}")

        if result.files_processed:
            click.echo("\nOutput files:")
            for r in result.files_processed:
                click.echo(f"  {r.output_path}")

        if result.files_skipped:
            click.echo("\nSkipped files:")
            for path, reason in result.files_skipped:
                click.echo(f"  {path.name}: {reason}")

        click.echo(click.style("\nDone!", fg="green"))

    except FileNotFoundError as e:
        click.echo(click.style(f"\nError: {e}", fg="red"))
        raise SystemExit(1) from e
    except ValueError as e:
        click.echo(click.style(f"\nError: {e}", fg="red"))
        raise SystemExit(1) from e


# Alias for registration
create_boundaries = create_boundaries_cmd
