"""CLI command for reprojecting GeoParquet files."""

import click

from ftw_dataset_tools.api import geo


@click.command("reproject")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    "output_file",
    type=click.Path(),
    default=None,
    help="Output file path. If not specified, creates <input>_<crs>.parquet.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite input file in place instead of creating a new file.",
)
@click.option(
    "--target-crs",
    default="EPSG:4326",
    show_default=True,
    help="Target CRS in format 'EPSG:XXXX'.",
)
def reproject_cmd(
    input_file: str,
    output_file: str | None,
    overwrite: bool,
    target_crs: str,
) -> None:
    """Reproject a GeoParquet file to a different CRS.

    \b
    INPUT_FILE: GeoParquet file to reproject

    \b
    Examples:
        ftwd reproject input.parquet
        ftwd reproject input.parquet -o output.parquet
        ftwd reproject input.parquet --overwrite
        ftwd reproject input.parquet --target-crs EPSG:32610
    """
    click.echo(f"Input file: {input_file}")
    click.echo(f"Target CRS: {target_crs}")

    # Determine output file
    effective_output = output_file
    if overwrite and output_file is None:
        effective_output = input_file

    def on_progress(msg: str) -> None:
        if msg.startswith("Warning:"):
            click.echo(click.style(msg, fg="yellow"))
        else:
            click.echo(msg)

    try:
        result = geo.reproject(
            input_file=input_file,
            output_file=effective_output,
            target_crs=target_crs,
            on_progress=on_progress,
        )

        click.echo("\nSummary:")
        click.echo(f"  Source CRS: {result.source_crs}")
        click.echo(f"  Target CRS: {result.target_crs}")
        click.echo(f"  Features reprojected: {result.feature_count:,}")
        click.echo(f"\nOutput written to: {result.output_path}")
        click.echo(click.style("Done!", fg="green"))

    except ValueError as e:
        raise click.ClickException(str(e)) from e


# Alias for registration
reproject = reproject_cmd
