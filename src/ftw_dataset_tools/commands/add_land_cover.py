"""CLI command for adding per-chip land cover to an existing chips file."""

import click
from pystac_client.exceptions import APIError

from ftw_dataset_tools.api import land_cover


@click.command("add-land-cover")
@click.argument("chips_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--year",
    type=int,
    required=True,
    help=(
        "Dataset year. IO maps cover 2017-2023; other years use the nearest map and "
        "are flagged with landcover_year_exact = false."
    ),
)
@click.option(
    "--id-col",
    "chips_id_col",
    default="id",
    show_default=True,
    help="Chip id column.",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=land_cover.DEFAULT_WORKERS,
    show_default=True,
    help="Concurrent tile reads.",
)
def add_land_cover_cmd(chips_file: str, year: int, chips_id_col: str, workers: int) -> None:
    """Add per-chip land cover from the Impact Observatory annual maps.

    Counts the 10 m IO land-cover pixels inside each chip and writes the class
    shares into CHIPS_FILE, in place. Rasters are read remotely from Microsoft
    Planetary Computer.

    Use this to add land cover to a dataset that is already built. To publish the
    new columns on the STAC items, set stages.chips.land_cover: true in the config
    and run: ftwd run config.yaml --only stac

    \b
    CHIPS_FILE: Chips GeoParquet in EPSG:4326 (e.g., <name>_chips.parquet)

    \b
    Examples:
        ftwd add-land-cover austria_chips.parquet --year 2020
        ftwd add-land-cover spain_chips.parquet --year 2025 --workers 16
    """

    def on_progress(msg: str) -> None:
        if msg.startswith("Warning:"):
            click.echo(click.style(msg, fg="yellow"))
        else:
            click.echo(msg)

    try:
        land_cover.add_land_cover(
            chips_file,
            year=year,
            chips_id_col=chips_id_col,
            workers=workers,
            on_progress=on_progress,
        )
    except (ValueError, OSError, APIError) as err:
        raise click.ClickException(str(err)) from err

    click.echo(click.style("Done!", fg="green"))


# Alias for registration
add_land_cover = add_land_cover_cmd
