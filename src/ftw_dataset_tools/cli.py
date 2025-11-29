"""Main CLI entry point for FTW Dataset Tools."""

import click

from ftw_dataset_tools import __version__
from ftw_dataset_tools.commands.add_field_stats import add_field_stats
from ftw_dataset_tools.commands.get_grid import get_grid
from ftw_dataset_tools.commands.reproject import reproject


@click.group()
@click.version_option(version=__version__, prog_name="ftwd")
def cli() -> None:
    """FTW Dataset Tools - CLI for creating Fields of the World benchmark dataset.

    This tool provides commands for:

    \b
    - Creating FTW grids
    - Subsetting grids based on fiboa field boundaries
    - Adding field coverage statistics to grids
    - Reprojecting GeoParquet files
    - Pulling GeoTIFF images for training data
    """


# Register commands
cli.add_command(add_field_stats)
cli.add_command(get_grid)
cli.add_command(reproject)


if __name__ == "__main__":
    cli()
