"""Main CLI entry point for FTW Dataset Tools."""

import click

from ftw_dataset_tools import __version__
from ftw_dataset_tools.commands.create_chips import create_chips
from ftw_dataset_tools.commands.create_ftw_grid import create_ftw_grid
from ftw_dataset_tools.commands.get_grid import get_grid
from ftw_dataset_tools.commands.reproject import reproject


@click.group()
@click.version_option(version=__version__, prog_name="ftwd")
def cli() -> None:
    """FTW Dataset Tools - CLI for creating Fields of the World benchmark dataset.

    This tool provides commands for:

    \b
    - Creating FTW grids
    - Creating chip definitions with field coverage statistics
    - Subsetting grids based on fiboa field boundaries
    - Reprojecting GeoParquet files
    - Pulling GeoTIFF images for training data
    """


# Register commands
cli.add_command(create_chips)
cli.add_command(create_ftw_grid)
cli.add_command(get_grid)
cli.add_command(reproject)


if __name__ == "__main__":
    cli()
