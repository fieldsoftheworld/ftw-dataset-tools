"""Main CLI entry point for FTW Dataset Tools."""

import click

from ftw_dataset_tools import __version__
from ftw_dataset_tools.commands.add_field_stats import add_field_stats
from ftw_dataset_tools.commands.create_boundaries import create_boundaries
from ftw_dataset_tools.commands.create_chips import create_chips
from ftw_dataset_tools.commands.create_dataset import create_dataset
from ftw_dataset_tools.commands.create_ftw_grid import create_ftw_grid
from ftw_dataset_tools.commands.create_masks import create_masks
from ftw_dataset_tools.commands.get_grid import get_grid
from ftw_dataset_tools.commands.reproject import reproject


@click.group()
@click.version_option(version=__version__, prog_name="ftwd")
def cli() -> None:
    """FTW Dataset Tools - CLI for creating Fields of the World benchmark dataset.

    This tool provides commands for:

    \b
    - Creating complete training datasets from field boundaries (create-dataset)
    - Creating FTW grids
    - Creating chip definitions with field coverage statistics
    - Creating boundary lines and raster masks
    - Reprojecting GeoParquet files
    """


# Register commands
cli.add_command(add_field_stats)
cli.add_command(create_boundaries)
cli.add_command(create_chips)
cli.add_command(create_dataset)
cli.add_command(create_ftw_grid)
cli.add_command(create_masks)
cli.add_command(get_grid)
cli.add_command(reproject)


if __name__ == "__main__":
    cli()
