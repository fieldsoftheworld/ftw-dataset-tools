"""Main CLI entry point for FTW Dataset Tools."""

import click

from ftw_dataset_tools import __version__
from ftw_dataset_tools.commands.add_field_stats import add_field_stats
from ftw_dataset_tools.commands.create_boundaries import create_boundaries
from ftw_dataset_tools.commands.create_chips import create_chips
from ftw_dataset_tools.commands.create_dataset import create_dataset
from ftw_dataset_tools.commands.create_ftw_grid import create_ftw_grid
from ftw_dataset_tools.commands.create_masks import create_masks
from ftw_dataset_tools.commands.create_splits import create_splits
from ftw_dataset_tools.commands.download_images import download_images
from ftw_dataset_tools.commands.download_images_planet import download_images_planet
from ftw_dataset_tools.commands.get_grid import get_grid
from ftw_dataset_tools.commands.select_images import select_images
from ftw_dataset_tools.commands.select_images_planet import select_images_planet


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
    """


# Register commands
cli.add_command(add_field_stats)
cli.add_command(create_boundaries)
cli.add_command(create_chips)
cli.add_command(create_dataset)
cli.add_command(create_ftw_grid)
cli.add_command(create_masks)
cli.add_command(create_splits)
cli.add_command(download_images)
cli.add_command(download_images_planet)
cli.add_command(get_grid)
cli.add_command(select_images)
cli.add_command(select_images_planet)


if __name__ == "__main__":
    cli()
