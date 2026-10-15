"""Main CLI entry point for FTW Dataset Tools."""

import click

from ftw_dataset_tools import __version__
from ftw_dataset_tools.commands.convert_previews import convert_previews
from ftw_dataset_tools.commands.create_boundaries import create_boundaries
from ftw_dataset_tools.commands.create_chips import create_chips
from ftw_dataset_tools.commands.create_dataset import create_dataset
from ftw_dataset_tools.commands.create_dataset_summary import create_dataset_summary
from ftw_dataset_tools.commands.create_ftw_grid import create_ftw_grid
from ftw_dataset_tools.commands.create_masks import create_masks
from ftw_dataset_tools.commands.create_splits import create_splits
from ftw_dataset_tools.commands.download_images import download_images
from ftw_dataset_tools.commands.get_grid import get_grid
from ftw_dataset_tools.commands.inspect_fields import inspect_fields
from ftw_dataset_tools.commands.run import run
from ftw_dataset_tools.commands.select_images import select_images

REMOVED_COMMANDS = {
    "add-field-stats": (
        "use 'ftwd create-chips FIELDS_FILE --grid-file GRID_FILE' instead "
        "(add --min-chip-area 0 to keep truncated chips)"
    ),
}


class FtwdGroup(click.Group):
    """Click group that points removed commands at their replacement."""

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        if args and args[0] in REMOVED_COMMANDS:
            ctx.fail(f"'{args[0]}' was removed; {REMOVED_COMMANDS[args[0]]}.")
        return super().resolve_command(ctx, args)


@click.group(cls=FtwdGroup)
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
cli.add_command(convert_previews)
cli.add_command(create_boundaries)
cli.add_command(create_chips)
cli.add_command(create_dataset)
cli.add_command(create_dataset_summary)
cli.add_command(create_ftw_grid)
cli.add_command(create_masks)
cli.add_command(create_splits)
cli.add_command(download_images)
cli.add_command(get_grid)
cli.add_command(inspect_fields)
cli.add_command(run)
cli.add_command(select_images)


if __name__ == "__main__":
    cli()
