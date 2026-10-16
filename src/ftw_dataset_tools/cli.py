"""Main CLI entry point for FTW Dataset Tools."""

import importlib

import click

from ftw_dataset_tools import __version__
from ftw_dataset_tools.commands.cli_logging import configure_cli_logging

# Command modules are imported on first use to keep startup fast.
COMMAND_MODULES = (
    "add_land_cover",
    "convert_previews",
    "create_boundaries",
    "create_chips",
    "create_dataset",
    "create_dataset_summary",
    "create_ftw_grid",
    "create_masks",
    "create_splits",
    "create_subset",
    "download_images",
    "export_coco",
    "get_grid",
    "inspect_fields",
    "run",
    "select_images",
)
_COMMANDS = {module.replace("_", "-"): module for module in COMMAND_MODULES}

REMOVED_COMMANDS = {
    "add-field-stats": (
        "use 'ftwd create-chips FIELDS_FILE --grid-file GRID_FILE' instead "
        "(add --min-chip-area 0 to keep truncated chips)"
    ),
}


class FtwdGroup(click.Group):
    """Click group that loads commands on first use and points removed ones at their
    replacement."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted({*super().list_commands(ctx), *_COMMANDS})

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        module = _COMMANDS.get(cmd_name)
        if module is None:
            return super().get_command(ctx, cmd_name)
        return getattr(importlib.import_module(f"ftw_dataset_tools.commands.{module}"), module)

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        if args and args[0] in REMOVED_COMMANDS:
            ctx.fail(f"'{args[0]}' was removed; {REMOVED_COMMANDS[args[0]]}.")
        return super().resolve_command(ctx, args)


@click.group(cls=FtwdGroup)
@click.version_option(version=__version__, prog_name="ftwd")
@click.option("-v", "--verbose", is_flag=True, help="Show debug output.")
def cli(verbose: bool) -> None:
    """FTW Dataset Tools - CLI for creating Fields of the World benchmark dataset.

    This tool provides commands for:

    \b
    - Creating complete training datasets from field boundaries (create-dataset)
    - Creating FTW grids
    - Creating chip definitions with field coverage statistics
    - Creating boundary lines and raster masks
    """
    configure_cli_logging(verbose)


if __name__ == "__main__":
    cli()
