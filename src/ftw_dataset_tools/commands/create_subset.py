"""CLI command for keeping a nested, hash-based scale of a chips file."""

import click

from ftw_dataset_tools.api import field_stats, scale
from ftw_dataset_tools.commands.options import KM_SIZE, KM_SIZE_HELP


@click.command("create-subset")
@click.argument("chips_file", type=click.Path(exists=True))
@click.option(
    "--percent",
    type=click.FloatRange(0, 100),
    required=True,
    help="Percent of 3x3 chip blocks to keep.",
)
@click.option(
    "--min-blocks",
    type=click.IntRange(min=0),
    default=scale.DEFAULT_MIN_BLOCKS_PER_SQUARE,
    show_default=True,
    help="Blocks always kept per MGRS 100 km square.",
)
@click.option(
    "--km-size",
    type=KM_SIZE,
    default=field_stats.DEFAULT_CHIP_KM_SIZE,
    show_default=True,
    help=KM_SIZE_HELP,
)
def create_subset(chips_file: str, percent: float, min_blocks: int, km_size: float) -> None:
    """Keep a reproducible scale of a chips file, in place.

    Each 3x3 block of chips gets a stable hash score; blocks scoring below PERCENT
    are kept, so a smaller scale is always a subset of a larger one.

    \b
    Example:
        ftwd create-subset chips.parquet --percent 10
    """
    try:
        result = scale.apply_scale(chips_file, percent, min_blocks, km_size)
    except ValueError as err:
        raise click.ClickException(str(err)) from err
    click.echo(f"Kept {result.kept_chips:,} of {result.total_chips:,} chips in {result.chips_file}")
