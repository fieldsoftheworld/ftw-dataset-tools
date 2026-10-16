"""CLI helpers shared by the commands that select imagery in either imagery mode."""

from __future__ import annotations

import click
from click.core import ParameterSource

from ftw_dataset_tools.api.imagery.mosaic_selection import MosaicYearError, check_mosaic_year
from ftw_dataset_tools.api.imagery.settings import MOSAIC_FALLBACK_YEAR
from ftw_dataset_tools.api.imagery.slots import DEFAULT_IMAGERY_MODE, IMAGERY_MODES

imagery_mode_option = click.option(
    "--imagery-mode",
    type=click.Choice(IMAGERY_MODES),
    default=DEFAULT_IMAGERY_MODE,
    show_default=True,
    help="scenes: a planting and a harvest scene per chip, from the crop calendar. "
    "mosaics: Q1-Q4 Sentinel-2 quarterly cloudless mosaics of one year (--year, default "
    f"{MOSAIC_FALLBACK_YEAR}).",
)


def reject_scene_only_options(ctx: click.Context, names: tuple[str, ...]) -> None:
    """Fail if any scene-only option was set explicitly in mosaic mode."""
    for name in names:
        if ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE:
            flag = "--" + name.replace("_", "-")
            raise click.BadParameter(
                "only applies to --imagery-mode scenes.", param_hint=f"'{flag}'"
            )


def resolve_mosaic_year(year: int | None) -> int:
    """``year``, or the fallback year with a warning; fails if that year is unavailable."""
    if year is None:
        year = MOSAIC_FALLBACK_YEAR
        click.echo(
            click.style(f"Warning: no --year given; using {year} for mosaics.", fg="yellow"),
            err=True,
        )
    try:
        check_mosaic_year(year)
    except MosaicYearError as err:
        raise click.BadParameter(str(err), param_hint="'--year'") from err
    return year
