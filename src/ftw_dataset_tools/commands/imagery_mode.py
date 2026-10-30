"""CLI helpers shared by the commands that select imagery in one or both imagery modes."""

from __future__ import annotations

import click
from click.core import ParameterSource

from ftw_dataset_tools.api.imagery.mosaic_selection import MosaicYearError, check_mosaic_year
from ftw_dataset_tools.api.imagery.settings import MOSAIC_FALLBACK_YEAR
from ftw_dataset_tools.api.imagery.slots import (
    DEFAULT_IMAGERY_MODE,
    IMAGERY_MODES,
    normalize_imagery_modes,
)

imagery_mode_option = click.option(
    "--imagery-mode",
    "imagery_modes",
    type=click.Choice(IMAGERY_MODES),
    multiple=True,
    default=(DEFAULT_IMAGERY_MODE,),
    show_default=True,
    callback=lambda _ctx, _param, value: normalize_imagery_modes(value),
    help="scenes: a planting and a harvest scene per chip, from the crop calendar. "
    "mosaics: Q1-Q4 Sentinel-2 quarterly cloudless mosaics of one year (--year, default "
    f"{MOSAIC_FALLBACK_YEAR}). Repeat the option to give every chip both; a year with "
    "no mosaics then skips them with a warning and still selects the scenes.",
)


def reject_scene_only_options(ctx: click.Context, names: tuple[str, ...]) -> None:
    """Fail if any scene-only option was set explicitly in mosaic mode."""
    for name in names:
        if ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE:
            flag = "--" + name.replace("_", "-")
            raise click.BadParameter(
                "only applies to --imagery-mode scenes.", param_hint=f"'{flag}'"
            )


def resolve_mosaic_year(year: int | None, imagery_modes: tuple[str, ...] = ("mosaics",)) -> int:
    """``year``, or the fallback year with a warning.

    With mosaics as the only mode an unavailable year fails here, before any work.
    Alongside scenes it is left to the selection, which skips the mosaics with a
    warning and still selects the scenes.
    """
    if year is None:
        year = MOSAIC_FALLBACK_YEAR
        click.echo(
            click.style(f"Warning: no --year given; using {year} for mosaics.", fg="yellow"),
            err=True,
        )
    if imagery_modes != ("mosaics",):
        return year
    try:
        check_mosaic_year(year)
    except MosaicYearError as err:
        raise click.BadParameter(str(err), param_hint="'--year'") from err
    return year
