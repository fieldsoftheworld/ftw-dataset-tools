"""CLI helpers shared by the commands that select imagery, in either source and mode."""

from __future__ import annotations

import click
from click.core import ParameterSource

from ftw_dataset_tools.api.imagery.mosaic_selection import MosaicYearError, check_mosaic_year
from ftw_dataset_tools.api.imagery.settings import MOSAIC_FALLBACK_YEAR
from ftw_dataset_tools.api.imagery.slots import (
    DEFAULT_IMAGERY_MODE,
    DEFAULT_SOURCE,
    IMAGERY_MODES,
    IMAGERY_SOURCES,
)

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


def planet_options(source_flag: str = "--source"):
    """The imagery source option (named ``source_flag``) plus the PlanetScope order options."""

    def decorate(command):
        return _add_planet_options(command, source_flag)

    return decorate


def _add_planet_options(command, source_flag: str):
    command = click.option(
        "--planet-timeout",
        type=click.FloatRange(min=0),
        default=120.0,
        show_default=True,
        help="planetscope: minutes to wait for Planet orders before leaving them for a rerun.",
    )(command)
    command = click.option(
        "--planet-wait/--no-planet-wait",
        default=True,
        show_default=True,
        help="planetscope: wait for Planet orders to finish. Without waiting, chips whose "
        "orders are still processing are skipped and picked up by a rerun.",
    )(command)
    return click.option(
        source_flag,
        "source",
        type=click.Choice(IMAGERY_SOURCES),
        default=DEFAULT_SOURCE,
        show_default=True,
        help="Imagery source. planetscope: PlanetScope 4-band surface reflectance scenes, "
        "ordered and cached under <dataset>/.planet (needs PL_API_KEY; scenes mode only).",
    )(command)


def check_source_options(ctx: click.Context, source: str, imagery_mode: str) -> None:
    """Fail on options that do not apply to the chosen source."""
    if source != "planetscope":
        for name in ("planet_wait", "planet_timeout"):
            if ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE:
                flag = "--" + name.replace("_", "-")
                raise click.BadParameter(
                    "only applies to the planetscope source.", param_hint=f"'{flag}'"
                )
        return
    if imagery_mode == "mosaics":
        raise click.BadParameter(
            "PlanetScope supports --imagery-mode scenes only for now.",
            param_hint="'--imagery-mode'",
        )
    if ctx.get_parameter_source("search_backend") == ParameterSource.COMMANDLINE:
        raise click.BadParameter(
            "only applies to the sentinel-2 source.", param_hint="'--search-backend'"
        )
