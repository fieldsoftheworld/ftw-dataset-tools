"""CLI options and error translation for Planet download stages."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import click


def runtime_options(function: Callable[..., Any]) -> Callable[..., Any]:
    """Common stage options, independent of any cluster's SLURM configuration."""
    options = [
        click.option("--work-dir", type=click.Path(path_type=Path), required=True),
        click.option("--workers", type=click.IntRange(1, 32), default=4, show_default=True),
        click.option("--shard-id", type=click.IntRange(0), default=0, show_default=True),
        click.option("--num-shards", type=click.IntRange(1), default=1, show_default=True),
    ]
    for option in reversed(options):
        function = option(function)
    return function


def invoke(function: Callable[..., int | None], *args: Any, **kwargs: Any) -> None:
    """Translate expected input/API failures without leaking request URLs."""
    from ftw_dataset_tools.api.planet.http import PlanetError

    if "shard_id" in kwargs and kwargs["shard_id"] >= kwargs["num_shards"]:
        raise click.BadParameter("shard_id must be smaller than num_shards")
    try:
        failures = function(*args, **kwargs)
    except (ValueError, KeyError, OSError, PlanetError) as error:
        raise click.ClickException(str(error)) from None
    if failures:
        raise click.ClickException(f"{failures} jobs failed; inspect the stage journal and rerun")
