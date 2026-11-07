"""CLI for the staged PlanetScope downloader."""

from pathlib import Path
from typing import Any

import click

from ftw_dataset_tools.commands.planet_common import input_options, invoke, runtime_options


@click.group()
def planet_scenes() -> None:
    """Search, activate, crop and quality-check PlanetScope SR + UDM2."""


@planet_scenes.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@runtime_options
@input_options
@click.option("--start", help="Inclusive acquisition start, overriding row dates.")
@click.option("--end", help="Exclusive acquisition end, overriding row dates.")
@click.option("--target-date", help="Preferred acquisition date for all polygons.")
@click.option("--date-column", default="determination_datetime", show_default=True)
@click.option(
    "--search-days",
    type=click.IntRange(1),
    default=30,
    show_default=True,
    help="Days either side of target when start/end are absent.",
)
@click.option("--max-cloud", type=click.FloatRange(0, 1), default=0.1, show_default=True)
@click.option("--min-coverage", type=click.FloatRange(0, 1, min_open=True), default=0.99)
@click.option("--max-candidates", type=click.IntRange(1), default=10)
def search(input_path: Path, work_dir: Path, **kwargs: Any) -> None:
    """Find scenes for GeoParquet or JSONL polygons."""
    from ftw_dataset_tools.api.planet import scene_workflow as workflow

    invoke(workflow.search, input_path, work_dir, **kwargs)


@planet_scenes.command()
@runtime_options
@click.option("--activation-timeout", type=click.FloatRange(0, min_open=True), default=3600)
def activate(work_dir: Path, **kwargs: Any) -> None:
    """Prewarm SR and UDM2 once per scene; optional before extract."""
    from ftw_dataset_tools.api.planet import scene_workflow as workflow

    invoke(workflow.activate, work_dir, **kwargs)


@planet_scenes.command()
@runtime_options
@click.option("--activation-timeout", type=click.FloatRange(0, min_open=True), default=3600)
def extract(work_dir: Path, **kwargs: Any) -> None:
    """Range-read scene groups; retry missing or corrupt SR/UDM2 independently."""
    from ftw_dataset_tools.api.planet import scene_workflow as workflow

    invoke(workflow.extract, work_dir, **kwargs)


@planet_scenes.command()
@click.option("--work-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--workers", type=click.IntRange(1, 32), default=4)
@click.option("--min-clear", type=click.FloatRange(0, 1), default=0.95)
def quality(work_dir: Path, **kwargs: Any) -> None:
    """Report current AOI-level UDM2 quality in quality.jsonl (one process)."""
    from ftw_dataset_tools.api.planet import scene_workflow as workflow

    invoke(workflow.report_quality, work_dir, **kwargs)


@planet_scenes.command("select")
@click.argument("identifier")
@click.option("--work-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--rank", type=click.IntRange(0), required=True)
def select_candidate(identifier: str, work_dir: Path, rank: int) -> None:
    """Select a zero-based alternate rank; rerun extract and quality afterward."""
    from ftw_dataset_tools.api.planet import scene_workflow as workflow

    invoke(workflow.select_alternate, work_dir, identifier, rank)
