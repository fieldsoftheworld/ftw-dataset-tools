"""CLI for reusable Planet Basemaps download stages."""

from pathlib import Path
from typing import Any

import click

from ftw_dataset_tools.commands.planet_common import input_options, invoke, runtime_options


@click.group()
def planet_basemaps() -> None:
    """Plan, cache and crop Planet Basemaps into ordered seasonal stacks."""


@planet_basemaps.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@runtime_options
@input_options
@click.option(
    "--mosaic", "mosaics", multiple=True, help="Exact mosaic name; repeat in stack order."
)
def plan(input_path: Path, work_dir: Path, **kwargs: Any) -> None:
    """Find mosaic quads for GeoParquet or JSONL polygons."""
    from ftw_dataset_tools.api.planet import basemap_workflow as workflow

    invoke(workflow.plan, input_path, work_dir, **kwargs)


@planet_basemaps.command()
@runtime_options
def download(work_dir: Path, **kwargs: Any) -> None:
    """Download unique quads into a validated shared cache."""
    from ftw_dataset_tools.api.planet import basemap_workflow as workflow

    invoke(workflow.download, work_dir, **kwargs)


@planet_basemaps.command()
@runtime_options
@click.option(
    "--band",
    "bands",
    type=click.IntRange(1),
    multiple=True,
    default=(1, 2, 3),
    show_default=True,
    help="One-based source band, repeated in output order.",
)
def assemble(work_dir: Path, **kwargs: Any) -> None:
    """Create one native-grid GeoTIFF per AOI, with periods then bands in order."""
    from ftw_dataset_tools.api.planet import basemap_workflow as workflow

    invoke(workflow.assemble, work_dir, **kwargs)
