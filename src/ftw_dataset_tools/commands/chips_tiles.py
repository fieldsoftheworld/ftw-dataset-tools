"""CLI command for tiling the root chips table into PMTiles."""

from pathlib import Path

import click

from ftw_dataset_tools.api.chips_tiles import DEFAULT_SERVER


@click.command("chips-tiles")
@click.option("--table", help="The chips table to tile (path or URL).")
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the PMTiles archive.",
)
@click.option(
    "--thumbnail",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Render a WebP thumbnail here instead of building tiles (needs --style and "
    "--tiles-url, and a running chiitiler).",
)
@click.option(
    "--style",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The MapLibre style to render the thumbnail with.",
)
@click.option("--tiles-url", help="The published chips.pmtiles the thumbnail renders.")
@click.option("--server", default=DEFAULT_SERVER, show_default=True, help="chiitiler's URL.")
def chips_tiles(
    table: str | None,
    out: Path | None,
    thumbnail: Path | None,
    style: Path | None,
    tiles_url: str | None,
    server: str,
) -> None:
    """Tile the root chips table into PMTiles: every chip location, at every zoom.

    Needs tippecanoe and tile-join on PATH.

    \b
    Examples:
        ftwd chips-tiles --table work/chips.parquet --out work/chips.pmtiles
        ftwd chips-tiles --thumbnail catalog/chips/thumbnail.webp \\
            --style catalog/chips/styles/split-admin2.json \\
            --tiles-url https://data.source.coop/ftw/benchmark-data/chips/chips.pmtiles
    """
    from ftw_dataset_tools.api.chips_tiles import ChipsTilesError, build_tiles

    if thumbnail is not None:
        _render(thumbnail, style, tiles_url, server)
        return
    if table is None or out is None:
        raise click.UsageError("give --table and --out (or --thumbnail)")
    try:
        count = build_tiles(table, out)
    except ChipsTilesError as err:
        raise click.ClickException(str(err)) from err
    click.echo(f"Wrote {out}: {count:,} chip locations")


def _render(thumbnail: Path, style: Path | None, tiles_url: str | None, server: str) -> None:
    from ftw_dataset_tools.api.chips_tiles import ChipsTilesError, render_thumbnail

    if style is None or tiles_url is None:
        raise click.UsageError("--thumbnail needs --style and --tiles-url")
    try:
        out = render_thumbnail(thumbnail, style=style, tiles_url=tiles_url, server=server)
    except ChipsTilesError as err:
        raise click.ClickException(str(err)) from err
    click.echo(f"Wrote {out}")
