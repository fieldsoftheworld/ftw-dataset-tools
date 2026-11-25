"""CLI command for building the root chips table of a benchmark catalog."""

from pathlib import Path

import click


def _parse_groups(values: tuple[str, ...]) -> dict[str, str]:
    groups = {}
    for value in values:
        dataset, sep, group = value.partition("=")
        if not sep or not dataset or not group:
            raise click.BadParameter(f"{value!r} is not <id>=<group>", param_hint="--group")
        groups[dataset] = group
    return groups


@click.command("chips-table")
@click.argument("catalog_json", metavar="CATALOG")
@click.option(
    "--data-base-url",
    default=None,
    help="Where the catalog's data files are served, e.g. "
    "https://data.source.coop/ftw/benchmark-data. Relative chips hrefs resolve against it "
    "plus the collection's folder. Default: the catalog's own folder, which only works when "
    "the data files sit next to the catalog; a catalog whose data is served elsewhere (like "
    "the benchmark catalog) needs this.",
)
@click.option(
    "--local",
    "local_dirs",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="A dataset directory written by ftwd run; its chips replace the published ones "
    "for that id (repeatable).",
)
@click.option(
    "--group",
    "groups",
    multiple=True,
    metavar="ID=GROUP",
    help="The group of a --local dataset the catalog does not have yet (repeatable).",
)
@click.option(
    "--previous",
    required=True,
    help="The published table (path or URL) to carry frozen splits from.",
)
@click.option(
    "--drop",
    multiple=True,
    help="A dataset whose frozen splits may be lost: left out entirely, or chips a rebuild "
    "no longer has (repeatable).",
)
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--workers", type=click.IntRange(min=1), default=8, show_default=True)
def chips_table(
    catalog_json: str,
    data_base_url: str | None,
    local_dirs: tuple[Path, ...],
    groups: tuple[str, ...],
    previous: str,
    drop: tuple[str, ...],
    out: Path,
    workers: int,
) -> None:
    """Build the root chips table: every chip of every dataset CATALOG links, with its splits.

    CATALOG is the root catalog.json (path or URL); its child links lead through the
    groups to each dataset's collection.json.

    \b
    Example:
        ftwd chips-table catalog/catalog.json \\
            --data-base-url https://data.source.coop/ftw/benchmark-data \\
            --previous https://data.source.coop/ftw/benchmark-data/chips/chips.parquet \\
            --local staging/pt_2021 --out work/chips.parquet
    """
    from ftw_dataset_tools.api.chips_table import (
        ChipsTableError,
        build_chips_table,
        resolve_inputs,
    )

    try:
        inputs = resolve_inputs(catalog_json, data_base_url, local_dirs, _parse_groups(groups))
        rows = build_chips_table(inputs, out, previous, drop=drop, workers=workers)
    except (ChipsTableError, OSError) as err:
        raise click.ClickException(str(err)) from err
    click.echo(f"Wrote {out}: {rows:,} chips from {len(inputs)} datasets")
