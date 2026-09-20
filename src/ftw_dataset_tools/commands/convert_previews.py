"""CLI command for converting a catalog's JPEG chip previews to WebP."""

from __future__ import annotations

from pathlib import Path

import click

from ftw_dataset_tools.api.imagery.parallel import DEFAULT_WORKERS, MAX_WORKERS


@click.command("convert-previews")
@click.argument("catalog_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--dry-run",
    is_flag=True,
    help="Report what would be converted without writing or deleting anything.",
)
@click.option(
    "--workers",
    type=click.IntRange(1, MAX_WORKERS),
    default=DEFAULT_WORKERS,
    show_default=True,
    help="Number of chips to render concurrently.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="List the chips that were skipped or failed.",
)
def convert_previews(catalog_dir: str, dry_run: bool, workers: int, verbose: bool) -> None:
    """Convert a catalog's JPEG chip previews to WebP.

    Chip previews are written as WebP, which is 25-35% smaller than JPEG at
    equivalent quality. A catalog built before that switch still has .jpg previews
    on disk and .jpg hrefs in its items; this converts it in place.

    Each preview is re-rendered from the imagery it was made from -- the chip's
    local clipped GeoTIFF when there is one, otherwise its remote scene -- so the
    result is compressed once rather than re-encoded from an already-lossy JPEG.
    The chip item and its season children are repointed at the WebP, and only then
    is the superseded .jpg removed.

    Chips that already have only WebP previews are skipped, so the command is safe
    to re-run and resumes cleanly after an interruption.

    \b
    CATALOG_DIR: Path to the collection directory (holding collection.json)

    \b
    Examples:
        ftwd convert-previews dataset/
        ftwd convert-previews dataset/ --dry-run
    """
    from ftw_dataset_tools.api.imagery.preview_conversion import (
        conversion_summary_line,
        convert_previews_for_catalog,
    )

    result = convert_previews_for_catalog(
        Path(catalog_dir),
        dry_run=dry_run,
        workers=workers,
    )

    if dry_run:
        click.echo(
            f"Would convert {result.chips_converted} chips "
            f"({result.previews_written} previews, {result.legacy_removed} JPEGs removed)."
        )
    else:
        click.echo(conversion_summary_line(result))

    if verbose:
        for entry in result.skipped_details:
            click.echo(f"  skipped {entry['chip']}: {entry['reason']}")
        for entry in result.failed_details:
            click.echo(f"  failed {entry['chip']}: {entry['error']}")

    if result.failed:
        raise click.ClickException(f"{result.failed} chips failed to convert")
