"""CLI command for exporting instance masks as COCO annotations."""

import click

from ftw_dataset_tools.api.coco import DEFAULT_SPLIT_COLUMN


@click.command("export-coco")
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False),
    default=None,
    help="Where to write the JSON files. Default: <DATASET_DIR>/coco",
)
@click.option(
    "--min-area",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Drop instances smaller than this many pixels.",
)
@click.option(
    "--splits",
    default=None,
    help="The benchmark's root chips table (path or URL) to take each chip's split from, "
    "matched on the collection id. Default: the items' ftw:split.",
)
@click.option(
    "--split-column",
    default=DEFAULT_SPLIT_COLUMN,
    show_default=True,
    help="The chips table column to read the split from.",
)
def export_coco(
    dataset_dir: str, output_dir: str | None, min_area: int, splits: str | None, split_column: str
) -> None:
    """Export instance masks as COCO instance segmentation JSON, one file per split.

    \b
    Examples:
        ftwd export-coco ./austria-dataset
        ftwd export-coco ./austria-dataset -o ./coco --min-area 10
        ftwd export-coco ./at_2024 --splits https://data.source.coop/ftw/benchmark-data/chips.parquet
    """
    from ftw_dataset_tools.api.assets import MaskReadError
    from ftw_dataset_tools.api.coco import SplitsTableError
    from ftw_dataset_tools.api.coco import export_coco as export_coco_impl

    try:
        result = export_coco_impl(
            dataset_dir,
            output_dir=output_dir,
            min_area=min_area,
            splits=splits,
            split_column=split_column,
        )
    except (FileNotFoundError, MaskReadError, SplitsTableError) as err:
        raise click.ClickException(str(err)) from err

    if not result.files:
        raise click.ClickException(f"No chips with instance masks found in {dataset_dir}")
    for split, path in result.files.items():
        click.echo(
            f"{split}: {result.images[split]} images, "
            f"{result.annotations[split]} annotations -> {path}"
        )
    if result.skipped_chips:
        click.echo(
            click.style(
                f"Skipped {len(result.skipped_chips)} chips without an instance mask",
                fg="yellow",
            )
        )
