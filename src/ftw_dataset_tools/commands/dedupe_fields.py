"""CLI command for making the field ids of a fields file unique."""

from __future__ import annotations

from pathlib import Path

import click

from ftw_dataset_tools.api import field_ids
from ftw_dataset_tools.api.config import ClassFilter, ConfigError


def require_unique_ids(path: str | Path, id_col: str = field_ids.ID_COLUMN) -> None:
    """Fail the command when ``path`` repeats a field id, pointing to dedupe-fields."""
    try:
        field_ids.require_unique_ids(path, id_col)
    except field_ids.DuplicateIdsError as err:
        raise click.ClickException(
            f"{err}. Field ids must be unique; run 'ftwd dedupe-fields {path} OUTPUT' "
            "and use its output instead."
        ) from err
    except field_ids.FieldIdsError as err:
        raise click.ClickException(str(err)) from err


def _same_file(a: str, b: str) -> bool:
    first, second = Path(a), Path(b)
    if second.exists():
        return first.samefile(second)
    return first.resolve() == second.resolve()


@click.command("dedupe-fields")
@click.argument("fields_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(dir_okay=False))
@click.option("--id-col", default=field_ids.ID_COLUMN, show_default=True, help="Field id column.")
@click.option(
    "--class-filter",
    "class_filter_file",
    type=click.Path(exists=True, dir_okay=False),
    help="Class filter YAML: of rows describing one parcel, keep one whose class it includes.",
)
def dedupe_fields_cmd(
    fields_file: str, output_file: str, id_col: str, class_filter_file: str | None
) -> None:
    """Write FIELDS_FILE to OUTPUT_FILE with every field id unique.

    ftwd run and create-dataset do this to the fields they write; use this command
    before create-chips, create-boundaries, create-masks or add-field-embeddings,
    which refuse fields that repeat an id. Rows sharing an id are resolved by the
    first rule that fits them:

    \b
    exact        identical rows (metrics:* within 0.01%): one is kept
    parts        one field split over rows with the same attributes and no
                 overlap: merged into one row
    same_parcel  the same polygon with different attributes: one row is kept
                 and the others' crop names go in other_crop_names
    renumbered   anything else: every row is kept as <id>_1..<id>_k, and the
                 source id goes in source_id

    A file without duplicate ids is copied unchanged. OUTPUT_FILE must be a new
    path: rewriting the fields of a built dataset in place would leave its chips,
    masks and outlines on the old ids (rerun ftwd run from reproject instead).

    \b
    Examples:
        ftwd dedupe-fields cz_2019.parquet cz_2019_unique.parquet
        ftwd dedupe-fields at_2024.parquet at_2024_unique.parquet --class-filter crops.yaml
    """
    if _same_file(fields_file, output_file):
        raise click.BadParameter(
            "must not be FIELDS_FILE: write the unique fields to a new file. For a dataset "
            "built by ftwd run, rerun it from the reproject stage instead, which dedupes "
            "the fields and rebuilds everything made from them.",
            param_hint="OUTPUT_FILE",
        )
    try:
        class_filter = ClassFilter.from_file(class_filter_file) if class_filter_file else None
        result = field_ids.dedupe_fields(
            fields_file, output_file, id_col=id_col, class_filter=class_filter
        )
    except (ConfigError, field_ids.FieldIdsError) as err:
        raise click.ClickException(str(err)) from err
    if result.added_columns:
        click.echo(f"Added column(s): {', '.join(result.added_columns)}")
    click.echo(f"Output written to: {output_file}")


# Alias for registration
dedupe_fields = dedupe_fields_cmd
