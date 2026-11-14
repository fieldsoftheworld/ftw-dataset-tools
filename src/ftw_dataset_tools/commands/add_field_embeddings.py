"""CLI command for writing per-field embedding statistics for an existing fields file."""

import click

from ftw_dataset_tools.api import field_embeddings
from ftw_dataset_tools.api.source import SourceFetchError


@click.command("add-field-embeddings")
@click.argument("fields_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(dir_okay=False))
@click.option(
    "--year",
    type=click.IntRange(2017, 2025),
    required=True,
    help="Embedding year (both products cover 2017-2025).",
)
@click.option("--id-col", default="id", show_default=True, help="Field id column.")
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=field_embeddings.DEFAULT_WORKERS,
    show_default=True,
    help="Blocks of fields read at once.",
)
def add_field_embeddings_cmd(
    fields_file: str, output_file: str, year: int, id_col: str, workers: int
) -> None:
    """Write per-field AlphaEarth Foundations and OlmoEarth v1.3 embedding statistics.

    For every field in FIELDS_FILE, the mean and standard deviation of each of the 64
    embedding dimensions over the field's 10 m pixels, from both products, are written
    to OUTPUT_FILE (Parquet, one row per field, keyed by the id column). Embeddings are
    read remotely from Source Cooperative.

    To publish the file with a dataset built by ftwd run, set
    stages.chips.field_embeddings: true instead; the chips stage writes it as
    <name>_field_embeddings.parquet and the stac stage adds it to the collection.

    \b
    FIELDS_FILE: Field polygons (GeoParquet, any CRS)
    OUTPUT_FILE: Parquet to write

    \b
    Examples:
        ftwd add-field-embeddings dk_2022_fields.parquet dk_2022_field_embeddings.parquet --year 2022
    """
    with click.progressbar(length=1, label="Fields") as bar:

        def progress(done: int, total: int) -> None:
            bar.length = max(total, 1)
            bar.update(done - bar.pos)

        try:
            field_embeddings.add_field_embeddings(
                fields_file,
                output_file,
                year=year,
                id_col=id_col,
                workers=workers,
                on_progress=progress,
            )
        except (ValueError, OSError, SourceFetchError) as err:
            raise click.ClickException(str(err)) from err

    click.echo(click.style("Done!", fg="green"))


# Alias for registration
add_field_embeddings = add_field_embeddings_cmd
