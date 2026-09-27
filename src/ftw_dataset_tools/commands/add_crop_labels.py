"""CLI command for adding HCAT3 crop labels to a fields file."""

from pathlib import Path

import click

from ftw_dataset_tools.api import crop_labels


@click.command("add-crop-labels")
@click.argument("fields_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output",
    "output_file",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output file. Defaults to <fields_stem>_hcat.parquet beside the input.",
)
@click.option(
    "--mapping",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="CSV mapping crop codes or names to HCAT3 (original_code/original_name, HCAT3_code).",
)
@click.option(
    "--column",
    default=None,
    help="Source crop column. Defaults to crop:code, then crop:name.",
)
@click.option("--strict", is_flag=True, default=False, help="Fail when any crop value is unmapped.")
@click.option(
    "--template-only",
    is_flag=True,
    default=False,
    help="Only write the mapping template (every crop value, prefilled where known).",
)
def add_crop_labels(
    fields_file: str,
    output_file: str | None,
    mapping: str | None,
    column: str | None,
    strict: bool,
    template_only: bool,
) -> None:
    """Add hcat:code, hcat:name and hcat:name_en to a fields file.

    Crop values are mapped through --mapping, or the dataset's crop:code_list when
    it is a bundled fiboa or EuroCrops list. Unmapped values are written to a
    template CSV; fill in its HCAT3 columns and pass it back as --mapping.

    \b
    Examples:
        ftwd add-crop-labels fields.parquet
        ftwd add-crop-labels fields.parquet --template-only
        ftwd add-crop-labels fields.parquet --mapping crops.csv -o labeled.parquet
    """
    if output_file is None:
        source = Path(fields_file)
        output_file = str(source.with_name(f"{source.stem}_hcat.parquet"))
    try:
        result = crop_labels.add_crop_labels(
            fields_file,
            output_file,
            mapping=mapping,
            column=column,
            strict=strict,
            template_only=template_only,
            on_progress=click.echo,
        )
    except crop_labels.CropLabelError as err:
        raise click.ClickException(str(err)) from err
    click.echo(crop_labels.crop_labels_summary(result))
    if not template_only:
        click.echo(f"Output: {output_file}")
