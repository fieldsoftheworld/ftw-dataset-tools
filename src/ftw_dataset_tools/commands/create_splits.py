"""CLI command for assigning train/val/test splits to chips."""

import click

from ftw_dataset_tools.api import splits


@click.command("create-splits")
@click.argument("chips_file", type=click.Path(exists=True))
@click.option(
    "--split-type",
    type=click.Choice(splits.SPLIT_TYPE_CHOICES),
    required=True,
    help=(
        "Split strategy. 'random-uniform': randomly assign individual chips. "
        "'block3x3': group chips into 3x3 spatial blocks and randomly assign blocks."
    ),
)
@click.option(
    "--split-percents",
    nargs=3,
    type=click.IntRange(0, 100),
    default=(80, 10, 10),
    show_default=True,
    help="Train, validation, and test percentages (must sum to 100).",
)
@click.option(
    "--random-seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducibility.",
)
def create_splits(
    chips_file: str,
    split_type: str,
    split_percents: tuple[int, int, int],
    random_seed: int,
) -> None:
    """Assign train/val/test splits to a chips file.

    Adds a 'split' column to the chips parquet file with values 'train', 'val', or 'test'.

    The CHIPS_FILE should be a GeoParquet file with chip definitions (e.g., created by
    create-chips command). It must contain an 'id' column with chip identifiers.

    \b
    Split strategies:
    - random-uniform: Randomly assigns each chip to train/val/test splits
    - block3x3: Groups chips into 3x3 spatial blocks and assigns blocks to splits
                (ensures spatial coherence within each split)

    \b
    Examples:
        # Random split with default 80/10/10 percentages
        ftwd create-splits chips.parquet --split-type random-uniform

        # Block-based split with custom percentages
        ftwd create-splits chips.parquet --split-type block3x3 --split-percents 70 20 10

        # With custom random seed
        ftwd create-splits chips.parquet --split-type random-uniform --random-seed 123
    """
    # Validate at CLI layer for immediate user feedback with proper Click error formatting
    try:
        validated_split_percents = splits.validate_split_percents(split_percents)
    except ValueError as err:
        raise click.BadParameter(str(err), param_hint="split-percents") from err

    try:
        result = splits.assign_splits(
            chips_file=chips_file,
            split_type=split_type,
            split_percents=validated_split_percents,
            random_seed=random_seed,
            on_progress=lambda msg: click.echo(msg),
        )

        click.echo()
        click.echo(click.style("✓ Splits assigned successfully", fg="green", bold=True))
        click.echo()
        click.echo(f"  Chips file: {result.chips_file}")
        click.echo(f"  Split type: {result.split_type}")
        click.echo(
            f"  Split percentages: {result.split_percents[0]}% train, "
            f"{result.split_percents[1]}% val, {result.split_percents[2]}% test"
        )
        click.echo()
        click.echo(f"  Total chips: {result.total_chips:,}")
        click.echo(f"  Train: {result.train_count:,}")
        click.echo(f"  Validation: {result.val_count:,}")
        click.echo(f"  Test: {result.test_count:,}")

    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    except Exception as e:
        raise click.ClickException(f"Error assigning splits: {e}") from e
