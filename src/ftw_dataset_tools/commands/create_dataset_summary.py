"""CLI command for creating dataset summary reports."""

from __future__ import annotations

from pathlib import Path

import click

from ftw_dataset_tools.api import dataset_summary


@click.command("create-dataset-summary")
@click.argument("dataset_dir", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(),
    default=None,
    help="Output path for markdown summary (default: dataset_dir/summary.md).",
)
@click.option(
    "--num-examples",
    type=int,
    default=10,
    show_default=True,
    help="Number of example chips to include in the report.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output.",
)
def create_dataset_summary_cmd(
    dataset_dir: str,
    output_path: str | None,
    num_examples: int,
    verbose: bool,
) -> None:
    """Create a summary report for a dataset.

    Generates a markdown report with statistics, visualizations, and example
    chips from a completed dataset.

    The dataset directory should contain a *-chips/ subdirectory with STAC items
    and a chips_*.parquet file with split information.

    The report includes:
    - Total chips and split counts (train/val/test)
    - Geographic distribution map of splits
    - Temporal distributions of planting and harvest dates
    - Cloud cover distributions
    - Example chip visualizations

    \b
    DATASET_DIR: Path to the dataset directory (containing *-chips/ subdirectory)

    \b
    Examples:
        ftwd create-dataset-summary ~/data/my-dataset
        ftwd create-dataset-summary ~/data/my-dataset -o report.md
        ftwd create-dataset-summary ~/data/my-dataset --num-examples 20 -v
    """
    dataset_dir_path = Path(dataset_dir)

    def on_progress(msg: str) -> None:
        if verbose:
            click.echo(msg)

    try:
        click.echo(f"Creating summary for: {dataset_dir_path.name}")

        result = dataset_summary.create_dataset_summary(
            dataset_dir=dataset_dir_path,
            output_path=output_path,
            num_examples=num_examples,
            on_progress=on_progress,
        )

        # Print summary
        click.echo("\nDataset Summary:")
        click.echo(f"  Total chips: {result.total_chips:,}")
        click.echo(
            f"  Train: {result.train_chips:,} "
            f"({result.train_chips / result.total_chips * 100:.1f}%)"
        )
        click.echo(
            f"  Validation: {result.val_chips:,} "
            f"({result.val_chips / result.total_chips * 100:.1f}%)"
        )
        click.echo(
            f"  Test: {result.test_chips:,} ({result.test_chips / result.total_chips * 100:.1f}%)"
        )

        if result.planting_dates:
            click.echo(f"\n  Planting images: {len(result.planting_dates):,}")
            click.echo(
                f"    Date range: {min(result.planting_dates).date()} to {max(result.planting_dates).date()}"
            )

        if result.harvest_dates:
            click.echo(f"\n  Harvest images: {len(result.harvest_dates):,}")
            click.echo(
                f"    Date range: {min(result.harvest_dates).date()} to {max(result.harvest_dates).date()}"
            )

        click.echo(f"\nReport written to: {result.output_path}")
        click.echo(click.style("Done!", fg="green"))

    except FileNotFoundError as e:
        click.echo(click.style(f"\nError: {e}", fg="red"))
        raise SystemExit(1) from e
    except ValueError as e:
        click.echo(click.style(f"\nError: {e}", fg="red"))
        raise SystemExit(1) from e
    except Exception as e:
        click.echo(click.style(f"\nUnexpected error: {e}", fg="red"))
        if verbose:
            raise
        raise SystemExit(1) from e


# Alias for registration
create_dataset_summary = create_dataset_summary_cmd
