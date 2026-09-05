"""CLI command for the config-driven dataset workflow (``ftwd run``)."""

from __future__ import annotations

import sys

import click

from ftw_dataset_tools.api import config as config_module
from ftw_dataset_tools.api import crop_stats, pipeline, source


def _on_progress(msg: str) -> None:
    """Color-coded general progress messages (mirrors create-dataset)."""
    if msg.startswith("Warning:"):
        click.echo(click.style(msg, fg="yellow"))
    elif "Error" in msg:
        click.echo(click.style(msg, fg="red"))
    elif "Reprojecting" in msg or "CRS" in msg:
        click.echo(click.style(msg, fg="cyan"))
    elif "complete" in msg.lower():
        click.echo(click.style(msg, fg="green"))
    else:
        click.echo(msg)


def _on_mask_progress(current: int, total: int) -> None:
    percent = int(100 * current / total) if total > 0 else 0
    bar_width = 40
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    sys.stdout.write(f"\r  Creating masks: |{bar}| {current}/{total} ({percent}%)")
    sys.stdout.flush()


def _on_mask_start(total_grids: int, filtered_grids: int) -> None:
    skipped = total_grids - filtered_grids
    if skipped > 0:
        click.echo(f"  Processing {filtered_grids:,} grids (skipping {skipped:,} below threshold)")
    else:
        click.echo(f"  Processing {filtered_grids:,} grids")


@click.command("run")
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--only",
    default=None,
    help="Run a single stage only (e.g. 'masks'). Forces the stage even if disabled.",
)
@click.option(
    "--from",
    "from_stage",
    default=None,
    help="Run from this stage through the end of the pipeline.",
)
@click.option(
    "--through",
    "through_stage",
    default=None,
    help="Run from the first stage through this stage (inclusive).",
)
@click.option(
    "--list-stages",
    is_flag=True,
    default=False,
    help="List the pipeline stages in order and exit.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show the resolved config and the stages that would run, without executing.",
)
def run_cmd(
    config_file: str,
    only: str | None,
    from_stage: str | None,
    through_stage: str | None,
    list_stages: bool,
    dry_run: bool,
) -> None:
    """Run the dataset creation pipeline from a YAML config file.

    The config captures every setting for the build, and a fully-resolved copy is
    written to the output directory (``ftwd-config.resolved.yaml``) and embedded in
    the STAC catalog for provenance.

    \b
    CONFIG_FILE: Path to a YAML config file (see configs/examples/config.yaml).

    \b
    Examples:
        ftwd run config.yaml
        ftwd run config.yaml --only masks
        ftwd run config.yaml --from select_images
        ftwd run config.yaml --dry-run
    """
    if list_stages:
        click.echo("Pipeline stages (in order):")
        for stage in pipeline.STAGE_ORDER:
            click.echo(f"  - {stage}")
        return

    if only is not None and (from_stage is not None or through_stage is not None):
        raise click.BadParameter("--only cannot be combined with --from/--through.")

    try:
        config = config_module.load_config(config_file)
    except config_module.ConfigError as err:
        raise click.ClickException(str(err)) from err

    try:
        stages = pipeline.resolve_stages(
            only=only,
            from_stage=from_stage,
            through_stage=through_stage,
            config=config,
        )
    except ValueError as err:
        raise click.BadParameter(str(err)) from err

    if not stages:
        raise click.ClickException(
            "No stages selected to run. Check enabled flags or stage selectors."
        )

    provenance = config.provenance_dict()

    if dry_run:
        import yaml

        click.echo(f"Source: {config.fields_file}")
        if config.source_via:
            click.echo(f"  via: {config.source_via}")
        click.echo(click.style("Config (resolved):", fg="cyan", bold=True))
        click.echo(yaml.safe_dump(provenance["config"], sort_keys=False))
        click.echo(click.style("Stages that would run:", fg="cyan", bold=True))
        for stage in stages:
            click.echo(f"  - {stage}")
        return

    click.echo(click.style("Running dataset pipeline from config", fg="cyan", bold=True))
    click.echo(f"Config: {config_file}")
    click.echo(f"Stages: {', '.join(stages)}")

    try:
        ctx = pipeline.build_context(
            config,
            on_progress=_on_progress,
            on_mask_progress=_on_mask_progress,
            on_mask_start=_on_mask_start,
            provenance=provenance,
        )
        pipeline.run_pipeline(ctx, stages)
        sys.stdout.write("\n")
        sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        click.echo(click.style("Interrupted by user.", fg="yellow"))
        raise SystemExit(130) from None
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        pipeline.StageInputError,
        source.SourceFetchError,
    ) as err:
        click.echo(click.style(f"\nError: {err}", fg="red"))
        raise SystemExit(1) from err

    _print_summary(ctx)


def _print_summary(ctx: pipeline.PipelineContext) -> None:
    """Print a short summary of what the run produced."""
    click.echo("")
    click.echo(click.style("Pipeline finished!", fg="green", bold=True))
    click.echo(f"  Output: {ctx.output_dir}")
    click.echo(f"  Provenance: {ctx.output_dir / 'ftwd-config.resolved.yaml'}")

    if ctx.chips_result:
        click.echo(f"  Grid cells with coverage: {ctx.chips_result.cells_with_coverage:,}")
    if ctx.splits_result:
        click.echo(
            f"  Splits: {ctx.splits_result.train_count} train, "
            f"{ctx.splits_result.val_count} val, {ctx.splits_result.test_count} test"
        )
    if ctx.masks_results:
        total = sum(r.total_created for r in ctx.masks_results.values())
        click.echo(f"  Masks created: {total:,}")
    if ctx.chips_result:
        click.echo(f"  {crop_stats.crop_stats_summary(ctx.crop_stats_result)}")
    if ctx.stac_result:
        click.echo(f"  STAC items: {ctx.stac_result.total_items:,}")
    if ctx.selection_result is not None:
        click.echo(f"  Imagery selected: {ctx.selection_result.successful}")
    if ctx.download_result is not None:
        click.echo(f"  Imagery downloaded: {ctx.download_result.successful}")
    if ctx.docs_result is not None:
        click.echo(pipeline.docs_summary_line(ctx.docs_result, ctx.config.stages.docs.pmtiles))


# Alias for registration
run = run_cmd
