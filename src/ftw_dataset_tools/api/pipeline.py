"""Stage-based orchestration for FTW dataset creation.

This module is the single source of truth for the dataset build pipeline. Each
stage is a small function operating on a :class:`PipelineContext`; the flag-driven
``create-dataset`` command and the config-driven ``ftwd run`` command both build a
context and execute the same stages.

Intermediate outputs live in the output directory under a fixed naming
convention, so any stage can be re-run on its own as long as its inputs exist:

    {output_dir}/{name}_fields.parquet          (reproject)
    {output_dir}/{name}_chips.parquet           (chips, scale, splits)
    {output_dir}/{name}_boundary_lines.parquet  (boundaries)
    {output_dir}/collection.json                 (stac)
    {output_dir}/chips/<mgrs100k>/<item_id>/     (masks, stac, imagery)
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import duckdb

from ftw_dataset_tools.api import (
    admin_names,
    boundaries,
    class_filter,
    crop_stats,
    docs,
    field_embeddings,
    field_stats,
    label_extent,
    land_cover,
    masks,
    scale,
    splits,
    stac,
    styles,
    tiles,
)
from ftw_dataset_tools.api import config as config_module
from ftw_dataset_tools.api.config import DOWNLOAD_MODE_PREVIEW
from ftw_dataset_tools.api.geo import (
    CRSInfo,
    detect_crs,
    detect_geometry_column,
    ensure_spatial_loaded,
    reproject,
    sql_path,
)
from ftw_dataset_tools.api.imagery import download_imagery_for_catalog
from ftw_dataset_tools.api.imagery.preview_workflow import preview_imagery_for_catalog
from ftw_dataset_tools.api.imagery.selection_workflow import (
    require_mosaic_years,
    select_imagery_modes_for_catalog,
)
from ftw_dataset_tools.api.imagery.slots import MOSAIC_SLOTS
from ftw_dataset_tools.api.logging_config import get_logger, success
from ftw_dataset_tools.api.masks import MaskType
from ftw_dataset_tools.api.source import (
    describe_local_source,
    fetch_source,
    installed_git_commit,
    is_url,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ftw_dataset_tools.api.config import DatasetConfig
    from ftw_dataset_tools.api.source import SourceRecord

logger = get_logger(__name__)

# Ordered list of all pipeline stages.
STAGE_ORDER = [
    "reproject",
    "filter",
    "chips",
    "scale",
    "splits",
    "boundaries",
    "masks",
    "stac",
    "select_images",
    "download_images",
    "docs",
]

# The stages that reach out to an imagery API. ``ftwd create-dataset`` does that
# work itself rather than through these stages, so it hooks it in at the position
# they occupy here (see ``run_pipeline``'s ``before_stage``) instead of running the
# imagery work after a pipeline that has already documented the collection.
IMAGERY_STAGES = ("select_images", "download_images")

# Stages that require a resolvable calendar year (for chip naming, temporal
# extent, or imagery search).
_YEAR_STAGES = {"masks", "stac", *IMAGERY_STAGES}

# Stages that read the original source input. Every other stage works from files
# already in the output directory, so a run that selects none of these must not
# download or re-hash the source.
_SOURCE_STAGES = {"reproject"}


class StageInputError(ValueError):
    """Raised when a stage is run without its required input files present."""


@dataclass
class DocsStageResult:
    """What the docs stage produced: tile archives, styles and documents."""

    tiles: dict[str, Path] = field(default_factory=dict)
    styles: list[styles.StyleResult] = field(default_factory=list)
    docs: list[Path] = field(default_factory=list)
    tippecanoe_used: bool = False


def imagery_failures(ctx: PipelineContext) -> str | None:
    """Why the run's imagery is incomplete, or None.

    Selection and download carry on past a chip that fails (a read error, a
    throttled request) so one bad chip does not lose the rest; the run still
    finishes its later stages. But a run with failed chips is not complete, and
    a rerun retries them, so the caller should exit non-zero rather than let a
    batch record the dataset as done.
    """
    counts = {
        "chip(s) failed selection": ctx.selection_result,
        "image(s) failed download": ctx.download_result,
        "preview(s) failed": ctx.preview_result,
    }
    failed = [
        f"{result.failed} {what}"
        for what, result in counts.items()
        if result is not None and result.failed
    ]
    return "; ".join(failed) or None


def docs_summary_line(result: DocsStageResult, pmtiles: str | bool) -> str:
    """One line naming the documents written and counting the tiles and styles.

    Shared by ``ftwd run`` and ``ftwd create-dataset``, which both run the docs
    stage and should report it identically.
    """
    parts = []
    if result.docs:
        parts.append(", ".join(path.name for path in result.docs))
    parts.append(f"{len(result.tiles)} PMTiles, {len(result.styles)} styles")
    line = f"  Docs: {'; '.join(parts)}"
    # Only "auto" silently does without tiles; "false" asked for none.
    if not result.tippecanoe_used and pmtiles == config_module.PMTILES_AUTO:
        line += " (tippecanoe not found)"
    return line


@dataclass
class PipelineContext:
    """Mutable state threaded through pipeline stages."""

    config: DatasetConfig
    # None when no selected stage reads the source input (see build_context).
    fields_input: Path | None
    output_dir: Path
    field_dataset: str
    effective_year: int | None = None
    has_temporal: bool = False
    # Raised by _validate_stage_selection only if a selected stage needs the year.
    temporal_error: stac.TemporalExtentError | None = None
    provenance: dict[str, Any] | None = None
    source: SourceRecord | None = None
    on_mask_progress: Callable[[int, int], None] | None = None
    on_mask_start: Callable[[int, int, int], None] | None = None

    # Accumulated results / state, populated as stages run.
    was_reprojected: bool = False
    source_crs: str | None = None
    chips_result: field_stats.FieldStatsResult | None = None
    crop_stats_result: crop_stats.CropStatsResult | None = None
    label_extent_result: label_extent.LabelExtentChipsResult | None = None
    scale_result: scale.ScaleResult | None = None
    land_cover_result: land_cover.LandCoverResult | None = None
    field_embeddings_result: field_embeddings.FieldEmbeddingsResult | None = None
    splits_result: splits.CreateSplitsResult | None = None
    boundaries_result: boundaries.CreateBoundariesResult | None = None
    masks_results: dict[str, masks.CreateMasksResult] = field(default_factory=dict)
    # (mask type, grid id, reason) for every mask a create_masks call could not
    # produce, accumulated across all mask types run in this pipeline invocation.
    masks_skipped: list[tuple[str, str, str]] = field(default_factory=list)
    stac_result: stac.STACGenerationResult | None = None
    selection_result: Any = None
    download_result: Any = None
    preview_result: Any = None
    docs_result: DocsStageResult | None = None

    # Derived output paths (fixed naming convention).
    output_fields_path: Path = field(init=False)
    field_polygons_path: Path = field(init=False)
    chips_path: Path = field(init=False)
    boundary_lines_path: Path = field(init=False)
    field_embeddings_path: Path = field(init=False)
    chips_base_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        name = self.field_dataset
        self.output_fields_path = self.output_dir / f"{name}_fields.parquet"
        # Field polygons consumed by chips/splits/boundaries/masks. When a class
        # filter is configured, these stages read the filtered file; otherwise the
        # full reprojected fields file. STAC always references output_fields_path.
        if self.config.class_filter is not None:
            self.field_polygons_path = self.output_dir / f"{name}_fields_filtered.parquet"
        else:
            self.field_polygons_path = self.output_fields_path
        self.chips_path = self.output_dir / f"{name}_chips.parquet"
        self.boundary_lines_path = self.output_dir / f"{name}_boundary_lines.parquet"
        self.field_embeddings_path = self.output_dir / f"{name}_field_embeddings.parquet"
        self.chips_base_dir = stac.chips_base_dir_for(self.output_dir)

    @property
    def field_polygons_producer(self) -> str:
        """Which stage produces field_polygons_path (for input-error messages)."""
        return "filter" if self.config.class_filter is not None else "reproject"


def _input_name_stem(config: DatasetConfig) -> str:
    """Default dataset name stem, from the URL path or the local filename.

    A pure string derivation: it never touches the network or the filesystem, so
    it works even when the source is never resolved.
    """
    if is_url(config.fields_file):
        return Path(urlsplit(config.fields_file).path).stem or "source"
    return Path(config.fields_file).stem


def _resolve_input(config: DatasetConfig) -> tuple[Path, SourceRecord]:
    """Resolve ``config.fields_file`` to a local path, fetching URLs into the cache.

    Returns the local fields path and the source record describing where the
    bytes came from.

    Raises:
        FileNotFoundError: If a local input fields file does not exist.
    """
    if is_url(config.fields_file):
        fetch_cfg = config.stages.fetch
        logger.info(f"Fetching source {config.fields_file}...")
        record = fetch_source(
            config.fields_file,
            Path(fetch_cfg.cache_dir).expanduser(),
            refresh=fetch_cfg.refresh,
        )
        logger.info(
            "Using cached copy" if record.fetched_at is None else f"Fetched {record.size:,} bytes"
        )
        return record.local_path, record

    fields_path = Path(config.fields_file).resolve()
    if not fields_path.exists():
        raise FileNotFoundError(f"Fields file not found: {fields_path}")
    return fields_path, describe_local_source(fields_path)


def _record_source_provenance(ctx: PipelineContext, source: SourceRecord | None) -> None:
    """Fill in the provenance ``source`` block and the ftwd commit.

    When this run never resolved the source, the record written by an earlier run
    into the output directory is carried forward, so re-running a single stage
    does not drop provenance from the catalog.
    """
    provenance = ctx.provenance
    if provenance is None:
        return
    if source is not None:
        provenance["source"] = source.to_dict(ctx.config.source_via)
    else:
        prior = config_module.read_provenance_file(ctx.output_dir)
        if prior is not None and prior.get("source"):
            provenance["source"] = prior["source"]
    if provenance.get("ftwd_git_commit") is None:
        provenance["ftwd_git_commit"] = installed_git_commit()


def _detect_temporal(ctx: PipelineContext) -> None:
    """Set ``effective_year`` / ``has_temporal`` from the best available fields file.

    Prefers the resolved source input; falls back to the reprojected fields file
    from an earlier run so a source-free run (e.g. ``--only stac``) can still find
    a datetime column without fetching the source.
    """
    ctx.effective_year = ctx.config.year
    fields_path = ctx.fields_input
    if fields_path is None and ctx.output_fields_path.exists():
        fields_path = ctx.output_fields_path
    if fields_path is not None and ctx.config.year is None:
        logger.info("Checking temporal extent availability...")
        try:
            ctx.effective_year = stac.resolve_year(fields_path)
        except stac.TemporalExtentError as err:
            ctx.temporal_error = err
        if ctx.effective_year is not None:
            logger.info(
                f"Using year {ctx.effective_year} from determination_datetime for chip naming"
            )
    ctx.has_temporal = ctx.effective_year is not None


def build_context(
    config: DatasetConfig,
    *,
    stages: list[str] | None = None,
    on_mask_progress: Callable[[int, int], None] | None = None,
    on_mask_start: Callable[[int, int, int], None] | None = None,
    provenance: dict[str, Any] | None = None,
) -> PipelineContext:
    """Resolve paths, detect temporal extent, and prepare the output directory.

    A URL ``fields_file`` is fetched into ``config.stages.fetch.cache_dir`` (or
    reused from a prior fetch); a local path is hashed for provenance instead.
    Pass ``stages`` (the stages about to run) to skip that work entirely when no
    selected stage reads the source input; omitting it resolves the source, as a
    full run does.

    Raises:
        FileNotFoundError: If a local input fields file does not exist.
    """
    name_stem = _input_name_stem(config)
    field_dataset = config.name or name_stem
    out_dir = (
        Path(config.output_dir).resolve()
        if config.output_dir is not None
        else Path(f"{name_stem}-dataset").resolve()
    )

    if stages is None or any(stage in _SOURCE_STAGES for stage in stages):
        fields_path, source = _resolve_input(config)
    else:
        logger.info("No selected stage reads the source input; skipping source resolution.")
        fields_path, source = None, None

    ctx = PipelineContext(
        config=config,
        fields_input=fields_path,
        output_dir=out_dir,
        field_dataset=field_dataset,
        provenance=provenance,
        source=source,
        on_mask_progress=on_mask_progress,
        on_mask_start=on_mask_start,
    )
    _record_source_provenance(ctx, source)
    _detect_temporal(ctx)
    return ctx


def resolve_stages(
    only: str | None = None,
    from_stage: str | None = None,
    through_stage: str | None = None,
    *,
    config: DatasetConfig | None = None,
) -> list[str]:
    """Determine which stages to run from --only/--from/--through selectors.

    Enabled-gated imagery stages are dropped from a range/full run when their
    config section is disabled, but ``--only`` always forces the named stage.
    """
    for name in (only, from_stage, through_stage):
        if name is not None and name not in STAGE_ORDER:
            raise ValueError(f"Unknown stage '{name}'. Valid stages: {', '.join(STAGE_ORDER)}.")

    if only is not None:
        return [only]

    start = STAGE_ORDER.index(from_stage) if from_stage else 0
    end = STAGE_ORDER.index(through_stage) + 1 if through_stage else len(STAGE_ORDER)
    selected = STAGE_ORDER[start:end]

    if config is not None:
        selected = [s for s in selected if _stage_enabled(s, config)]
    return selected


def _stage_enabled(stage: str, config: DatasetConfig) -> bool:
    if stage == "filter":
        return config.class_filter is not None
    if stage == "scale":
        return config.stages.scale.percent < 100
    if stage == "select_images":
        return config.stages.select_images.enabled
    if stage == "download_images":
        return config.stages.download_images.enabled
    return True


def run_pipeline(
    ctx: PipelineContext,
    stages_to_run: list[str],
    *,
    before_stage: dict[str, Callable[[PipelineContext], None]] | None = None,
) -> PipelineContext:
    """Validate and execute the requested stages in order, mutating ``ctx``.

    ``before_stage`` maps a stage name to work that runs at that stage's position in
    :data:`STAGE_ORDER`, whether or not the stage itself was selected. It is how
    ``create-dataset`` slots its own imagery selection and download into the run:
    both entry points then take their ordering from ``STAGE_ORDER`` alone, so the
    docs stage stays downstream of imagery in either one.
    """
    _validate_stage_selection(ctx, stages_to_run)
    hooks = dict(before_stage or {})
    for name in hooks:
        if name not in STAGE_ORDER:
            raise ValueError(f"Unknown stage '{name}'. Valid stages: {', '.join(STAGE_ORDER)}.")

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    ctx.chips_base_dir.mkdir(exist_ok=True)
    if ctx.provenance is not None:
        config_module.write_provenance_file(ctx.provenance, ctx.output_dir)

    for stage in STAGE_ORDER:
        hook = hooks.get(stage)
        if hook is not None:
            hook(ctx)
        if stage in stages_to_run:
            _STAGE_FUNCS[stage](ctx)

    success(logger, "Pipeline complete!")
    return ctx


def _validate_stage_selection(ctx: PipelineContext, stages_to_run: list[str]) -> None:
    """Front-load validation so failures surface before any heavy work runs."""
    if "splits" in stages_to_run:
        split_type = ctx.config.stages.splits.split_type
        if split_type is None:
            raise ValueError("split_type is required and cannot be None")
        if split_type not in splits.SPLIT_TYPE_CHOICES:
            raise ValueError(f"split_type must be one of: {splits.SPLIT_TYPE_CHOICES_STR}")

    if any(s in _YEAR_STAGES for s in stages_to_run) and not ctx.has_temporal:
        if ctx.temporal_error is not None:
            raise ctx.temporal_error
        raise ValueError(
            "Cannot determine temporal extent for STAC catalog. "
            "Either provide a 'year' or ensure the fields file has a "
            "'determination_datetime' column."
        )

    # Mosaics alone for a year without them fail here, before any heavy work.
    if "select_images" in stages_to_run and ctx.effective_year is not None:
        require_mosaic_years([ctx.effective_year], ctx.config.stages.select_images.imagery_modes)


def _require(path: Path, *, stage: str, produced_by: str) -> None:
    """Ensure a stage's input exists, with a clear message for standalone runs."""
    if not path.exists():
        raise StageInputError(
            f"Stage '{stage}' requires {path.name}, which is missing. "
            f"Run the '{produced_by}' stage first."
        )


def _label_extent_mismatch(configured: str | None, built: str | None) -> str:
    """Why chips built with extent fingerprint ``built`` don't fit the ``configured`` one."""
    if configured is None:
        return "is not set, but {chips} was built with one"
    if built is None:
        return "is set, but {chips} was built without one"
    return "is set, but {chips} was built from a different extent file (or an older version of it)"


def _require_label_extent_match(ctx: PipelineContext, *, stage: str) -> None:
    """Ensure the chips file was built with the label extent this run is configured with.

    The extent drops chips and records ``label_extent_pct`` in the chips stage only, so
    a run starting later (``--from masks``) after the setting or the extent file changed
    would mix the two. The chips file records the extent file's sha256 for this check.
    """
    extent_file = ctx.config.stages.masks.label_extent
    configured = label_extent.extent_fingerprint(extent_file) if extent_file else None
    built = label_extent.chips_fingerprint(ctx.chips_path)
    if configured == built:
        return
    reason = _label_extent_mismatch(configured, built).format(chips=ctx.chips_path.name)
    raise StageInputError(
        f"Stage '{stage}': stages.masks.label_extent {reason}. Rerun from the 'chips' stage."
    )


# ---- stage implementations ----------------------------------------------


def stage_reproject(ctx: PipelineContext) -> None:
    """Reproject the input to EPSG:4326 if needed, else copy it into the output."""
    if ctx.fields_input is None:
        raise StageInputError(
            "Stage 'reproject' needs the source input, but this context was built "
            "without resolving it. Rebuild the context including the reproject stage."
        )
    logger.info("Checking CRS...")
    geom_col = detect_geometry_column(ctx.fields_input) or "geometry"
    crs_info = detect_crs(ctx.fields_input, geom_col)
    ctx.source_crs = str(crs_info)

    if crs_info.authority_code is None or crs_info.authority_code.upper() != "EPSG:4326":
        if ctx.config.skip_reproject:
            raise ValueError(
                f"Input file has CRS '{crs_info}' but EPSG:4326 is required. "
                "Remove skip_reproject to auto-reproject."
            )
        logger.info(f"Reprojecting from {crs_info} to EPSG:4326...")
        reproject(
            ctx.fields_input,
            ctx.output_fields_path,
            target_crs="EPSG:4326",
        )
        ctx.was_reprojected = True
        _drop_declared_crs(ctx.output_fields_path)
        logger.info(f"Reprojected to: {ctx.output_fields_path}")
    elif _declares_crs(crs_info):
        logger.info("CRS is EPSG:4326, rewriting with the default CRS...")
        _write_default_crs(ctx.fields_input, ctx.output_fields_path)
    else:
        logger.info("CRS is already EPSG:4326, copying to output directory...")
        shutil.copy2(ctx.fields_input, ctx.output_fields_path)


def _declares_crs(crs_info: CRSInfo) -> bool:
    return crs_info.projjson is not None or crs_info.wkt is not None


def _drop_declared_crs(path: Path) -> None:
    """Rewrite lon/lat fields in place so they carry the default CRS.

    An explicit EPSG:4326 makes DuckDB refuse to intersect these fields with the
    chips, which carry the default OGC:CRS84, and the published queries join them.
    """
    if not _declares_crs(detect_crs(path)):
        return
    tmp = path.with_suffix(".default-crs.parquet")
    _write_default_crs(path, tmp)
    tmp.replace(path)


def _write_default_crs(src: Path | str, dst: Path) -> None:
    """Copy lon/lat GeoParquet without an explicit CRS, i.e. with the default OGC:CRS84.

    Only the ``geo`` metadata changes, so this streams the row groups through pyarrow;
    geoparquet-io has no way to drop a declared CRS without reprojecting.
    """
    import json

    import pyarrow.parquet as pq

    source = pq.ParquetFile(src)
    metadata = dict(source.schema_arrow.metadata)
    geo = json.loads(metadata[b"geo"])
    for column in geo.get("columns", {}).values():
        column.pop("crs", None)
    metadata[b"geo"] = json.dumps(geo).encode()
    schema = source.schema_arrow.with_metadata(metadata)
    with pq.ParquetWriter(dst, schema, compression="zstd") as writer:
        for batch in source.iter_batches():
            writer.write_batch(batch)


def stage_filter(ctx: PipelineContext) -> None:
    """Apply the class filter, writing field-only polygons for downstream stages.

    No-op when no class filter is configured. The full source stays in
    output_fields_path; only include-classes are written to field_polygons_path.
    """
    cf = ctx.config.class_filter
    if cf is None:
        return
    _require(ctx.output_fields_path, stage="filter", produced_by="reproject")

    column = class_filter.resolve_column(ctx.output_fields_path, cf)
    logger.info(f"Filtering fields by column '{column}'...")
    distinct = class_filter.get_distinct_classes(ctx.output_fields_path, column)
    cf.validate_against(distinct)
    class_filter.write_filtered_fields(
        ctx.output_fields_path, ctx.field_polygons_path, cf, column=column
    )
    logger.info(
        f"Wrote filtered fields ({len(cf.include)} field class(es)): {ctx.field_polygons_path.name}"
    )


def _subset_local_grid(ctx: PipelineContext, grid_file: str) -> str:
    """Pre-filter a local grid to the fields' bounds so chips never loads a huge grid.

    A global FTW grid can have tens of millions of cells; loading it whole would
    exhaust memory. When the grid has a ``bbox`` column, DuckDB prunes row groups
    by bbox statistics, so this is fast. Returns the (small) subset path, or the
    original path if the grid has no bbox column to filter on.
    """
    conn = duckdb.connect(":memory:")
    ensure_spatial_loaded(conn)
    try:
        grid_sql = sql_path(grid_file)
        grid_cols = [r[0] for r in conn.execute(f"DESCRIBE SELECT * FROM '{grid_sql}'").fetchall()]
        if "bbox" not in grid_cols:
            logger.info("Local grid has no bbox column; loading it in full (may be slow).")
            return grid_file

        # Fields bounds from their bbox column (present after reproject/filter).
        xmin, ymin, xmax, ymax = conn.execute(
            f"SELECT MIN(bbox.xmin), MIN(bbox.ymin), MAX(bbox.xmax), MAX(bbox.ymax) "
            f"FROM '{sql_path(ctx.field_polygons_path)}'"
        ).fetchone()

        subset_path = ctx.output_dir / f"{ctx.field_dataset}_grid.parquet"
        subset_sql = sql_path(subset_path)
        conn.execute(
            f"COPY (SELECT * FROM '{grid_sql}' WHERE bbox.xmin <= {xmax} "
            f"AND bbox.xmax >= {xmin} AND bbox.ymin <= {ymax} AND bbox.ymax >= {ymin}) "
            f"TO '{subset_sql}' (FORMAT PARQUET)"
        )
        n = conn.execute(f"SELECT COUNT(*) FROM '{subset_sql}'").fetchone()[0]
        logger.info(f"Subset local grid to {n:,} cells within fields bounds -> {subset_path.name}")
        return str(subset_path)
    finally:
        conn.close()


def stage_chips(ctx: PipelineContext) -> None:
    """Create chips with field coverage statistics."""
    _require(ctx.field_polygons_path, stage="chips", produced_by=ctx.field_polygons_producer)
    chips_cfg = ctx.config.stages.chips
    grid_file = chips_cfg.grid_file
    grid_source = chips_cfg.grid_source or field_stats.DEFAULT_FTW_GRID_SOURCE
    if grid_file:
        logger.info(f"Using local FTW grid: {grid_file}")
        grid_file = _subset_local_grid(ctx, grid_file)
    elif chips_cfg.grid_source:
        logger.info(f"Using grid source: {grid_source}")
    logger.info("Creating chips with field coverage statistics...")
    ctx.chips_result = field_stats.add_field_stats(
        fields_file=str(ctx.field_polygons_path),
        grid_file=grid_file,
        grid_source=grid_source,
        output_file=str(ctx.chips_path),
        min_coverage=ctx.config.stages.chips.min_coverage,
        min_chip_area=chips_cfg.min_chip_area if chips_cfg.min_chip_area > 0 else None,
        km_size=chips_cfg.km_size,
        drop_border_chips=ctx.config.stages.chips.drop_border_chips,
        border_gap_chips=ctx.config.stages.chips.border_gap_chips,
        batch_size=chips_cfg.coverage_batch_size,
    )
    logger.info(
        f"Created chips: {ctx.chips_result.total_cells:,} cells, "
        f"{ctx.chips_result.cells_with_coverage:,} with coverage"
    )
    extent_file = ctx.config.stages.masks.label_extent
    if extent_file is not None:
        ctx.label_extent_result = label_extent.apply_label_extent(ctx.chips_path, extent_file)
    if ctx.config.stages.chips.crop_stats:
        # chips and field_polygons_path share a CRS: chips derive from the same
        # (reprojected, class-filtered) fields used here, so no reprojection is needed.
        ctx.crop_stats_result = crop_stats.add_crop_stats(ctx.chips_path, ctx.field_polygons_path)
    else:
        # Never publish a previous run's composition when the step is turned off.
        crop_stats.drop_crop_stats(ctx.chips_path)
    _add_land_cover(ctx)
    _add_admin_names(ctx)
    _add_field_embeddings(ctx)


def _add_field_embeddings(ctx: PipelineContext) -> None:
    """Write the per-field embedding statistics, unless the file already matches.

    Reading the embeddings under a national dataset takes hours, so a file written for
    the same fields (by content, not timestamp: reproject rewrites the fields file on
    every run), year and products is kept.
    """
    path = ctx.field_embeddings_path
    if not ctx.config.stages.chips.field_embeddings:
        if path.exists():
            path.unlink()
            logger.info(f"Removed {path.name} (stages.chips.field_embeddings is off)")
        shutil.rmtree(path.with_name(path.name + ".partial"), ignore_errors=True)
        return
    if ctx.effective_year is None:
        raise StageInputError(
            "stages.chips.field_embeddings needs the dataset year: set year in the config"
        )
    if field_embeddings.is_current(path, ctx.output_fields_path, year=ctx.effective_year):
        logger.info(f"Keeping {path.name}: same fields, year and products")
        return
    ctx.field_embeddings_result = field_embeddings.add_field_embeddings(
        ctx.output_fields_path, path, year=ctx.effective_year
    )


def _add_admin_names(ctx: PipelineContext) -> None:
    """Run the admin names step, or clear a previous run's columns when it is off."""
    if not ctx.config.stages.chips.admin_names:
        admin_names.drop_admin_names(ctx.chips_path)
        return
    admin_names.add_admin_names(ctx.chips_path)


def _add_land_cover(ctx: PipelineContext) -> None:
    """Run the land cover step, or clear a previous run's columns when it is off."""
    if not ctx.config.stages.chips.land_cover:
        land_cover.drop_land_cover(ctx.chips_path)
        return
    ctx.land_cover_result = land_cover.add_land_cover(ctx.chips_path, year=ctx.effective_year)


def stage_scale(ctx: PipelineContext) -> None:
    """Keep the configured percent of chip blocks and record each chip's scale score."""
    _require(ctx.chips_path, stage="scale", produced_by="chips")
    scale_cfg = ctx.config.stages.scale
    ctx.scale_result = scale.apply_scale(
        ctx.chips_path,
        percent=scale_cfg.percent,
        min_blocks_per_square=scale_cfg.min_blocks_per_square,
        km_size=ctx.config.stages.chips.km_size,
    )
    logger.info(
        f"Scale {scale_cfg.percent:g}%: kept {ctx.scale_result.kept_chips:,} of "
        f"{ctx.scale_result.total_chips:,} chips"
    )


def stage_splits(ctx: PipelineContext) -> None:
    """Assign train/val/test splits to the chips file."""
    _require(ctx.chips_path, stage="splits", produced_by="chips")
    split_cfg = ctx.config.stages.splits
    logger.info(f"Assigning {split_cfg.split_type} splits...")
    ctx.splits_result = splits.assign_splits(
        chips_file=str(ctx.chips_path),
        split_type=split_cfg.split_type,
        split_percents=split_cfg.split_percents,
        random_seed=split_cfg.random_seed,
        fields_file=str(ctx.field_polygons_path),
        km_size=ctx.config.stages.chips.km_size,
    )


def stage_boundaries(ctx: PipelineContext) -> None:
    """Create boundary lines from field polygons."""
    _require(ctx.field_polygons_path, stage="boundaries", produced_by=ctx.field_polygons_producer)
    logger.info("Creating boundary lines...")
    result = boundaries.create_boundaries(
        input_path=str(ctx.field_polygons_path),
        output_dir=str(ctx.output_dir),
        output_prefix=f"{ctx.field_dataset}_boundary_lines_",
    )
    # Rename to the canonical naming convention.
    if result.files_processed:
        original_output = result.files_processed[0].output_path
        if original_output != ctx.boundary_lines_path:
            shutil.move(str(original_output), str(ctx.boundary_lines_path))
    ctx.boundaries_result = result
    logger.info(f"Created boundary lines: {result.total_features:,} features")


# String name -> (enum, subdir key used as masks_results key).
_MASK_TYPE_MAPPING = [
    (MaskType.INSTANCE, "instance", "instance"),
    (MaskType.SEMANTIC_2_CLASS, "semantic_2class", "semantic_2_class"),
    (MaskType.SEMANTIC_3_CLASS, "semantic_3class", "semantic_3_class"),
    (MaskType.DECODE_BOUNDARY, "decode_boundary", "decode_boundary"),
    (MaskType.DECODE_DISTANCE, "decode_distance", "decode_distance"),
]


def _build_chip_dirs(ctx: PipelineContext) -> dict[str, Path]:
    """Create per-chip directories for grids above the coverage threshold."""
    return masks.build_chip_dirs(
        chips_file=ctx.chips_path,
        chips_base_dir=ctx.chips_base_dir,
        min_coverage=ctx.config.stages.chips.min_coverage,
        year=ctx.effective_year,
    )


# Reasons are truncated before grouping/logging so one exceptionally long
# message (e.g. a full stack-trace-like string) doesn't dominate the summary.
_SKIPPED_REASON_MAX_LEN = 160
_SKIPPED_REASONS_TO_LOG = 3


def _log_skipped_masks(
    ctx: PipelineContext, mask_type: MaskType, mask_result: masks.CreateMasksResult
) -> None:
    """Log the top skipped-mask reasons and accumulate them on the context.

    Per-cell failures are otherwise silent: create_masks only returns counts,
    so without this a build can lose masks (e.g. Austria's non-numeric field
    ids) with nothing in the log explaining why.
    """
    if mask_result.total_skipped == 0:
        return

    reason_counts: dict[str, int] = {}
    for grid_id, reason in mask_result.masks_skipped:
        ctx.masks_skipped.append((mask_type.value, grid_id, reason))
        truncated = reason[:_SKIPPED_REASON_MAX_LEN]
        reason_counts[truncated] = reason_counts.get(truncated, 0) + 1

    top_reasons = sorted(reason_counts.items(), key=lambda item: item[1], reverse=True)
    for reason, count in top_reasons[:_SKIPPED_REASONS_TO_LOG]:
        logger.warning(
            f"Skipped {mask_result.total_skipped} {mask_type.value} mask(s): {reason} (x{count})"
        )


def stage_masks(ctx: PipelineContext) -> None:
    """Create the requested raster mask types for each chip."""
    _require(ctx.chips_path, stage="masks", produced_by="chips")
    _require(ctx.field_polygons_path, stage="masks", produced_by=ctx.field_polygons_producer)
    _require(ctx.boundary_lines_path, stage="masks", produced_by="boundaries")
    _require_label_extent_match(ctx, stage="masks")

    chip_dirs = _build_chip_dirs(ctx)
    logger.info(f"Created {len(chip_dirs)} chip directories")

    masks_cfg = ctx.config.stages.masks
    requested = [
        (mask_type, subdir_name)
        for mask_type, subdir_name, type_name in _MASK_TYPE_MAPPING
        if type_name in masks_cfg.mask_types
    ]
    background_class_value = 3 if masks_cfg.presence_only else 0

    requested_types = [mask_type for mask_type, _ in requested]
    subdir_by_type = dict(requested)

    # Types sharing a rasterization are burned once, so this is a single pass over
    # the chips rather than one pass per type.
    logger.info(f"Creating {', '.join(m.value for m in requested_types)} masks...")
    mask_results = masks.create_masks(
        chips_file=str(ctx.chips_path),
        boundaries_file=str(ctx.field_polygons_path),
        boundary_lines_file=str(ctx.boundary_lines_path),
        output_dir=str(ctx.chips_base_dir),
        field_dataset=ctx.field_dataset,
        mask_types=requested_types,
        min_coverage=ctx.config.stages.chips.min_coverage,
        resolution=masks_cfg.resolution,
        num_workers=masks_cfg.workers,
        chip_dirs=chip_dirs,
        year=ctx.effective_year,
        background_class_value=background_class_value,
        skip_existing=masks_cfg.skip_existing,
        crop_column=masks_cfg.crop_column,
        utm=masks_cfg.utm,
        label_extent_file=masks_cfg.label_extent,
        on_progress=ctx.on_mask_progress,
        on_start=ctx.on_mask_start,
    )

    for mask_type, mask_result in mask_results.items():
        ctx.masks_results[subdir_by_type[mask_type]] = mask_result
        logger.info(f"Created {mask_result.total_created} {mask_type.value} masks")
        _log_skipped_masks(ctx, mask_type, mask_result)

    # Every type shares one worker pool, so the count is the same on each result.
    pool_restarts = max((r.pool_restarts for r in mask_results.values()), default=0)
    if pool_restarts > 0:
        logger.warning(
            f"Worker pool restarted {pool_restarts} time(s) "
            "(a worker died; lower stages.masks.workers if this repeats)"
        )


def stage_stac(ctx: PipelineContext) -> None:
    """Generate the STAC static catalog, embedding provenance if available."""
    _require(ctx.chips_path, stage="stac", produced_by="chips")
    _require(ctx.output_fields_path, stage="stac", produced_by="reproject")
    _require(ctx.boundary_lines_path, stage="stac", produced_by="boundaries")
    if ctx.config.class_filter is not None:
        _require(ctx.field_polygons_path, stage="stac", produced_by="filter")
    _require_label_extent_match(ctx, stage="stac")

    # The chips stage drops the composition columns when the step is off, but a run
    # starting at or after splits never reaches it, so a chips file left by an earlier
    # run would republish that run's composition onto every item. Dropping here, at the
    # only stage that publishes them, is a no-op when they are absent.
    if not ctx.config.stages.chips.crop_stats and crop_stats.drop_crop_stats(ctx.chips_path):
        logger.info("Dropped stale crop composition columns from the chips file")
    # The same holds for land cover, including columns added by ``ftwd add-land-cover``:
    # the config decides what is published, so a backfill needs land_cover: true.
    if not ctx.config.stages.chips.land_cover and land_cover.drop_land_cover(ctx.chips_path):
        logger.info(
            "Dropped land cover columns from the chips file "
            "(set stages.chips.land_cover: true to publish them)"
        )
    if not ctx.config.stages.chips.admin_names and admin_names.drop_admin_names(ctx.chips_path):
        logger.info(
            "Dropped admin name columns from the chips file "
            "(set stages.chips.admin_names: true to publish them)"
        )

    logger.info("Generating STAC catalog...")
    ctx.stac_result = stac.generate_stac_catalog(
        output_dir=ctx.output_dir,
        field_dataset=ctx.field_dataset,
        fields_file=ctx.output_fields_path,
        chips_file=ctx.chips_path,
        boundary_lines_file=ctx.boundary_lines_path,
        filtered_fields_file=(
            ctx.field_polygons_path if ctx.config.class_filter is not None else None
        ),
        field_embeddings_file=_published_field_embeddings(ctx),
        year=ctx.effective_year,
        provenance=ctx.provenance,
        checksums=ctx.config.stages.stac.checksums,
        crop_calendar=ctx.config.stages.stac.crop_calendar,
        background_class_value=3 if ctx.config.stages.masks.presence_only else 0,
        label_extent=ctx.config.stages.masks.label_extent is not None,
        config=ctx.config,
    )
    n = ctx.stac_result.total_items
    k = len(ctx.stac_result.subcatalog_paths)
    logger.info(f"Created STAC collection with {n} items in {k} sub-catalog(s)")


def _published_field_embeddings(ctx: PipelineContext) -> Path | None:
    """The field embeddings file to publish: required when the step is on, never when off."""
    if not ctx.config.stages.chips.field_embeddings:
        return None
    _require(ctx.field_embeddings_path, stage="stac", produced_by="chips")
    try:
        current = ctx.effective_year is not None and field_embeddings.is_current(
            ctx.field_embeddings_path, ctx.output_fields_path, year=ctx.effective_year
        )
    except field_embeddings.FieldEmbeddingsError as err:
        raise StageInputError(str(err)) from err
    if not current:
        raise StageInputError(
            f"{ctx.field_embeddings_path.name} is not for the current fields, year and "
            "products. Run the 'chips' stage first."
        )
    return ctx.field_embeddings_path


def stage_select_images(ctx: PipelineContext) -> None:
    """Select Sentinel-2 imagery (seasonal scenes, quarterly mosaics, or both) for each chip."""
    _require(ctx.output_dir / "collection.json", stage="select_images", produced_by="stac")
    select_cfg = ctx.config.stages.select_images
    modes = select_cfg.imagery_modes
    # Scenes need a year; mosaics use the same one, or each chip's own without it.
    if "scenes" in modes and ctx.effective_year is None:
        raise ValueError("A year is required for image selection.")

    logger.info("Selecting imagery...")
    ctx.selection_result = select_imagery_modes_for_catalog(
        ctx.output_dir,
        modes,
        year=ctx.effective_year,
        cloud_cover_chip=select_cfg.cloud_cover_chip,
        nodata_max=select_cfg.nodata_max,
        buffer_days=select_cfg.buffer_days,
        num_buffer_expansions=select_cfg.num_buffer_expansions,
        buffer_expansion_size=select_cfg.buffer_expansion_size,
        workers=select_cfg.effective_workers,
        search_backend=select_cfg.search_backend,
    )


def stage_download_images(ctx: PipelineContext) -> None:
    """Download selected imagery and generate thumbnails."""
    _require(
        ctx.output_dir / "collection.json",
        stage="download_images",
        produced_by="stac",
    )
    download_cfg = ctx.config.stages.download_images
    slots = None
    if download_cfg.mode == DOWNLOAD_MODE_PREVIEW:
        logger.info("Rendering chip previews from the remote scenes...")
        ctx.preview_result = preview_imagery_for_catalog(
            catalog_dir=ctx.output_dir,
            resume=download_cfg.resume,
            workers=download_cfg.workers,
        )
        # Preview covers scenes only; mosaics have no remote true-colour image, so
        # alongside scenes they are clipped as usual.
        if "mosaics" not in ctx.config.stages.select_images.imagery_modes:
            return
        slots = MOSAIC_SLOTS
    logger.info("Downloading imagery...")
    ctx.download_result = download_imagery_for_catalog(
        catalog_dir=ctx.output_dir,
        bands=download_cfg.bands,
        resolution=download_cfg.resolution,
        workers=download_cfg.workers,
        # Defaults to True: a pipeline run is resumable by definition, every other
        # stage reuses what is already on disk. Without it a rerun re-attempts every
        # chip - and fails on all of them, because a completed download replaces the
        # child's band assets with the local `image`, so there are no band hrefs left
        # to fetch. Set stages.download_images.resume false to force a re-download,
        # which is what a changed `bands` or `resolution` needs.
        resume=download_cfg.resume,
        slots=slots,
    )


TIPPECANOE_MISSING_WARNING = (
    "tippecanoe not found; no PMTiles or styles were written "
    "(install tippecanoe or set stages.docs.pmtiles: false to silence)"
)

# (asset key, context attribute holding the source parquet, output name, tile spec)
_TILE_TARGETS = (
    ("chips_tiles", "chips_path", "chips.pmtiles", tiles.CHIPS_TILES),
    ("fields_tiles", "field_polygons_path", "fields.pmtiles", tiles.FIELDS_TILES),
)


def _build_tiles(ctx: PipelineContext) -> dict[str, Path]:
    """Build the collection's PMTiles, honouring the ``pmtiles`` auto/true/false setting."""
    setting = ctx.config.stages.docs.pmtiles
    if setting is False:
        return {}
    if not tiles.tippecanoe_available():
        if setting is True:
            raise RuntimeError(
                "stages.docs.pmtiles is true but tippecanoe is not installed. "
                "Install tippecanoe, or set stages.docs.pmtiles to auto or false."
            )
        logger.warning(TIPPECANOE_MISSING_WARNING)
        return {}

    built: dict[str, Path] = {}
    for key, attribute, out_name, spec in _TILE_TARGETS:
        source = getattr(ctx, attribute)
        built[key] = tiles.build_pmtiles(source, ctx.output_dir / out_name, spec)
    return built


def _write_styles(ctx: PipelineContext, built: dict[str, Path]) -> list[styles.StyleResult]:
    """Write the MapLibre styles for whichever tile archives were built."""
    if not built:
        return []
    return styles.write_styles(
        ctx.output_dir,
        ctx.field_dataset,
        ctx.chips_path,
        ctx.field_polygons_path,
        # Styles are written under styles/, one directory below the PMTiles they
        # reference, so the embedded source URL must climb back out of styles/.
        chips_tiles=f"../{built['chips_tiles'].name}" if "chips_tiles" in built else None,
        fields_tiles=f"../{built['fields_tiles'].name}" if "fields_tiles" in built else None,
    )


def _write_docs(ctx: PipelineContext, style_results: list[styles.StyleResult]) -> list[Path]:
    """Write README.md and AGENTS.md, as far as the config asks for them."""
    docs_cfg = ctx.config.stages.docs
    if not (docs_cfg.readme or docs_cfg.agents):
        return []
    return docs.write_docs(
        ctx.output_dir,
        ctx.output_dir / "collection.json",
        ctx.chips_path,
        ctx.field_polygons_path,
        style_results,
        ctx.config.config_dict(),
        readme=docs_cfg.readme,
        agents=docs_cfg.agents,
    )


def stage_docs(ctx: PipelineContext) -> None:
    """Build tiles and styles, write the documents, and register them on the collection."""
    collection_json = ctx.output_dir / "collection.json"
    _require(collection_json, stage="docs", produced_by="stac")
    _require(ctx.chips_path, stage="docs", produced_by="chips")
    _require(ctx.field_polygons_path, stage="docs", produced_by=ctx.field_polygons_producer)

    built = _build_tiles(ctx)
    style_results = _write_styles(ctx, built)
    written = _write_docs(ctx, style_results)
    ctx.docs_result = DocsStageResult(built, style_results, written, bool(built))

    # Registered even when everything is empty: that call is also what retracts the
    # tiles, styles and document links an earlier run left in the collection.
    docs.register_docs_assets(collection_json, tiles=built, styles=style_results, docs=written)

    if not (built or written):
        logger.info("Nothing to document: PMTiles are off and README/AGENTS are disabled")
        return

    logger.info(
        f"Documented the collection: {len(built)} tile archive(s), "
        f"{len(style_results)} style(s), {len(written)} document(s)"
    )


_STAGE_FUNCS: dict[str, Callable[[PipelineContext], None]] = {
    "reproject": stage_reproject,
    "filter": stage_filter,
    "chips": stage_chips,
    "scale": stage_scale,
    "splits": stage_splits,
    "boundaries": stage_boundaries,
    "masks": stage_masks,
    "stac": stage_stac,
    "select_images": stage_select_images,
    "download_images": stage_download_images,
    "docs": stage_docs,
}
