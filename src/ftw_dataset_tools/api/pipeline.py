"""Stage-based orchestration for FTW dataset creation.

This module is the single source of truth for the dataset build pipeline. Each
stage is a small function operating on a :class:`PipelineContext`; the flag-driven
``create-dataset`` command and the config-driven ``ftwd run`` command both build a
context and execute the same stages.

Intermediate outputs live in the output directory under a fixed naming
convention, so any stage can be re-run on its own as long as its inputs exist:

    {output_dir}/{name}_fields.parquet          (reproject)
    {output_dir}/{name}_chips.parquet           (chips, splits)
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
    boundaries,
    class_filter,
    crop_stats,
    docs,
    field_stats,
    masks,
    splits,
    stac,
    styles,
    tiles,
)
from ftw_dataset_tools.api import config as config_module
from ftw_dataset_tools.api.geo import (
    detect_crs,
    detect_geometry_column,
    ensure_spatial_loaded,
    reproject,
)
from ftw_dataset_tools.api.imagery import (
    download_imagery_for_catalog,
    select_imagery_for_catalog,
)
from ftw_dataset_tools.api.masks import MaskType, get_item_id, get_mgrs_square
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

# Ordered list of all pipeline stages.
STAGE_ORDER = [
    "reproject",
    "filter",
    "chips",
    "splits",
    "boundaries",
    "masks",
    "stac",
    "select_images",
    "download_images",
    "docs",
]

# Stages that require a resolvable calendar year (for chip naming, temporal
# extent, or imagery search).
_YEAR_STAGES = {"masks", "stac", "select_images", "download_images"}


class StageInputError(ValueError):
    """Raised when a stage is run without its required input files present."""


@dataclass
class DocsStageResult:
    """What the docs stage produced: tile archives, styles and documents."""

    tiles: dict[str, Path] = field(default_factory=dict)
    styles: list[styles.StyleResult] = field(default_factory=list)
    docs: list[Path] = field(default_factory=list)
    tippecanoe_used: bool = False


@dataclass
class PipelineContext:
    """Mutable state threaded through pipeline stages."""

    config: DatasetConfig
    fields_input: Path
    output_dir: Path
    field_dataset: str
    effective_year: int | None
    has_temporal: bool
    provenance: dict[str, Any] | None = None
    source: SourceRecord | None = None
    on_progress: Callable[[str], None] | None = None
    on_mask_progress: Callable[[int, int], None] | None = None
    on_mask_start: Callable[[int, int], None] | None = None

    # Accumulated results / state, populated as stages run.
    was_reprojected: bool = False
    source_crs: str | None = None
    chips_result: field_stats.FieldStatsResult | None = None
    crop_stats_result: crop_stats.CropStatsResult | None = None
    splits_result: splits.CreateSplitsResult | None = None
    boundaries_result: boundaries.CreateBoundariesResult | None = None
    masks_results: dict[str, masks.CreateMasksResult] = field(default_factory=dict)
    stac_result: stac.STACGenerationResult | None = None
    selection_result: Any = None
    download_result: Any = None
    docs_result: DocsStageResult | None = None

    # Derived output paths (fixed naming convention).
    output_fields_path: Path = field(init=False)
    field_polygons_path: Path = field(init=False)
    chips_path: Path = field(init=False)
    boundary_lines_path: Path = field(init=False)
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
        self.chips_base_dir = self.output_dir / "chips"

    @property
    def field_polygons_producer(self) -> str:
        """Which stage produces field_polygons_path (for input-error messages)."""
        return "filter" if self.config.class_filter is not None else "reproject"

    def log(self, msg: str) -> None:
        if self.on_progress:
            self.on_progress(msg)


def _resolve_input(
    config: DatasetConfig, *, log: Callable[[str], None]
) -> tuple[Path, str, SourceRecord]:
    """Resolve ``config.fields_file`` to a local path, fetching URLs into the cache.

    Returns the local fields path, the default dataset name stem (derived from
    the URL or local filename), and the source record describing where the
    bytes came from.

    Raises:
        FileNotFoundError: If a local input fields file does not exist.
    """
    if is_url(config.fields_file):
        name_stem = Path(urlsplit(config.fields_file).path).stem or "source"
        fetch_cfg = config.stages.fetch
        log(f"Fetching source {config.fields_file}...")
        record = fetch_source(
            config.fields_file,
            Path(fetch_cfg.cache_dir).expanduser(),
            refresh=fetch_cfg.refresh,
        )
        log("Using cached copy" if record.fetched_at is None else f"Fetched {record.size:,} bytes")
        return record.local_path, name_stem, record

    fields_path = Path(config.fields_file).resolve()
    if not fields_path.exists():
        raise FileNotFoundError(f"Fields file not found: {fields_path}")
    return fields_path, fields_path.stem, describe_local_source(fields_path)


def build_context(
    config: DatasetConfig,
    *,
    on_progress: Callable[[str], None] | None = None,
    on_mask_progress: Callable[[int, int], None] | None = None,
    on_mask_start: Callable[[int, int], None] | None = None,
    provenance: dict[str, Any] | None = None,
) -> PipelineContext:
    """Resolve paths, detect temporal extent, and prepare the output directory.

    A URL ``fields_file`` is fetched into ``config.stages.fetch.cache_dir`` (or
    reused from a prior fetch); a local path is hashed for provenance instead.

    Raises:
        FileNotFoundError: If a local input fields file does not exist.
    """

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    fields_path, name_stem, source = _resolve_input(config, log=log)

    field_dataset = config.name or name_stem
    out_dir = (
        Path(config.output_dir).resolve()
        if config.output_dir is not None
        else Path(f"{name_stem}-dataset").resolve()
    )

    if provenance is not None:
        provenance["source"] = source.to_dict(config.source_via)
        if provenance.get("ftwd_git_commit") is None:
            provenance["ftwd_git_commit"] = installed_git_commit()

    # Resolve the calendar year (year arg wins; else derive from datetime column).
    log("Checking temporal extent availability...")
    datetime_col = stac.detect_datetime_column(fields_path)
    effective_year = config.year
    if datetime_col:
        log(f"Found '{datetime_col}' column for temporal extent")
        if effective_year is None:
            effective_year = stac.get_year_from_datetime_column(fields_path, datetime_col)
            if effective_year:
                log(f"Using year {effective_year} from {datetime_col} for chip naming")
    elif config.year is not None:
        log(f"Using year {config.year} for temporal extent")
    has_temporal = datetime_col is not None or config.year is not None

    ctx = PipelineContext(
        config=config,
        fields_input=fields_path,
        output_dir=out_dir,
        field_dataset=field_dataset,
        effective_year=effective_year,
        has_temporal=has_temporal,
        provenance=provenance,
        source=source,
        on_progress=on_progress,
        on_mask_progress=on_mask_progress,
        on_mask_start=on_mask_start,
    )
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
    if stage == "select_images":
        return config.stages.select_images.enabled
    if stage == "download_images":
        return config.stages.download_images.enabled
    return True


def run_pipeline(ctx: PipelineContext, stages_to_run: list[str]) -> PipelineContext:
    """Validate and execute the requested stages in order, mutating ``ctx``."""
    _validate_stage_selection(ctx, stages_to_run)

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    ctx.chips_base_dir.mkdir(exist_ok=True)
    if ctx.provenance is not None:
        config_module.write_provenance_file(ctx.provenance, ctx.output_dir)

    for stage in STAGE_ORDER:
        if stage in stages_to_run:
            _STAGE_FUNCS[stage](ctx)

    ctx.log("Pipeline complete!")
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
        raise ValueError(
            "Cannot determine temporal extent for STAC catalog. "
            "Either provide a 'year' or ensure the fields file has a "
            "'determination_datetime' column."
        )


def _require(path: Path, *, stage: str, produced_by: str) -> None:
    """Ensure a stage's input exists, with a clear message for standalone runs."""
    if not path.exists():
        raise StageInputError(
            f"Stage '{stage}' requires {path.name}, which is missing. "
            f"Run the '{produced_by}' stage first."
        )


# ---- stage implementations ----------------------------------------------


def stage_reproject(ctx: PipelineContext) -> None:
    """Reproject the input to EPSG:4326 if needed, else copy it into the output."""
    ctx.log("Checking CRS...")
    geom_col = detect_geometry_column(ctx.fields_input) or "geometry"
    crs_info = detect_crs(ctx.fields_input, geom_col)
    ctx.source_crs = str(crs_info)

    if crs_info.authority_code is None or crs_info.authority_code.upper() != "EPSG:4326":
        if ctx.config.skip_reproject:
            raise ValueError(
                f"Input file has CRS '{crs_info}' but EPSG:4326 is required. "
                "Remove skip_reproject to auto-reproject."
            )
        ctx.log(f"Reprojecting from {crs_info} to EPSG:4326...")
        reproject(
            ctx.fields_input,
            ctx.output_fields_path,
            target_crs="EPSG:4326",
            on_progress=ctx.log,
        )
        ctx.was_reprojected = True
        ctx.log(f"Reprojected to: {ctx.output_fields_path}")
    else:
        ctx.log("CRS is already EPSG:4326, copying to output directory...")
        shutil.copy2(ctx.fields_input, ctx.output_fields_path)


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
    ctx.log(f"Filtering fields by column '{column}'...")
    distinct = class_filter.get_distinct_classes(ctx.output_fields_path, column)
    cf.validate_against(distinct, on_progress=ctx.log)
    class_filter.write_filtered_fields(
        ctx.output_fields_path, ctx.field_polygons_path, cf, column=column
    )
    ctx.log(
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
        grid_cols = [r[0] for r in conn.execute(f"DESCRIBE SELECT * FROM '{grid_file}'").fetchall()]
        if "bbox" not in grid_cols:
            ctx.log("Local grid has no bbox column; loading it in full (may be slow).")
            return grid_file

        # Fields bounds from their bbox column (present after reproject/filter).
        xmin, ymin, xmax, ymax = conn.execute(
            f"SELECT MIN(bbox.xmin), MIN(bbox.ymin), MAX(bbox.xmax), MAX(bbox.ymax) "
            f"FROM '{ctx.field_polygons_path}'"
        ).fetchone()

        subset_path = ctx.output_dir / f"{ctx.field_dataset}_grid.parquet"
        conn.execute(
            f"COPY (SELECT * FROM '{grid_file}' WHERE bbox.xmin <= {xmax} "
            f"AND bbox.xmax >= {xmin} AND bbox.ymin <= {ymax} AND bbox.ymax >= {ymin}) "
            f"TO '{subset_path}' (FORMAT PARQUET)"
        )
        n = conn.execute(f"SELECT COUNT(*) FROM '{subset_path}'").fetchone()[0]
        ctx.log(f"Subset local grid to {n:,} cells within fields bounds -> {subset_path.name}")
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
        ctx.log(f"Using local FTW grid: {grid_file}")
        grid_file = _subset_local_grid(ctx, grid_file)
    elif chips_cfg.grid_source:
        ctx.log(f"Using grid source: {grid_source}")
    ctx.log("Creating chips with field coverage statistics...")
    ctx.chips_result = field_stats.add_field_stats(
        fields_file=str(ctx.field_polygons_path),
        grid_file=grid_file,
        grid_source=grid_source,
        output_file=str(ctx.chips_path),
        min_coverage=ctx.config.stages.chips.min_coverage,
        drop_border_chips=ctx.config.stages.chips.drop_border_chips,
        on_progress=ctx.log,
    )
    ctx.log(
        f"Created chips: {ctx.chips_result.total_cells:,} cells, "
        f"{ctx.chips_result.cells_with_coverage:,} with coverage"
    )
    if ctx.config.stages.chips.crop_stats:
        # chips and field_polygons_path share a CRS: chips derive from the same
        # (reprojected, class-filtered) fields used here, so no reprojection is needed.
        ctx.crop_stats_result = crop_stats.add_crop_stats(
            ctx.chips_path, ctx.field_polygons_path, on_progress=ctx.log
        )
    else:
        # Never publish a previous run's composition when the step is turned off.
        crop_stats.drop_crop_stats(ctx.chips_path)


def stage_splits(ctx: PipelineContext) -> None:
    """Assign train/val/test splits to the chips file."""
    _require(ctx.chips_path, stage="splits", produced_by="chips")
    split_cfg = ctx.config.stages.splits
    ctx.log(f"Assigning {split_cfg.split_type} splits...")
    ctx.splits_result = splits.assign_splits(
        chips_file=str(ctx.chips_path),
        split_type=split_cfg.split_type,
        split_percents=split_cfg.split_percents,
        random_seed=split_cfg.random_seed,
        fields_file=str(ctx.field_polygons_path),
        on_progress=ctx.log,
    )


def stage_boundaries(ctx: PipelineContext) -> None:
    """Create boundary lines from field polygons."""
    _require(ctx.field_polygons_path, stage="boundaries", produced_by=ctx.field_polygons_producer)
    ctx.log("Creating boundary lines...")
    result = boundaries.create_boundaries(
        input_path=str(ctx.field_polygons_path),
        output_dir=str(ctx.output_dir),
        output_prefix=f"{ctx.field_dataset}_boundary_lines_",
        on_progress=ctx.log,
    )
    # Rename to the canonical naming convention.
    if result.files_processed:
        original_output = result.files_processed[0].output_path
        if original_output != ctx.boundary_lines_path:
            shutil.move(str(original_output), str(ctx.boundary_lines_path))
    ctx.boundaries_result = result
    ctx.log(f"Created boundary lines: {result.total_features:,} features")


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
    min_coverage = ctx.config.stages.chips.min_coverage
    conn = duckdb.connect(":memory:")
    ensure_spatial_loaded(conn)
    grid_ids = conn.execute(
        f"SELECT id FROM '{ctx.chips_path}' WHERE field_coverage_pct >= {min_coverage}"
    ).fetchall()
    conn.close()

    chip_dirs: dict[str, Path] = {}
    for (grid_id,) in grid_ids:
        item_id = get_item_id(str(grid_id), ctx.effective_year)
        chip_dir = ctx.chips_base_dir / get_mgrs_square(str(grid_id)) / item_id
        chip_dir.mkdir(parents=True, exist_ok=True)
        chip_dirs[item_id] = chip_dir
    return chip_dirs


def stage_masks(ctx: PipelineContext) -> None:
    """Create the requested raster mask types for each chip."""
    _require(ctx.chips_path, stage="masks", produced_by="chips")
    _require(ctx.field_polygons_path, stage="masks", produced_by=ctx.field_polygons_producer)
    _require(ctx.boundary_lines_path, stage="masks", produced_by="boundaries")

    chip_dirs = _build_chip_dirs(ctx)
    ctx.log(f"Created {len(chip_dirs)} chip directories")

    masks_cfg = ctx.config.stages.masks
    requested = [
        (mask_type, subdir_name)
        for mask_type, subdir_name, type_name in _MASK_TYPE_MAPPING
        if type_name in masks_cfg.mask_types
    ]
    background_class_value = 3 if masks_cfg.presence_only else 0

    for mask_type, subdir_name in requested:
        ctx.log(f"Creating {mask_type.value} masks...")
        mask_result = masks.create_masks(
            chips_file=str(ctx.chips_path),
            boundaries_file=str(ctx.field_polygons_path),
            boundary_lines_file=str(ctx.boundary_lines_path),
            output_dir=str(ctx.chips_base_dir),
            field_dataset=ctx.field_dataset,
            mask_type=mask_type,
            min_coverage=ctx.config.stages.chips.min_coverage,
            resolution=masks_cfg.resolution,
            num_workers=masks_cfg.workers,
            chip_dirs=chip_dirs,
            year=ctx.effective_year,
            background_class_value=background_class_value,
            on_progress=ctx.on_mask_progress,
            on_start=ctx.on_mask_start,
        )
        ctx.masks_results[subdir_name] = mask_result
        ctx.log(f"Created {mask_result.total_created} {mask_type.value} masks")


def stage_stac(ctx: PipelineContext) -> None:
    """Generate the STAC static catalog, embedding provenance if available."""
    _require(ctx.chips_path, stage="stac", produced_by="chips")
    _require(ctx.output_fields_path, stage="stac", produced_by="reproject")
    _require(ctx.boundary_lines_path, stage="stac", produced_by="boundaries")

    ctx.log("Generating STAC catalog...")
    ctx.stac_result = stac.generate_stac_catalog(
        output_dir=ctx.output_dir,
        field_dataset=ctx.field_dataset,
        fields_file=ctx.output_fields_path,
        chips_file=ctx.chips_path,
        boundary_lines_file=ctx.boundary_lines_path,
        chips_base_dir=ctx.chips_base_dir,
        filtered_fields_file=(
            ctx.field_polygons_path if ctx.config.class_filter is not None else None
        ),
        year=ctx.effective_year,
        provenance=ctx.provenance,
        checksums=ctx.config.stages.stac.checksums,
        background_class_value=3 if ctx.config.stages.masks.presence_only else 0,
        on_progress=ctx.log,
        config=ctx.config,
    )
    n = ctx.stac_result.total_items
    k = len(ctx.stac_result.subcatalog_paths)
    ctx.log(f"Created STAC collection with {n} items in {k} sub-catalog(s)")


def stage_select_images(ctx: PipelineContext) -> None:
    """Select cloud-free Sentinel-2 scenes for each chip."""
    _require(ctx.output_dir / "collection.json", stage="select_images", produced_by="stac")
    if ctx.effective_year is None:
        raise ValueError("A year is required for image selection.")

    select_cfg = ctx.config.stages.select_images
    ctx.log("Selecting imagery...")
    ctx.selection_result = select_imagery_for_catalog(
        catalog_dir=ctx.output_dir,
        year=ctx.effective_year,
        cloud_cover_chip=select_cfg.cloud_cover_chip,
        nodata_max=select_cfg.nodata_max,
        buffer_days=select_cfg.buffer_days,
        num_buffer_expansions=select_cfg.num_buffer_expansions,
        buffer_expansion_size=select_cfg.buffer_expansion_size,
    )


def stage_download_images(ctx: PipelineContext) -> None:
    """Download selected imagery and generate thumbnails."""
    _require(
        ctx.output_dir / "collection.json",
        stage="download_images",
        produced_by="stac",
    )
    download_cfg = ctx.config.stages.download_images
    ctx.log("Downloading imagery...")
    ctx.download_result = download_imagery_for_catalog(
        catalog_dir=ctx.output_dir,
        bands=download_cfg.bands,
        resolution=download_cfg.resolution,
    )


TIPPECANOE_MISSING_WARNING = (
    "Warning: tippecanoe not found; no PMTiles or styles were written "
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
        ctx.log(TIPPECANOE_MISSING_WARNING)
        return {}

    built: dict[str, Path] = {}
    for key, attribute, out_name, spec in _TILE_TARGETS:
        source = getattr(ctx, attribute)
        built[key] = tiles.build_pmtiles(
            source, ctx.output_dir / out_name, spec, on_progress=ctx.log
        )
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
        chips_tiles=f"./{built['chips_tiles'].name}" if "chips_tiles" in built else None,
        fields_tiles=f"./{built['fields_tiles'].name}" if "fields_tiles" in built else None,
        on_progress=ctx.log,
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
        on_progress=ctx.log,
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
        ctx.log("Nothing to document: PMTiles are off and README/AGENTS are disabled")
        return

    ctx.log(
        f"Documented the collection: {len(built)} tile archive(s), "
        f"{len(style_results)} style(s), {len(written)} document(s)"
    )


_STAGE_FUNCS: dict[str, Callable[[PipelineContext], None]] = {
    "reproject": stage_reproject,
    "filter": stage_filter,
    "chips": stage_chips,
    "splits": stage_splits,
    "boundaries": stage_boundaries,
    "masks": stage_masks,
    "stac": stage_stac,
    "select_images": stage_select_images,
    "download_images": stage_download_images,
    "docs": stage_docs,
}
