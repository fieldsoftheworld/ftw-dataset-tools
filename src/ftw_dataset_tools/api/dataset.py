"""Core API for creating complete training datasets from field boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ftw_dataset_tools.api import (
    boundaries,
    crop_stats,
    field_stats,
    masks,
    pipeline,
    splits,
    stac,
)
from ftw_dataset_tools.api.config import ClassFilter, DatasetConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@dataclass
class CreateDatasetResult:
    """Result of dataset creation operation."""

    output_dir: Path
    field_dataset: str
    fields_file: Path
    chips_file: Path
    boundary_lines_file: Path
    chips_base_dir: Path | None = None
    was_reprojected: bool = False
    source_crs: str | None = None
    chips_result: field_stats.FieldStatsResult | None = None
    splits_result: splits.CreateSplitsResult | None = None
    crop_stats_result: crop_stats.CropStatsResult | None = None
    boundaries_result: boundaries.CreateBoundariesResult | None = None
    masks_results: dict[str, masks.CreateMasksResult] = field(default_factory=dict)
    stac_result: stac.STACGenerationResult | None = None

    @property
    def total_masks_created(self) -> int:
        """Total number of masks created across all types."""
        return sum(r.total_created for r in self.masks_results.values())


def create_dataset(
    fields_file: str | Path,
    output_dir: str | Path = "./dataset",
    field_dataset: str | None = None,
    split_type: str | None = None,
    split_percents: tuple[int, int, int] = (80, 10, 10),
    min_coverage: float = 0.01,
    resolution: float = 10.0,
    num_workers: int | None = None,
    skip_reproject: bool = False,
    year: int | None = None,
    mask_types: list[str] | None = None,
    presence_only: bool = False,
    drop_border_chips: bool = False,
    checksums: bool = False,
    class_filter: str | Path | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_mask_progress: Callable[[int, int], None] | None = None,
    on_mask_start: Callable[[int, int], None] | None = None,
) -> CreateDatasetResult:
    """
    Create a complete training dataset from a fields file.

    This function orchestrates the full dataset creation pipeline:
    1. Reproject fields to EPSG:4326 if needed
    2. Create chips with field coverage statistics
    3. Create dataset splits based on the chosen strategy
    4. Create boundary lines from polygons
    5. Create all three mask types (instance, semantic_2class, semantic_3class)
    6. Generate STAC static catalog
    7. Document the collection: PMTiles and MapLibre styles when tippecanoe is
       installed, README.md and AGENTS.md, all registered on the collection

    Step 7 has no keyword argument here; configure or disable it through
    ``stages.docs`` in a config file with ``ftwd run``, or stop the run earlier
    with ``ftwd run --through stac``.

    Args:
        fields_file: Path to input GeoParquet file with field polygons
        output_dir: Output directory for all generated files (default: ./dataset)
        field_dataset: Name for the dataset (default: input filename stem)
        split_type: Dataset split strategy
        split_percents: Train/val/test percentages (must be three integers summing to 100; default: 80 10 10)
        min_coverage: Minimum coverage percentage to include grids (default: 0.01)
        resolution: Pixel resolution in meters for masks (default: 10.0)
        num_workers: Number of parallel workers for mask creation
        skip_reproject: If True, fail instead of reprojecting non-4326 inputs
        year: Year for temporal extent (required if fields lack determination_datetime)
        mask_types: List of mask types to generate (e.g., ["instance", "semantic_2_class"]). If None, generates all types.
        presence_only: If True, background class value is 3 instead of 0 (for presence-only labels)
        drop_border_chips: If True, remove chips touching outer boundary (edges of convex hull). Useful when fields at boundary may have partial coverage.
        checksums: Compute file:checksum for every asset (slow; default False).
        class_filter: Optional path to a class filter YAML (column + include/exclude
            lists). Include classes count as field; all others become background.
        on_progress: Optional callback for progress messages
        on_mask_progress: Optional callback (current, total) for mask creation progress
        on_mask_start: Optional callback (total_grids, filtered_grids) for mask start

    Returns:
        CreateDatasetResult with paths to all created files

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If input CRS is not EPSG:4326 and skip_reproject is True
        ValueError: If year not provided and no determination_datetime column
    """
    # Build a config from the keyword arguments so this flag-driven entry point
    # runs through the same stage engine as the config-driven ``ftwd run``.
    config = DatasetConfig.from_kwargs(
        fields_file=str(fields_file),
        output_dir=str(output_dir) if output_dir is not None else None,
        field_dataset=field_dataset,
        split_type=split_type,
        split_percents=split_percents,
        min_coverage=min_coverage,
        resolution=resolution,
        num_workers=num_workers,
        skip_reproject=skip_reproject,
        year=year,
        mask_types=mask_types,
        presence_only=presence_only,
        drop_border_chips=drop_border_chips,
    )
    config.stages.stac.checksums = checksums
    if class_filter is not None:
        config.stages.masks.class_filter = str(class_filter)
        config.class_filter = ClassFilter.from_file(class_filter)
    provenance = config.provenance_dict()

    ctx = pipeline.build_context(
        config,
        on_progress=on_progress,
        on_mask_progress=on_mask_progress,
        on_mask_start=on_mask_start,
        provenance=provenance,
    )
    # Imagery stages are disabled in from_kwargs(), so this runs reproject..stac, docs.
    stages = pipeline.resolve_stages(config=config)
    pipeline.run_pipeline(ctx, stages)

    return CreateDatasetResult(
        output_dir=ctx.output_dir,
        field_dataset=ctx.field_dataset,
        fields_file=ctx.output_fields_path,
        chips_file=ctx.chips_path,
        boundary_lines_file=ctx.boundary_lines_path,
        chips_base_dir=ctx.chips_base_dir,
        was_reprojected=ctx.was_reprojected,
        source_crs=ctx.source_crs,
        chips_result=ctx.chips_result,
        splits_result=ctx.splits_result,
        crop_stats_result=ctx.crop_stats_result,
        boundaries_result=ctx.boundaries_result,
        masks_results=ctx.masks_results,
        stac_result=ctx.stac_result,
    )
