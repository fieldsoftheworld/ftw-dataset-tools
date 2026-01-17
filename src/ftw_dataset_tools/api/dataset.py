"""Core API for creating complete training datasets from field boundaries."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from ftw_dataset_tools.api import boundaries, field_stats, masks, stac
from ftw_dataset_tools.api.geo import (
    detect_crs,
    detect_geometry_column,
    ensure_spatial_loaded,
    reproject,
)
from ftw_dataset_tools.api.masks import MaskType, get_item_id

if TYPE_CHECKING:
    from collections.abc import Callable


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
    min_coverage: float = 0.01,
    resolution: float = 10.0,
    num_workers: int | None = None,
    skip_reproject: bool = False,
    year: int | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_mask_progress: Callable[[int, int], None] | None = None,
    on_mask_start: Callable[[int, int], None] | None = None,
) -> CreateDatasetResult:
    """
    Create a complete training dataset from a fields file.

    This function orchestrates the full dataset creation pipeline:
    1. Reproject fields to EPSG:4326 if needed
    2. Create chips with field coverage statistics
    3. Create boundary lines from polygons
    4. Create all three mask types (instance, semantic_2class, semantic_3class)
    5. Generate STAC static catalog

    Args:
        fields_file: Path to input GeoParquet file with field polygons
        output_dir: Output directory for all generated files (default: ./dataset)
        field_dataset: Name for the dataset (default: input filename stem)
        min_coverage: Minimum coverage percentage to include grids (default: 0.01)
        resolution: Pixel resolution in meters for masks (default: 10.0)
        num_workers: Number of parallel workers for mask creation
        skip_reproject: If True, fail instead of reprojecting non-4326 inputs
        year: Year for temporal extent (required if fields lack determination_datetime)
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
    fields_path = Path(fields_file).resolve()
    out_dir = Path(output_dir).resolve()

    if not fields_path.exists():
        raise FileNotFoundError(f"Fields file not found: {fields_path}")

    # Default field_dataset to input filename stem
    if field_dataset is None:
        field_dataset = fields_path.stem

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    log(f"Creating dataset '{field_dataset}' from {fields_path.name}")

    # Early validation: Check temporal extent availability
    # This avoids processing the entire dataset only to fail at STAC generation
    log("Checking temporal extent availability...")
    datetime_col = stac.detect_datetime_column(fields_path)
    if datetime_col:
        log(f"Found '{datetime_col}' column for temporal extent")
    elif year is not None:
        log(f"Using --year {year} for temporal extent")
    else:
        raise ValueError(
            "Cannot determine temporal extent for STAC catalog. "
            "Either provide --year parameter or ensure fields file has "
            "'determination_datetime' column."
        )

    # Create output directory structure
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chips base directory (will contain per-grid subdirectories)
    chips_base_dir = out_dir / f"{field_dataset}-chips"
    chips_base_dir.mkdir(exist_ok=True)

    # Step 1: Check CRS and reproject if needed
    log("Checking CRS...")
    geom_col = detect_geometry_column(fields_path) or "geometry"
    crs_info = detect_crs(fields_path, geom_col)
    source_crs = str(crs_info)
    was_reprojected = False

    output_fields_path = out_dir / f"{field_dataset}_fields.parquet"

    if crs_info.authority_code is None or crs_info.authority_code.upper() != "EPSG:4326":
        if skip_reproject:
            raise ValueError(
                f"Input file has CRS '{crs_info}' but EPSG:4326 is required. "
                "Remove --skip-reproject to auto-reproject."
            )

        log(f"Reprojecting from {crs_info} to EPSG:4326...")
        reproject_result = reproject(
            fields_path,
            output_fields_path,
            target_crs="EPSG:4326",
            on_progress=log,
        )
        fields_path = reproject_result.output_path
        was_reprojected = True
        log(f"Reprojected to: {output_fields_path}")
    else:
        # Copy fields file to output directory
        log("CRS is already EPSG:4326, copying to output directory...")
        shutil.copy2(fields_path, output_fields_path)
        fields_path = output_fields_path

    # Step 2: Create chips
    log("Creating chips with field coverage statistics...")
    chips_path = out_dir / f"{field_dataset}_chips.parquet"

    chips_result = field_stats.add_field_stats(
        fields_file=str(fields_path),
        grid_file=None,  # Use default FTW grid from Source Coop
        output_file=str(chips_path),
        min_coverage=min_coverage,
        on_progress=log,
    )
    log(
        f"Created chips: {chips_result.total_cells:,} cells, {chips_result.cells_with_coverage:,} with coverage"
    )

    # Step 3: Create boundary lines
    log("Creating boundary lines...")
    boundary_lines_path = out_dir / f"{field_dataset}_boundary_lines.parquet"

    boundaries_result = boundaries.create_boundaries(
        input_path=str(fields_path),
        output_dir=str(out_dir),
        output_prefix=f"{field_dataset}_boundary_lines_",
        on_progress=log,
    )

    # Rename the output file to match our naming convention
    if boundaries_result.files_processed:
        original_output = boundaries_result.files_processed[0].output_path
        if original_output != boundary_lines_path:
            shutil.move(str(original_output), str(boundary_lines_path))

    log(f"Created boundary lines: {boundaries_result.total_features:,} features")

    # Step 4: Create masks for all three types
    masks_results: dict[str, masks.CreateMasksResult] = {}

    # Get list of grid IDs from chips file to create directories
    conn = duckdb.connect(":memory:")
    ensure_spatial_loaded(conn)
    grid_ids = conn.execute(f"""
        SELECT id FROM '{chips_path}'
        WHERE field_coverage_pct >= {min_coverage}
    """).fetchall()
    conn.close()

    # Create chip directories and build mapping
    # Directory names include year if provided (e.g., ftw-34UFF1628_2024)
    chip_dirs: dict[str, Path] = {}
    for (grid_id,) in grid_ids:
        grid_id_str = str(grid_id)
        item_id = get_item_id(grid_id_str, year)
        chip_dir = chips_base_dir / item_id
        chip_dir.mkdir(exist_ok=True)
        chip_dirs[item_id] = chip_dir

    log(f"Created {len(chip_dirs)} chip directories")

    mask_type_mapping = [
        (MaskType.INSTANCE, "instance"),
        (MaskType.SEMANTIC_2_CLASS, "semantic_2class"),
        (MaskType.SEMANTIC_3_CLASS, "semantic_3class"),
    ]

    for mask_type, subdir_name in mask_type_mapping:
        log(f"Creating {mask_type.value} masks...")

        mask_result = masks.create_masks(
            chips_file=str(chips_path),
            boundaries_file=str(fields_path),
            boundary_lines_file=str(boundary_lines_path),
            output_dir=str(chips_base_dir),  # Fallback, not used when chip_dirs provided
            field_dataset=field_dataset,
            mask_type=mask_type,
            min_coverage=min_coverage,
            resolution=resolution,
            num_workers=num_workers,
            chip_dirs=chip_dirs,
            year=year,
            on_progress=on_mask_progress,
            on_start=on_mask_start,
        )

        masks_results[subdir_name] = mask_result
        log(f"Created {mask_result.total_created} {mask_type.value} masks")

    # Step 5: Generate STAC catalog
    log("Generating STAC catalog...")
    stac_result = stac.generate_stac_catalog(
        output_dir=out_dir,
        field_dataset=field_dataset,
        fields_file=output_fields_path,
        chips_file=chips_path,
        boundary_lines_file=boundary_lines_path,
        chips_base_dir=chips_base_dir,
        year=year,
        on_progress=log,
    )
    log(f"Created STAC catalog with {stac_result.total_items} items")

    log("Dataset creation complete!")

    return CreateDatasetResult(
        output_dir=out_dir,
        field_dataset=field_dataset,
        fields_file=output_fields_path,
        chips_file=chips_path,
        boundary_lines_file=boundary_lines_path,
        chips_base_dir=chips_base_dir,
        was_reprojected=was_reprojected,
        source_crs=source_crs,
        chips_result=chips_result,
        boundaries_result=boundaries_result,
        masks_results=masks_results,
        stac_result=stac_result,
    )
