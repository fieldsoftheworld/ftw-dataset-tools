"""Core API for creating raster masks from vector boundaries."""

from __future__ import annotations

import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import numpy as np
import rasterio
from rasterio import features
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from ftw_dataset_tools.api.geo import detect_geometry_column, ensure_spatial_loaded

if TYPE_CHECKING:
    from collections.abc import Callable


class MaskType(str, Enum):
    """Type of mask to create."""

    INSTANCE = "instance"
    SEMANTIC_2_CLASS = "semantic_2_class"
    SEMANTIC_3_CLASS = "semantic_3_class"


def get_mask_filename(grid_id: str, mask_type: MaskType) -> str:
    """
    Generate mask filename for a grid cell.

    Args:
        grid_id: The grid cell ID
        mask_type: Type of mask

    Returns:
        Filename like "{grid_id}_{mask_type.value}.tif"
    """
    return f"{grid_id}_{mask_type.value}.tif"


def get_mask_output_path(
    grid_id: str,
    mask_type: MaskType,
    chip_dirs: dict[str, Path] | None,
    output_dir: Path,
    field_dataset: str,
) -> Path:
    """
    Get the output path for a mask file.

    Args:
        grid_id: The grid cell ID
        mask_type: Type of mask
        chip_dirs: Optional dict mapping grid_id to chip directory
        output_dir: Fallback output directory
        field_dataset: Dataset name (used in filename when chip_dirs is None)

    Returns:
        Full path for the mask file
    """
    if chip_dirs is not None and grid_id in chip_dirs:
        # Co-located with STAC item: simple filename
        return chip_dirs[grid_id] / get_mask_filename(grid_id, mask_type)
    else:
        # Legacy: dataset prefix in filename
        return output_dir / f"{field_dataset}_{grid_id}_{mask_type.value}.tif"


@dataclass
class MaskResult:
    """Result of a single mask creation."""

    grid_id: str
    output_path: Path
    width: int
    height: int


@dataclass
class CreateMasksResult:
    """Result of mask creation operation."""

    masks_created: list[MaskResult]
    masks_skipped: list[tuple[str, str]]  # (grid_id, reason)
    field_dataset: str

    @property
    def total_created(self) -> int:
        """Total number of masks created."""
        return len(self.masks_created)

    @property
    def total_skipped(self) -> int:
        """Total number of masks skipped."""
        return len(self.masks_skipped)


def _get_geometries_in_bounds(
    conn: duckdb.DuckDBPyConnection,
    file_path: Path,
    geom_col: str,
    bounds: tuple[float, float, float, float],
    id_col: str | None = None,
) -> list[tuple]:
    """Get geometries from a parquet file that intersect the given bounds."""
    minx, miny, maxx, maxy = bounds

    # Build query
    if id_col:
        select_cols = f'"{id_col}" as id, ST_AsText("{geom_col}") as wkt'
    else:
        select_cols = f'ST_AsText("{geom_col}") as wkt'

    query = f"""
        SELECT {select_cols}
        FROM '{file_path}'
        WHERE ST_Intersects(
            "{geom_col}",
            ST_GeomFromText('POLYGON(({minx} {miny}, {maxx} {miny}, {maxx} {maxy}, {minx} {maxy}, {minx} {miny}))')
        )
    """

    return conn.execute(query).fetchall()


def _wkt_to_geometry(wkt: str):
    """Convert WKT to a shapely geometry."""
    from shapely import wkt as shapely_wkt

    return shapely_wkt.loads(wkt)


def _create_single_mask(
    conn: duckdb.DuckDBPyConnection,
    grid_id: str,
    bounds: tuple[float, float, float, float],
    crs: CRS,
    boundaries_path: Path,
    boundary_lines_path: Path,
    boundaries_geom_col: str,
    boundary_lines_geom_col: str,
    output_path: Path,
    mask_type: MaskType,
    resolution: float = 10.0,
    id_col: str | None = None,
) -> MaskResult:
    """Create a single mask for a grid cell."""
    minx, miny, maxx, maxy = bounds

    # Adjust resolution for geographic CRS (lat/long)
    # 10 meters is approximately 0.0001 degrees at the equator
    actual_resolution = resolution
    if crs.is_geographic:
        # Convert meters to approximate degrees (1 degree ~ 111,000 meters at equator)
        actual_resolution = resolution / 111000.0

    # Calculate dimensions based on resolution
    width = int((maxx - minx) / actual_resolution)
    height = int((maxy - miny) / actual_resolution)

    # Ensure minimum dimensions
    if width < 1 or height < 1:
        raise ValueError(
            f"Grid cell too small for resolution {resolution}m: calculated {width}x{height} pixels"
        )

    # Create transform
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Set data type based on mask type
    dtype = np.uint32 if mask_type == MaskType.INSTANCE else np.uint8

    # Initialize mask
    mask = np.zeros((height, width), dtype=dtype)

    # Get boundaries within bounds
    if mask_type == MaskType.INSTANCE and id_col:
        boundaries = _get_geometries_in_bounds(
            conn, boundaries_path, boundaries_geom_col, bounds, id_col=id_col
        )
        # Rasterize with ID values
        if boundaries:
            shapes = [(_wkt_to_geometry(wkt), int(id_val)) for id_val, wkt in boundaries]
            features.rasterize(shapes, out=mask, transform=transform, all_touched=False)
    else:
        boundaries = _get_geometries_in_bounds(conn, boundaries_path, boundaries_geom_col, bounds)
        # Rasterize with value 1
        if boundaries:
            shapes = [(_wkt_to_geometry(wkt), 1) for (wkt,) in boundaries]
            features.rasterize(shapes, out=mask, transform=transform, all_touched=False)

    # Get boundary lines
    boundary_lines = _get_geometries_in_bounds(
        conn, boundary_lines_path, boundary_lines_geom_col, bounds
    )

    if boundary_lines:
        if mask_type == MaskType.SEMANTIC_3_CLASS:
            # Burn boundary lines as class 2
            features.rasterize(
                [(_wkt_to_geometry(wkt), 2) for (wkt,) in boundary_lines],
                out=mask,
                transform=transform,
                all_touched=True,
            )
        else:
            # Burn boundary lines as 0 (background)
            features.rasterize(
                [(_wkt_to_geometry(wkt), 0) for (wkt,) in boundary_lines],
                out=mask,
                transform=transform,
                all_touched=True,
            )

    # Write as GeoTIFF first
    temp_path = output_path.with_suffix(".tmp.tif")
    with rasterio.open(
        temp_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=0,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
    ) as dst:
        dst.write(mask, 1)

    # Convert to COG
    _convert_to_cog(temp_path, output_path)
    temp_path.unlink()

    return MaskResult(
        grid_id=grid_id,
        output_path=output_path,
        width=width,
        height=height,
    )


def _convert_to_cog(input_path: Path, output_path: Path) -> None:
    """Convert a GeoTIFF to Cloud Optimized GeoTIFF (COG)."""
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(
            driver="COG",
            compress="deflate",
        )

        # Read data
        data = src.read()

        # Write as COG
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)


def _process_single_grid_cell(args: tuple) -> tuple[MaskResult | None, tuple[str, str] | None]:
    """
    Worker function to process a single grid cell.

    This function is designed to be called from a ProcessPoolExecutor.
    It creates its own DuckDB connection since connections can't be shared across processes.

    Args:
        args: Tuple of (grid_id, bounds, crs_wkt, boundaries_path, boundary_lines_path,
              boundaries_geom_col, boundary_lines_geom_col, output_path, mask_type,
              resolution, id_col)

    Returns:
        Tuple of (MaskResult or None, error tuple or None)
    """
    (
        grid_id,
        bounds,
        crs_wkt,
        boundaries_path,
        boundary_lines_path,
        boundaries_geom_col,
        boundary_lines_geom_col,
        output_path,
        mask_type,
        resolution,
        id_col,
    ) = args

    # Suppress stdout/stderr from GDAL/rasterio progress output at OS level
    # (GDAL writes to C file descriptors, not Python's sys.stdout/stderr)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)

        # Convert string back to enum and CRS
        mask_type_enum = MaskType(mask_type)
        crs = CRS.from_wkt(crs_wkt)

        # Create DuckDB connection for this process
        conn = duckdb.connect(":memory:")
        ensure_spatial_loaded(conn)

        try:
            result = _create_single_mask(
                conn=conn,
                grid_id=grid_id,
                bounds=bounds,
                crs=crs,
                boundaries_path=Path(boundaries_path),
                boundary_lines_path=Path(boundary_lines_path),
                boundaries_geom_col=boundaries_geom_col,
                boundary_lines_geom_col=boundary_lines_geom_col,
                output_path=Path(output_path),
                mask_type=mask_type_enum,
                resolution=resolution,
                id_col=id_col,
            )
            conn.close()
            return (result, None)
        except Exception as e:
            conn.close()
            return (None, (grid_id, str(e)))
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        os.close(devnull_fd)


def create_masks(
    chips_file: str | Path,
    boundaries_file: str | Path,
    boundary_lines_file: str | Path,
    output_dir: str | Path = "./masks",
    field_dataset: str = "unknown",
    grid_id_col: str = "id",
    mask_type: MaskType = MaskType.SEMANTIC_2_CLASS,
    coverage_col: str = "field_coverage_pct",
    min_coverage: float = 0.01,
    resolution: float = 10.0,
    num_workers: int | None = None,
    chip_dirs: dict[str, Path] | None = None,  # noqa: ARG001 - used in Task 3
    on_progress: Callable[[int, int], None] | None = None,
    on_start: Callable[[int, int], None] | None = None,
) -> CreateMasksResult:
    """
    Create raster masks from vector boundaries for each grid cell.

    Args:
        chips_file: Path to chips GeoParquet file (from create-chips)
        boundaries_file: Path to boundaries GeoParquet file (polygons)
        boundary_lines_file: Path to boundary lines GeoParquet file
        output_dir: Output directory for masks (default: ./masks)
        field_dataset: Name of the field dataset (used in output filenames)
        grid_id_col: Column name for grid cell ID (default: "id")
        mask_type: Type of mask to create (default: semantic_2_class)
        coverage_col: Column name for field coverage percentage (to filter grids)
        min_coverage: Minimum coverage percentage to process (default: 0.01)
        resolution: Pixel resolution in CRS units (default: 10.0 meters)
        num_workers: Number of parallel workers (default: number of CPUs)
        chip_dirs: Optional dict mapping grid_id to output directory.
                   If provided, masks are written to chip-specific directories.
                   If None, all masks go to output_dir with dataset prefix in filename.
        on_progress: Optional callback (current, total) for progress updates
        on_start: Optional callback (total_grids, filtered_grids) called before processing

    Returns:
        CreateMasksResult with information about created and skipped masks

    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If required columns are missing
    """
    chips_path = Path(chips_file).resolve()
    boundaries_path = Path(boundaries_file).resolve()
    boundary_lines_path = Path(boundary_lines_file).resolve()
    output_path = Path(output_dir).resolve()

    for path, name in [
        (chips_path, "Chips file"),
        (boundaries_path, "Boundaries file"),
        (boundary_lines_path, "Boundary lines file"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Detect geometry columns
    grid_geom_col = detect_geometry_column(chips_path) or "geometry"
    boundaries_geom_col = detect_geometry_column(boundaries_path) or "geometry"
    boundary_lines_geom_col = detect_geometry_column(boundary_lines_path) or "geometry"

    # Create DuckDB connection
    conn = duckdb.connect(":memory:")
    ensure_spatial_loaded(conn)

    # Get total grid count (before filtering)
    total_count_result = conn.execute(f"SELECT COUNT(*) FROM '{chips_path}'").fetchone()
    total_grids = total_count_result[0] if total_count_result else 0

    # Build query with optional coverage filter
    coverage_filter = ""
    if coverage_col:
        coverage_filter = f'WHERE "{coverage_col}" >= {min_coverage}'

    # Get grid cells with bounds
    query = f"""
        SELECT
            "{grid_id_col}" as grid_id,
            ST_XMin("{grid_geom_col}") as minx,
            ST_YMin("{grid_geom_col}") as miny,
            ST_XMax("{grid_geom_col}") as maxx,
            ST_YMax("{grid_geom_col}") as maxy
        FROM '{chips_path}'
        {coverage_filter}
    """

    try:
        grid_cells = conn.execute(query).fetchall()
    except duckdb.BinderException as e:
        if grid_id_col in str(e):
            raise ValueError(f"Grid ID column '{grid_id_col}' not found in grid file") from e
        if coverage_col and coverage_col in str(e):
            raise ValueError(f"Coverage column '{coverage_col}' not found in grid file") from e
        raise

    total_cells = len(grid_cells)

    # Call on_start callback with grid counts
    if on_start:
        on_start(total_grids, total_cells)

    # Get CRS from GeoParquet metadata
    geo_meta_result = conn.execute(
        "SELECT value FROM parquet_kv_metadata(?) WHERE key = 'geo'", [str(chips_path)]
    ).fetchone()

    crs = None
    if geo_meta_result:
        geo_meta = json.loads(geo_meta_result[0])
        columns = geo_meta.get("columns", {})
        geom_info = columns.get(grid_geom_col, {})
        crs_info = geom_info.get("crs")
        if crs_info and isinstance(crs_info, dict):
            # Convert PROJJSON to CRS
            crs = CRS.from_user_input(crs_info)
        elif crs_info is None:
            # GeoParquet spec: missing CRS means WGS84
            crs = CRS.from_epsg(4326)

    if not crs:
        crs = CRS.from_epsg(4326)

    # Determine ID column for instance masks
    id_col_for_instance = None
    if mask_type == MaskType.INSTANCE:
        # Try to find an ID column in boundaries file
        try:
            schema = conn.execute(f"DESCRIBE SELECT * FROM '{boundaries_path}'").fetchall()
            col_names = [row[0] for row in schema]
            for candidate in ["id", "ID", "fid", "FID", "objectid", "OBJECTID"]:
                if candidate in col_names:
                    id_col_for_instance = candidate
                    break
        except Exception:
            pass

    # Close the main connection - workers will create their own
    conn.close()

    # Determine number of workers (default: half of CPUs, minimum 1)
    if num_workers is None:
        cpu_count = multiprocessing.cpu_count()
        num_workers = max(1, cpu_count // 2)

    # Convert CRS to WKT for serialization
    crs_wkt = crs.to_wkt()

    # Prepare arguments for parallel processing
    work_items = []
    for grid_id, minx, miny, maxx, maxy in grid_cells:
        grid_id_str = str(grid_id)
        mask_filename = f"{field_dataset}_{grid_id_str}_{mask_type.value}.tif"
        mask_path = output_path / mask_filename

        work_items.append(
            (
                grid_id_str,
                (minx, miny, maxx, maxy),
                crs_wkt,
                str(boundaries_path),
                str(boundary_lines_path),
                boundaries_geom_col,
                boundary_lines_geom_col,
                str(mask_path),
                mask_type.value,  # Pass as string for serialization
                resolution,
                id_col_for_instance,
            )
        )

    # Process in parallel
    created: list[MaskResult] = []
    skipped: list[tuple[str, str]] = []
    completed = 0
    executor = None

    try:
        executor = ProcessPoolExecutor(max_workers=num_workers)
        # Submit all tasks
        futures = {executor.submit(_process_single_grid_cell, item): item[0] for item in work_items}

        # Process results as they complete
        for future in as_completed(futures):
            completed += 1
            if on_progress:
                on_progress(completed, total_cells)

            try:
                result, error = future.result()
                if result:
                    created.append(result)
                elif error:
                    skipped.append(error)
            except Exception as e:
                grid_id = futures[future]
                skipped.append((grid_id, str(e)))

    except KeyboardInterrupt:
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if executor:
            executor.shutdown(wait=True)

    return CreateMasksResult(
        masks_created=created,
        masks_skipped=skipped,
        field_dataset=field_dataset,
    )
