"""Core API for generating STAC static catalogs from dataset outputs."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import pystac
from pystac import Asset, Catalog, Collection, Extent, Item, SpatialExtent, TemporalExtent

from ftw_dataset_tools.api.geo import ensure_spatial_loaded
from ftw_dataset_tools.api.masks import MaskType

if TYPE_CHECKING:
    from collections.abc import Callable

# STAC version
STAC_VERSION = "1.0.0"

# Media types
MEDIA_TYPE_PARQUET = "application/vnd.apache.parquet"
MEDIA_TYPE_COG = "image/tiff; application=geotiff; profile=cloud-optimized"

__all__ = [
    "STACGenerationResult",
    "generate_stac_catalog",
    "get_temporal_extent_from_year",
]


@dataclass
class STACGenerationResult:
    """Result of STAC catalog generation."""

    catalog_path: Path
    source_collection_path: Path
    chips_collection_path: Path
    items_parquet_path: Path
    total_items: int
    temporal_extent: tuple[datetime, datetime]


@dataclass
class ChipInfo:
    """Information about a single chip for STAC item creation."""

    grid_id: str
    geometry: dict  # GeoJSON geometry
    bbox: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    properties: dict = field(default_factory=dict)


def detect_datetime_column(file_path: str | Path) -> str | None:
    """
    Check if file has determination_datetime column.

    Args:
        file_path: Path to parquet file

    Returns:
        Column name if found, None otherwise
    """
    conn = duckdb.connect(":memory:")
    try:
        schema = conn.execute(f"DESCRIBE SELECT * FROM '{file_path}'").fetchall()
        col_names = [row[0].lower() for row in schema]

        # Check for fiboa determination_datetime column
        if "determination_datetime" in col_names:
            return "determination_datetime"

        return None
    finally:
        conn.close()


def get_temporal_extent_from_data(
    file_path: str | Path,
    datetime_col: str = "determination_datetime",
) -> tuple[datetime, datetime]:
    """
    Extract min/max datetime from data column.

    Args:
        file_path: Path to parquet file
        datetime_col: Name of datetime column

    Returns:
        Tuple of (start_datetime, end_datetime)
    """
    conn = duckdb.connect(":memory:")
    try:
        result = conn.execute(f"""
            SELECT
                MIN("{datetime_col}") as min_dt,
                MAX("{datetime_col}") as max_dt
            FROM '{file_path}'
        """).fetchone()

        if result and result[0] and result[1]:
            min_dt = result[0]
            max_dt = result[1]

            # Ensure timezone aware
            if min_dt.tzinfo is None:
                min_dt = min_dt.replace(tzinfo=UTC)
            if max_dt.tzinfo is None:
                max_dt = max_dt.replace(tzinfo=UTC)

            return (min_dt, max_dt)

        raise ValueError(f"Could not extract datetime range from column '{datetime_col}'")
    finally:
        conn.close()


def get_temporal_extent_from_year(year: int) -> tuple[datetime, datetime]:
    """
    Create temporal extent spanning full year.

    Args:
        year: The year to create extent for

    Returns:
        Tuple of (start_datetime, end_datetime) for Jan 1 to Dec 31
    """
    start = datetime(year, 1, 1, 0, 0, 0, tzinfo=UTC)
    end = datetime(year, 12, 31, 23, 59, 59, tzinfo=UTC)
    return (start, end)


def _get_dataset_bounds(file_path: Path, geom_col: str = "geometry") -> list[float]:
    """Get overall bounding box from a parquet file."""
    conn = duckdb.connect(":memory:")
    ensure_spatial_loaded(conn)
    try:
        result = conn.execute(f"""
            SELECT
                MIN(ST_XMin("{geom_col}")) as xmin,
                MIN(ST_YMin("{geom_col}")) as ymin,
                MAX(ST_XMax("{geom_col}")) as xmax,
                MAX(ST_YMax("{geom_col}")) as ymax
            FROM '{file_path}'
        """).fetchone()

        if result:
            return [result[0], result[1], result[2], result[3]]
        return [-180, -90, 180, 90]
    finally:
        conn.close()


def _extract_chips_info(
    chips_file: Path,
    grid_id_col: str = "id",
    geom_col: str = "geometry",
) -> list[ChipInfo]:
    """
    Extract chip information from chips parquet file.

    Args:
        chips_file: Path to chips parquet file
        grid_id_col: Column name for grid ID
        geom_col: Column name for geometry

    Returns:
        List of ChipInfo objects
    """
    conn = duckdb.connect(":memory:")
    ensure_spatial_loaded(conn)
    try:
        # Get chip info with geometry as GeoJSON
        results = conn.execute(f"""
            SELECT
                "{grid_id_col}" as grid_id,
                ST_AsGeoJSON("{geom_col}") as geojson,
                ST_XMin("{geom_col}") as xmin,
                ST_YMin("{geom_col}") as ymin,
                ST_XMax("{geom_col}") as xmax,
                ST_YMax("{geom_col}") as ymax
            FROM '{chips_file}'
        """).fetchall()

        chips = []
        for row in results:
            grid_id, geojson, xmin, ymin, xmax, ymax = row
            geometry = json.loads(geojson)
            chips.append(
                ChipInfo(
                    grid_id=str(grid_id),
                    geometry=geometry,
                    bbox=(xmin, ymin, xmax, ymax),
                )
            )
        return chips
    finally:
        conn.close()


def _create_root_catalog(
    dataset_name: str,
    description: str | None = None,
) -> Catalog:
    """Create the root STAC catalog."""
    catalog = Catalog(
        id=dataset_name,
        description=description or f"FTW training dataset: {dataset_name}",
        title=f"{dataset_name} Dataset",
    )
    return catalog


def _create_source_collection(
    dataset_name: str,
    fields_file: Path,
    boundary_lines_file: Path,
    temporal_extent: tuple[datetime, datetime],
    spatial_extent: list[float],
) -> Collection:
    """
    Create 'source' collection with parquet assets.

    Args:
        dataset_name: Name of the dataset
        fields_file: Path to fields parquet file
        boundary_lines_file: Path to boundary lines parquet file
        temporal_extent: Tuple of (start, end) datetime
        spatial_extent: Bounding box [xmin, ymin, xmax, ymax]

    Returns:
        pystac Collection
    """
    collection = Collection(
        id=f"{dataset_name}-source",
        description=f"Source vector data for {dataset_name} dataset",
        title=f"{dataset_name} Source Data",
        extent=Extent(
            spatial=SpatialExtent(bboxes=[spatial_extent]),
            temporal=TemporalExtent(intervals=[[temporal_extent[0], temporal_extent[1]]]),
        ),
    )

    # Add assets with relative paths from source/ directory
    collection.add_asset(
        key="fields",
        asset=Asset(
            href=f"../{fields_file.name}",
            media_type=MEDIA_TYPE_PARQUET,
            title="Field boundary polygons",
            roles=["data"],
        ),
    )

    collection.add_asset(
        key="boundary_lines",
        asset=Asset(
            href=f"../{boundary_lines_file.name}",
            media_type=MEDIA_TYPE_PARQUET,
            title="Field boundary lines",
            roles=["data"],
        ),
    )

    return collection


def _create_chip_item(
    chip_info: ChipInfo,
    field_dataset: str,
    temporal_extent: tuple[datetime, datetime],
    chip_dir: Path | None = None,
    mask_dirs: dict[str, Path] | None = None,
) -> Item | None:
    """
    Create a STAC Item for a single chip.

    Args:
        chip_info: ChipInfo with geometry and bbox
        field_dataset: Dataset name
        temporal_extent: Tuple of (start, end) datetime
        chip_dir: Directory containing co-located masks (new structure)
        mask_dirs: Dict mapping mask type name to directory path (legacy structure)

    Returns:
        pystac Item, or None if no mask files exist
    """
    grid_id = chip_info.grid_id

    # Check which mask files exist
    mask_assets = {}
    mask_type_map = {
        "instance": MaskType.INSTANCE,
        "semantic_2class": MaskType.SEMANTIC_2_CLASS,
        "semantic_3class": MaskType.SEMANTIC_3_CLASS,
    }

    for mask_name, mask_type in mask_type_map.items():
        if chip_dir is not None:
            # New structure: masks co-located with item
            mask_filename = f"{grid_id}_{mask_type.value}.tif"
            mask_path = chip_dir / mask_filename
            if mask_path.exists():
                mask_assets[f"{mask_name}_mask"] = Asset(
                    href=f"./{mask_filename}",
                    media_type=MEDIA_TYPE_COG,
                    title=_get_mask_title(mask_name),
                    roles=["labels"],
                )
        elif mask_dirs is not None and mask_name in mask_dirs:
            # Legacy structure: masks in type-based directories
            mask_filename = f"{field_dataset}_{grid_id}_{mask_type.value}.tif"
            mask_path = mask_dirs[mask_name] / mask_filename
            if mask_path.exists():
                rel_path = f"../../label_masks/{mask_name}/{mask_filename}"
                mask_assets[f"{mask_name}_mask"] = Asset(
                    href=rel_path,
                    media_type=MEDIA_TYPE_COG,
                    title=_get_mask_title(mask_name),
                    roles=["labels"],
                )

    # Skip if no masks exist
    if not mask_assets:
        return None

    # Create item with datetime range
    item = Item(
        id=grid_id,
        geometry=chip_info.geometry,
        bbox=list(chip_info.bbox),
        datetime=None,  # Use start/end instead
        properties={
            "start_datetime": temporal_extent[0].isoformat(),
            "end_datetime": temporal_extent[1].isoformat(),
        },
    )

    # Add mask assets
    for key, asset in mask_assets.items():
        item.add_asset(key=key, asset=asset)

    return item


def _get_mask_title(mask_name: str) -> str:
    """Get human-readable title for mask type."""
    titles = {
        "instance": "Instance segmentation mask",
        "semantic_2class": "Binary semantic mask (field/background)",
        "semantic_3class": "3-class semantic mask (field/boundary/background)",
    }
    return titles.get(mask_name, f"{mask_name} mask")


def _create_chips_collection(
    dataset_name: str,
    chips_file: Path,
    items: list[Item],
    temporal_extent: tuple[datetime, datetime],
    spatial_extent: list[float],
) -> Collection:
    """
    Create 'chips' collection.

    Args:
        dataset_name: Name of the dataset
        chips_file: Path to chips parquet file
        items: List of STAC items
        temporal_extent: Tuple of (start, end) datetime
        spatial_extent: Bounding box [xmin, ymin, xmax, ymax]

    Returns:
        pystac Collection with items
    """
    collection = Collection(
        id=f"{dataset_name}-chips",
        description=f"Chip items with label masks for {dataset_name} dataset",
        title=f"{dataset_name} Chips",
        extent=Extent(
            spatial=SpatialExtent(bboxes=[spatial_extent]),
            temporal=TemporalExtent(intervals=[[temporal_extent[0], temporal_extent[1]]]),
        ),
    )

    # Add chips parquet as collection asset
    collection.add_asset(
        key="chips",
        asset=Asset(
            href=f"../{chips_file.name}",
            media_type=MEDIA_TYPE_PARQUET,
            title="Chip definitions with field coverage",
            roles=["data"],
        ),
    )

    # Add stac-geoparquet as collection asset
    collection.add_asset(
        key="items",
        asset=Asset(
            href="items.parquet",
            media_type="application/vnd.apache.parquet; profile=stac-geoparquet",
            title="STAC items in GeoParquet format",
            roles=["stac-items"],
        ),
    )

    # Add items to collection
    for item in items:
        collection.add_item(item)

    return collection


async def _write_items_parquet_async(
    items: list[Item],
    output_path: Path,
) -> None:
    """Write STAC items to stac-geoparquet format using rustac."""
    import rustac

    # Convert pystac Items to dicts
    item_dicts = [item.to_dict() for item in items]

    # Write using rustac async API
    await rustac.write(str(output_path), item_dicts)


def _write_items_parquet(
    items: list[Item],
    output_path: Path,
) -> None:
    """Synchronous wrapper for writing stac-geoparquet."""
    asyncio.run(_write_items_parquet_async(items, output_path))


def generate_stac_catalog(
    output_dir: Path | str,
    field_dataset: str,
    fields_file: Path | str,
    chips_file: Path | str,
    boundary_lines_file: Path | str,
    chips_base_dir: Path | None = None,
    mask_dirs: dict[str, Path] | None = None,
    year: int | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> STACGenerationResult:
    """
    Generate complete STAC static catalog from dataset outputs.

    Args:
        output_dir: Base directory for dataset and STAC output
        field_dataset: Dataset name (used in IDs and paths)
        fields_file: Path to fields parquet file
        chips_file: Path to chips parquet file
        boundary_lines_file: Path to boundary lines parquet file
        chips_base_dir: Base directory containing chip subdirectories with co-located masks.
                        If provided, expects structure: {chips_base_dir}/{grid_id}/{grid_id}_*.tif
        mask_dirs: Legacy - Dict mapping mask type name to directory path.
                   Ignored if chips_base_dir is provided.
        year: Optional year for temporal extent (required if no determination_datetime)
        on_progress: Optional callback for progress messages

    Returns:
        STACGenerationResult with paths to generated files

    Raises:
        ValueError: If year not provided and no determination_datetime column
    """
    output_dir = Path(output_dir)
    fields_file = Path(fields_file)
    chips_file = Path(chips_file)
    boundary_lines_file = Path(boundary_lines_file)

    # TODO: Task 7 will implement chips_base_dir logic
    _ = chips_base_dir  # Placeholder until Task 7 implementation

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    # Determine temporal extent
    log("Determining temporal extent...")
    datetime_col = detect_datetime_column(fields_file)

    if datetime_col:
        log(f"Using '{datetime_col}' column for temporal extent")
        temporal_extent = get_temporal_extent_from_data(fields_file, datetime_col)
    elif year is not None:
        log(f"Using year {year} for temporal extent")
        temporal_extent = get_temporal_extent_from_year(year)
    else:
        raise ValueError(
            "Cannot determine temporal extent. Either provide --year parameter "
            "or ensure fields file has 'determination_datetime' column."
        )

    # Get spatial extent from fields
    log("Calculating spatial extent...")
    spatial_extent = _get_dataset_bounds(fields_file)

    # Extract chip info
    log("Extracting chip information...")
    chip_infos = _extract_chips_info(chips_file)
    log(f"Found {len(chip_infos)} chips")

    # Create items for each chip
    log("Creating STAC items...")
    items = []
    for chip_info in chip_infos:
        item = _create_chip_item(
            chip_info=chip_info,
            field_dataset=field_dataset,
            temporal_extent=temporal_extent,
            mask_dirs=mask_dirs,
        )
        if item:
            items.append(item)

    log(f"Created {len(items)} items with mask assets")

    # Create root catalog
    log("Creating root catalog...")
    catalog = _create_root_catalog(field_dataset)

    # Create source collection
    log("Creating source collection...")
    source_collection = _create_source_collection(
        dataset_name=field_dataset,
        fields_file=fields_file,
        boundary_lines_file=boundary_lines_file,
        temporal_extent=temporal_extent,
        spatial_extent=spatial_extent,
    )

    # Create chips collection with items
    log("Creating chips collection...")
    chips_collection = _create_chips_collection(
        dataset_name=field_dataset,
        chips_file=chips_file,
        items=items,
        temporal_extent=temporal_extent,
        spatial_extent=spatial_extent,
    )

    # Add collections to catalog
    catalog.add_child(source_collection)
    catalog.add_child(chips_collection)

    # Normalize and save catalog
    log("Writing STAC catalog...")
    catalog.normalize_hrefs(str(output_dir))
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)

    # Get actual paths after normalization
    catalog_path = output_dir / "catalog.json"
    # pystac creates directories named after collection IDs
    source_collection_path = output_dir / f"{field_dataset}-source" / "collection.json"
    chips_collection_path = output_dir / f"{field_dataset}-chips" / "collection.json"
    items_parquet_path = output_dir / f"{field_dataset}-chips" / "items.parquet"

    # Write stac-geoparquet to the correct location
    if items:
        log("Writing stac-geoparquet...")
        _write_items_parquet(items, items_parquet_path)

    log("STAC catalog generation complete")

    return STACGenerationResult(
        catalog_path=catalog_path,
        source_collection_path=source_collection_path,
        chips_collection_path=chips_collection_path,
        items_parquet_path=items_parquet_path,
        total_items=len(items),
        temporal_extent=temporal_extent,
    )
