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
from pystac import (
    Asset,
    Catalog,
    Collection,
    Extent,
    Item,
    ItemAssetDefinition,
    Link,
    Provider,
    ProviderRole,
    SpatialExtent,
    TemporalExtent,
)
from pystac.extensions.version import VersionExtension
from pystac.layout import TemplateLayoutStrategy

from ftw_dataset_tools.api.assets import (
    add_file_info,
    add_mask_classification,
    add_raster_bands,
    add_table_columns,
)
from ftw_dataset_tools.api.geo import ensure_spatial_loaded
from ftw_dataset_tools.api.masks import MaskType, get_mgrs_square

if TYPE_CHECKING:
    from collections.abc import Callable

    from ftw_dataset_tools.api.config import DatasetConfig, MetadataConfig

# Media types
MEDIA_TYPE_PARQUET = "application/vnd.apache.parquet"
MEDIA_TYPE_COG = "image/tiff; application=geotiff; profile=cloud-optimized"
MEDIA_TYPE_JPEG = "image/jpeg"

# Layout strategy for the single collection: sub-catalogs per MGRS square, items
# co-located with their assets inside the square's sub-catalog directory.
CHIP_LAYOUT = TemplateLayoutStrategy(
    catalog_template="chips/${id}/catalog.json", item_template="${id}/${id}.json"
)

__all__ = [
    "STACGenerationResult",
    "generate_stac_catalog",
    "get_temporal_extent_from_year",
    "get_year_from_datetime_column",
]

# Registry of mask asset names. Module-level so the drift-guard tests can
# assert it stays in sync with the other mask-type registries: a missing entry
# here silently drops that mask from the STAC items.
_MASK_TYPE_BY_ASSET_NAME = {
    "instance": MaskType.INSTANCE,
    "semantic_2class": MaskType.SEMANTIC_2_CLASS,
    "semantic_3class": MaskType.SEMANTIC_3_CLASS,
    "decode_boundary": MaskType.DECODE_BOUNDARY,
    "decode_distance": MaskType.DECODE_DISTANCE,
}

_MASK_TITLES = {
    "instance": "Instance segmentation mask",
    "semantic_2class": "Binary semantic mask (field/background)",
    "semantic_3class": "3-class semantic mask (field/boundary/background)",
    "decode_boundary": "DECODE field boundary mask",
    "decode_distance": "DECODE normalized distance-to-boundary map",
}


@dataclass
class STACGenerationResult:
    """Result of STAC collection generation."""

    collection_path: Path
    items_parquet_path: Path | None
    subcatalog_paths: dict[str, Path]
    total_items: int
    temporal_extent: tuple[datetime, datetime]


@dataclass
class ChipInfo:
    """Information about a single chip for STAC item creation."""

    grid_id: str
    geometry: dict  # GeoJSON geometry
    bbox: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
    year: int | None = None  # Optional year for year-based naming
    properties: dict = field(default_factory=dict)

    @property
    def item_id(self) -> str:
        """Get the STAC item ID, including year if set."""
        if self.year is not None:
            return f"{self.grid_id}_{self.year}"
        return self.grid_id

    @property
    def dir_name(self) -> str:
        """Get the directory name for this chip, including year if set."""
        return self.item_id


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


def get_year_from_datetime_column(
    file_path: str | Path,
    datetime_col: str = "determination_datetime",
) -> int | None:
    """
    Extract year from datetime column using the most common year.

    This is useful for image selection where we need a single year
    to query crop calendar and STAC catalogs.

    Args:
        file_path: Path to parquet file
        datetime_col: Name of datetime column

    Returns:
        Most common year in the column, or None if extraction fails
    """
    conn = duckdb.connect(":memory:")
    try:
        # Get the most common year (mode) from the datetime column
        result = conn.execute(f"""
            SELECT EXTRACT(YEAR FROM "{datetime_col}") as year, COUNT(*) as cnt
            FROM '{file_path}'
            WHERE "{datetime_col}" IS NOT NULL
            GROUP BY year
            ORDER BY cnt DESC
            LIMIT 1
        """).fetchone()

        if result and result[0]:
            return int(result[0])

        return None
    except Exception:
        return None
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


#: Chips columns that, when present, are copied onto item properties (NULLs omitted).
OPTIONAL_CHIP_COLUMNS = {
    "split": "ftw:split",
    "field_coverage_pct": "ftw:field_coverage_pct",
    "hcat_dominant_code": "ftw:hcat_dominant_code",
    "hcat_dominant_name_en": "ftw:hcat_dominant_name_en",
    "hcat_dominant_pct": "ftw:hcat_dominant_pct",
    "hcat_top": "ftw:hcat_top",
}

#: Columns that need an explicit cast (test fixtures built with geopandas can store
#: integer/float columns containing None as float64; DuckDB otherwise returns those
#: as-is, which would make item JSON carry a float instead of an int).
_OPTIONAL_COLUMN_CASTS = {
    "hcat_dominant_code": "BIGINT",
    "hcat_dominant_pct": "DOUBLE",
    "field_coverage_pct": "DOUBLE",
}


def _optional_column_select(col: str) -> str:
    """SQL select expression for one optional chip column, casting where needed."""
    cast = _OPTIONAL_COLUMN_CASTS.get(col)
    if cast:
        return f'CAST("{col}" AS {cast}) AS "{col}"'
    return f'"{col}"'


def _extract_chips_info(
    chips_file: Path,
    grid_id_col: str = "id",
    geom_col: str = "geometry",
    year: int | None = None,
) -> list[ChipInfo]:
    """
    Extract chip information from chips parquet file.

    Args:
        chips_file: Path to chips parquet file
        grid_id_col: Column name for grid ID
        geom_col: Column name for geometry
        year: Optional year for year-based naming convention

    Returns:
        List of ChipInfo objects
    """
    conn = duckdb.connect(":memory:")
    ensure_spatial_loaded(conn)
    try:
        existing_cols = {
            row[0] for row in conn.execute(f"DESCRIBE SELECT * FROM '{chips_file}'").fetchall()
        }
        optional_cols = [col for col in OPTIONAL_CHIP_COLUMNS if col in existing_cols]
        optional_select = "".join(f", {_optional_column_select(col)}" for col in optional_cols)

        # Get chip info with geometry as GeoJSON
        results = conn.execute(f"""
            SELECT
                "{grid_id_col}" as grid_id,
                ST_AsGeoJSON("{geom_col}") as geojson,
                ST_XMin("{geom_col}") as xmin,
                ST_YMin("{geom_col}") as ymin,
                ST_XMax("{geom_col}") as xmax,
                ST_YMax("{geom_col}") as ymax
                {optional_select}
            FROM '{chips_file}'
        """).fetchall()

        chips = []
        for row in results:
            grid_id, geojson, xmin, ymin, xmax, ymax, *optional_values = row
            geometry = json.loads(geojson)
            properties = {
                OPTIONAL_CHIP_COLUMNS[col]: value
                for col, value in zip(optional_cols, optional_values, strict=True)
                if value is not None
            }
            chips.append(
                ChipInfo(
                    grid_id=str(grid_id),
                    geometry=geometry,
                    bbox=(xmin, ymin, xmax, ymax),
                    year=year,
                    properties=properties,
                )
            )
        return chips
    finally:
        conn.close()


def _updated_stamp(provenance: dict | None) -> str:
    """ISO 8601 UTC 'updated' value: the build's generated_at, else now (trailing Z)."""
    raw = (provenance or {}).get("generated_at")
    stamp = datetime.fromisoformat(raw) if raw else datetime.now(UTC)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _apply_collection_metadata(
    collection: Collection,
    metadata: MetadataConfig | None,
    *,
    updated: str,
    title: str | None = None,
    description: str | None = None,
) -> None:
    """Stamp config metadata, the version extension and 'updated' on a collection."""
    collection.extra_fields["updated"] = updated
    if metadata is None:
        return
    if title:
        collection.title = title
    if description:
        collection.description = description
    if metadata.license:
        collection.license = metadata.license
    if metadata.license_url:
        collection.add_link(Link(rel="license", target=metadata.license_url, title="License"))
    if metadata.keywords:
        collection.keywords = list(metadata.keywords)
    if metadata.providers:
        collection.providers = [
            Provider(name=p.name, roles=[ProviderRole(r) for r in p.roles], url=p.url)
            for p in metadata.providers
        ]
    if metadata.version:
        VersionExtension.ext(collection, add_if_missing=True).version = metadata.version


def _collection_ftw_properties(config: DatasetConfig) -> dict:
    """Build settings a consumer needs to interpret the chips, as ftw: fields."""
    stages = config.stages
    props: dict = {
        "ftw:split_type": stages.splits.split_type,
        "ftw:split_seed": stages.splits.random_seed,
        "ftw:split_percents": list(stages.splits.split_percents),
        "ftw:mask_types": list(stages.masks.mask_types),
        "ftw:mask_resolution_m": stages.masks.resolution,
        "ftw:presence_only": stages.masks.presence_only,
        "ftw:min_coverage_pct": stages.chips.min_coverage,
    }
    if stages.select_images.enabled:
        sel = stages.select_images
        props.update(
            {
                "ftw:cloud_cover_chip_threshold": sel.cloud_cover_chip,
                "ftw:nodata_max": sel.nodata_max,
                "ftw:buffer_days": sel.buffer_days,
                "ftw:num_buffer_expansions": sel.num_buffer_expansions,
                "ftw:buffer_expansion_size": sel.buffer_expansion_size,
            }
        )
    return {k: v for k, v in props.items() if v is not None}


def _group_items_by_square(items: list[Item], squares: dict[str, str]) -> dict[str, list[Item]]:
    """Group chip items by MGRS 100 km square, keyed by the square id, sorted.

    Args:
        items: Chip items to group.
        squares: Mapping of item id to MGRS square, as derived when the items were
            created (so grouping agrees with where each item's directory lives).
    """
    groups: dict[str, list[Item]] = {}
    for item in items:
        groups.setdefault(squares.get(item.id, "other"), []).append(item)
    return {square: groups[square] for square in sorted(groups)}


def _build_item_assets() -> dict[str, ItemAssetDefinition]:
    """Declare the assets a chip item may carry (core ``item_assets`` collection field)."""
    defs: dict[str, ItemAssetDefinition] = {}
    for mask_name in _MASK_TYPE_BY_ASSET_NAME:
        defs[f"{mask_name}_mask"] = ItemAssetDefinition(
            {"type": MEDIA_TYPE_COG, "roles": ["labels"], "title": _MASK_TITLES[mask_name]}
        )
    for season in ("planting", "harvest"):
        defs[f"{season}_image"] = ItemAssetDefinition(
            {
                "type": MEDIA_TYPE_COG,
                "roles": ["data"],
                "title": f"{season.capitalize()} season imagery",
            }
        )
    defs["thumbnail"] = ItemAssetDefinition(
        {"type": MEDIA_TYPE_JPEG, "roles": ["thumbnail"], "title": "Chip preview"}
    )
    return defs


def _add_parquet_asset(
    collection: Collection,
    key: str,
    path: Path,
    title: str,
    *,
    checksums: bool = False,
    roles: list[str] | None = None,
) -> None:
    """Add a GeoParquet asset to a collection, with file info and column stats."""
    collection.add_asset(
        key=key,
        asset=Asset(
            href=f"./{path.name}",
            media_type=MEDIA_TYPE_PARQUET,
            title=title,
            roles=roles if roles is not None else ["data"],
        ),
    )
    add_file_info(collection.assets[key], path, checksum=checksums)
    add_table_columns(collection.assets[key], path)


def _create_collection(
    dataset_name: str,
    fields_file: Path,
    boundary_lines_file: Path,
    chips_file: Path,
    temporal_extent: tuple[datetime, datetime],
    spatial_extent: list[float],
    *,
    filtered_fields_file: Path | None = None,
    checksums: bool = False,
) -> Collection:
    """
    Create the single dataset collection with source-data and chip-definition assets.

    Args:
        dataset_name: Name of the dataset (used as the collection id)
        fields_file: Path to fields parquet file
        boundary_lines_file: Path to boundary lines parquet file
        chips_file: Path to chips parquet file
        temporal_extent: Tuple of (start, end) datetime
        spatial_extent: Bounding box [xmin, ymin, xmax, ymax]
        filtered_fields_file: Optional path to the class-filtered fields parquet file
        checksums: Compute file:checksum (multihash sha256) for every asset.

    Returns:
        pystac Collection
    """
    collection = Collection(
        id=dataset_name,
        description=f"Benchmark chips with label masks for {dataset_name}",
        title=dataset_name,
        extent=Extent(
            spatial=SpatialExtent(bboxes=[spatial_extent]),
            temporal=TemporalExtent(intervals=[[temporal_extent[0], temporal_extent[1]]]),
        ),
    )

    _add_parquet_asset(
        collection, "fields", fields_file, "Field boundary polygons", checksums=checksums
    )

    if filtered_fields_file is not None:
        _add_parquet_asset(
            collection,
            "fields_filtered",
            filtered_fields_file,
            "Field polygons after the class filter",
            checksums=checksums,
        )

    _add_parquet_asset(
        collection,
        "boundary_lines",
        boundary_lines_file,
        "Field boundary lines",
        checksums=checksums,
    )

    _add_parquet_asset(
        collection,
        "chips",
        chips_file,
        "Chip definitions with field coverage",
        checksums=checksums,
    )

    collection.item_assets = _build_item_assets()

    return collection


def _create_chip_item(
    chip_info: ChipInfo,
    temporal_extent: tuple[datetime, datetime],
    chip_dir: Path,
    checksums: bool = False,
    background_class_value: int = 0,
) -> Item | None:
    """
    Create a STAC Item for a single chip.

    Args:
        chip_info: ChipInfo with geometry and bbox (includes optional year)
        temporal_extent: Tuple of (start, end) datetime
        chip_dir: Directory containing co-located masks
        checksums: Compute file:checksum (multihash sha256) for every mask asset.
        background_class_value: Pixel value used for background in masks
            (3 for presence-only).

    Returns:
        pystac Item, or None if no mask files exist
    """
    grid_id = chip_info.grid_id
    item_id = chip_info.item_id  # Includes year if set
    year = chip_info.year

    # Check which mask files exist (masks co-located with the item)
    mask_assets = {}
    for mask_name, mask_type in _MASK_TYPE_BY_ASSET_NAME.items():
        # Filename includes year if year is set
        if year is not None:
            mask_filename = f"{grid_id}_{year}_{mask_type.value}.tif"
        else:
            mask_filename = f"{grid_id}_{mask_type.value}.tif"
        mask_path = chip_dir / mask_filename
        if mask_path.exists():
            mask_assets[f"{mask_name}_mask"] = (
                Asset(
                    href=f"./{mask_filename}",
                    media_type=MEDIA_TYPE_COG,
                    title=_get_mask_title(mask_name),
                    roles=["labels"],
                ),
                mask_path,
            )

    # Skip if no masks exist
    if not mask_assets:
        return None

    # Build properties
    properties = {
        "start_datetime": temporal_extent[0].isoformat(),
        "end_datetime": temporal_extent[1].isoformat(),
    }

    # Add FTW extension properties if year is set
    if year is not None:
        properties["ftw:calendar_year"] = year

    # Merge in split, coverage and crop composition properties, if the chips file had them
    properties.update(chip_info.properties)

    # Create item with datetime range
    item = Item(
        id=item_id,  # Use item_id which includes year if set
        geometry=chip_info.geometry,
        bbox=list(chip_info.bbox),
        datetime=None,  # Use start/end instead
        properties=properties,
    )

    # Add mask assets, then decorate them (owner must be set first)
    for key, (asset, mask_path) in mask_assets.items():
        item.add_asset(key=key, asset=asset)
        add_file_info(asset, mask_path, checksum=checksums)
        add_raster_bands(asset, mask_path)
        add_mask_classification(
            asset, key.removesuffix("_mask"), background_value=background_class_value
        )

    return item


def _get_mask_title(mask_name: str) -> str:
    """Get human-readable title for mask type."""
    return _MASK_TITLES.get(mask_name, f"{mask_name} mask")


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
    chips_base_dir: Path | str,
    *,
    filtered_fields_file: Path | None = None,
    year: int | None = None,
    provenance: dict | None = None,
    config: DatasetConfig | None = None,
    on_progress: Callable[[str], None] | None = None,
    checksums: bool = False,
    background_class_value: int = 0,
) -> STACGenerationResult:
    """
    Generate a single self-contained STAC collection from dataset outputs.

    The output is one collection at ``output_dir/collection.json`` whose children
    are per-MGRS-square sub-catalogs holding the chip items (custom, non-FTW grid
    ids are grouped under an ``other`` sub-catalog).

    Args:
        output_dir: Base directory for dataset and STAC output
        field_dataset: Dataset name (used as the collection id)
        fields_file: Path to fields parquet file
        chips_file: Path to chips parquet file
        boundary_lines_file: Path to boundary lines parquet file
        chips_base_dir: Base directory containing chip subdirectories with co-located
                        masks, nested by MGRS square:
                        {chips_base_dir}/{square}/{item_id}/{item_id}_*.tif
        filtered_fields_file: Optional path to the class-filtered fields parquet file;
            when given, a ``fields_filtered`` asset is added to the collection.
        year: Optional year for temporal extent (required if no determination_datetime)
        provenance: Optional resolved-config record embedded on the collection under
                    the ``ftw:config`` extra field for reproducibility.
        config: Resolved dataset config; supplies metadata and the ftw: build properties
            written on the collection.
        on_progress: Optional callback for progress messages
        checksums: Compute file:checksum (multihash sha256) for every asset. Slow; default False.
        background_class_value: Pixel value used for background in masks (3 for presence-only).

    Returns:
        STACGenerationResult with paths to generated files

    Raises:
        ValueError: If year not provided and no determination_datetime column
    """
    output_dir = Path(output_dir)
    fields_file = Path(fields_file)
    chips_file = Path(chips_file)
    boundary_lines_file = Path(boundary_lines_file)
    chips_base_dir = Path(chips_base_dir)
    filtered_fields_file = Path(filtered_fields_file) if filtered_fields_file else None

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

    # Extract chip info (pass year for year-based naming)
    log("Extracting chip information...")
    chip_infos = _extract_chips_info(chips_file, year=year)
    log(f"Found {len(chip_infos)} chips")

    # Create items for each chip, nested by MGRS square
    log("Creating STAC items...")
    items = []
    item_squares: dict[str, str] = {}
    for chip_info in chip_infos:
        square = get_mgrs_square(chip_info.grid_id)
        chip_dir = chips_base_dir / square / chip_info.dir_name
        if not chip_dir.exists():
            # Skip chips without directories (no masks generated)
            continue

        item = _create_chip_item(
            chip_info=chip_info,
            temporal_extent=temporal_extent,
            chip_dir=chip_dir,
            checksums=checksums,
            background_class_value=background_class_value,
        )
        if item:
            items.append(item)
            item_squares[item.id] = square

    log(f"Created {len(items)} items with mask assets")

    # Create the single collection
    log("Creating collection...")
    metadata = config.metadata if config is not None else None
    if metadata is None or not metadata.license:
        log("Warning: no metadata.license; the collection is not Portolan-publishable without one")

    collection = _create_collection(
        dataset_name=field_dataset,
        fields_file=fields_file,
        boundary_lines_file=boundary_lines_file,
        chips_file=chips_file,
        temporal_extent=temporal_extent,
        spatial_extent=spatial_extent,
        filtered_fields_file=filtered_fields_file,
        checksums=checksums,
    )

    updated = _updated_stamp(provenance)
    base_title = metadata.title if (metadata and metadata.title) else field_dataset
    _apply_collection_metadata(
        collection,
        metadata,
        updated=updated,
        title=base_title,
        description=metadata.description if metadata else None,
    )
    if config is not None and config.source_via:
        collection.add_link(
            Link(
                rel="via",
                target=config.source_via,
                media_type="application/json",
                title="Source field boundary collection",
            )
        )
    if config is not None:
        collection.extra_fields.update(_collection_ftw_properties(config))
    if provenance is not None:
        collection.extra_fields["ftw:config"] = provenance

    # Group items by MGRS square and add one sub-catalog per square
    groups = _group_items_by_square(items, item_squares)
    for square, square_items in groups.items():
        sub = Catalog(
            id=square,
            description=f"Chips in MGRS 100 km square {square}",
            title=square,
        )
        for item in square_items:
            sub.add_item(item)
            item.set_collection(collection)
        collection.add_child(sub)

    # normalize_hrefs assigns each item's self href in-memory (needed for rustac to
    # serialize them below) without writing any files yet.
    log("Writing STAC catalog...")
    collection.normalize_hrefs(str(output_dir), strategy=CHIP_LAYOUT)

    # Write stac-geoparquet before the single collection.save() below so its file:size
    # lands in the collection.json that save writes; a second save_object() call
    # would otherwise write an absolute filesystem self link into that file.
    items_parquet_path: Path | None = None
    if items:
        log("Writing stac-geoparquet...")
        items_parquet_path = output_dir / "items.parquet"
        _write_items_parquet(items, items_parquet_path)
        _add_parquet_asset(
            collection,
            "items",
            items_parquet_path,
            "STAC items in GeoParquet format (collection mirror)",
            checksums=checksums,
            roles=["collection-mirror"],
        )

    collection.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)

    log("STAC catalog generation complete")

    return STACGenerationResult(
        collection_path=output_dir / "collection.json",
        items_parquet_path=items_parquet_path,
        subcatalog_paths={
            square: output_dir / "chips" / square / "catalog.json" for square in groups
        },
        total_items=len(items),
        temporal_extent=temporal_extent,
    )
