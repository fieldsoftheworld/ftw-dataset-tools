"""Grid utilities for fetching and generating MGRS grids."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import geopandas as gpd
import mgrs
import numpy as np
from shapely.geometry import Polygon

from ftw_dataset_tools.api.geo import detect_crs, write_geoparquet

if TYPE_CHECKING:
    from collections.abc import Callable

# Default MGRS grid source (precomputed tiles on Source Coop)
DEFAULT_GRID_SOURCE = "s3://us-west-2.opendata.source.coop/tge-labs/mgrs/gzd_partition/*/*.parquet"

# Special value to generate MGRS tiles dynamically instead of fetching from remote
DYNAMIC_GRID = "__DYNAMIC__"

# MGRS tile ID regex pattern
# Format: GZD (1-2 digits + letter) + Square ID (2 letters) + Numeric (even length)
# Examples: "33UUP5081" (1km), "33UUP50008100" (100m), "33UUP" (100km)
_MGRS_PATTERN = re.compile(r"^(\d{1,2}[C-HJ-NP-X])([A-HJ-NP-Z]{2})(\d*)$")

# Tile size in meters for each precision level
_PRECISION_TO_SIZE = {
    0: 100_000,  # 100km
    1: 10_000,  # 10km
    2: 1_000,  # 1km
    3: 100,  # 100m
    4: 10,  # 10m
    5: 1,  # 1m
}


class CRSError(Exception):
    """Raised when input file has incorrect CRS."""

    def __init__(self, crs: str, input_file: str) -> None:
        self.crs = crs
        self.input_file = input_file
        super().__init__(
            f"Input file has CRS '{crs}', but must be EPSG:4326.\n"
            f"Please reproject first with:\n"
            f"  ftwd reproject {input_file} --target-crs EPSG:4326"
        )


@dataclass
class GetGridResult:
    """Result of a get-grid operation."""

    output_path: Path
    grid_count: int
    bounds: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax


def get_grid(
    input_file: str | Path,
    output_file: str | Path | None = None,
    grid_source: str = DEFAULT_GRID_SOURCE,
    geom_col: str = "geometry",
    precise: bool = False,
    precision: int = 2,
    on_progress: Callable[[str], None] | None = None,
) -> GetGridResult:
    """
    Fetch MGRS grid cells that intersect the input file's geometries.

    Args:
        input_file: Path to input GeoParquet file (must be EPSG:4326)
        output_file: Path to output file. If None, creates <input>_grid.parquet
        grid_source: URL/path to the grid source (default: MGRS grid on Source Coop).
                     Use DYNAMIC_GRID to generate tiles on-the-fly instead of fetching.
        geom_col: Name of the geometry column in the input file
        precise: If True, use geometry union for precise matching. If False (default),
                 use bounding box only (faster but may include extra grids).
                 Ignored when grid_source=DYNAMIC_GRID.
        precision: MGRS precision level for dynamic generation (0=100km, 1=10km,
                   2=1km, 3=100m, 4=10m, 5=1m). Default is 2 (1km tiles). Only used when
                   grid_source=DYNAMIC_GRID.
        on_progress: Optional callback for progress messages

    Returns:
        GetGridResult with information about the operation

    Raises:
        FileNotFoundError: If input file doesn't exist
        CRSError: If input file is not in EPSG:4326
    """
    input_path = Path(input_file).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    # Check CRS using geo module
    crs_info = detect_crs(input_path, geom_col)
    crs_str = str(crs_info)

    if crs_info.authority_code is None or crs_info.authority_code.upper() != "EPSG:4326":
        raise CRSError(crs_str, str(input_file))

    log(f"CRS: {crs_str}")

    # Determine output path
    if output_file:
        out_path = Path(output_file).resolve()
    else:
        out_path = input_path.parent / f"{input_path.stem}_grid.parquet"

    # Handle dynamic grid generation
    if grid_source == DYNAMIC_GRID:
        return _get_grid_dynamic(
            input_path=input_path,
            output_path=out_path,
            geom_col=geom_col,
            precision=precision,
            on_progress=on_progress,
        )

    # Fetch from remote source
    return _get_grid_remote(
        input_path=input_path,
        output_path=out_path,
        grid_source=grid_source,
        geom_col=geom_col,
        precise=precise,
        on_progress=on_progress,
    )


def _get_grid_dynamic(
    input_path: Path,
    output_path: Path,
    geom_col: str,  # noqa: ARG001 - kept for API consistency with _get_grid_remote
    precision: int,
    on_progress: Callable[[str], None] | None,
) -> GetGridResult:
    """Generate MGRS grid tiles dynamically for input file bounds."""

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    log("Generating grid dynamically...")

    # Load input and compute bounds
    gdf_input = gpd.read_parquet(input_path)
    total_bounds = gdf_input.total_bounds  # (minx, miny, maxx, maxy)
    bounds = (total_bounds[0], total_bounds[1], total_bounds[2], total_bounds[3])

    log(f"Loaded {len(gdf_input):,} features")

    # Generate grid
    gdf = _generate_mgrs_grid(
        bounds=bounds,
        precision=precision,
        on_progress=on_progress,
    )

    # Write output
    log(f"Writing output to: {output_path}")
    write_geoparquet(output_path, gdf=gdf)

    return GetGridResult(
        output_path=output_path,
        grid_count=len(gdf),
        bounds=bounds,
    )


def _get_grid_remote(
    input_path: Path,
    output_path: Path,
    grid_source: str,
    geom_col: str,
    precise: bool,
    on_progress: Callable[[str], None] | None,
) -> GetGridResult:
    """Fetch MGRS grid tiles from a remote source."""

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    # Create DuckDB connection with required extensions
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute("INSTALL httpfs; LOAD httpfs;")

    # Set S3 region for Source Coop
    conn.execute("SET s3_region = 'us-west-2';")

    # Load input file
    log("Loading input file...")
    conn.execute(f"CREATE TABLE input_data AS SELECT * FROM '{input_path}'")
    feature_count = conn.execute("SELECT COUNT(*) FROM input_data").fetchone()[0]
    log(f"Loaded {feature_count:,} features")

    # Compute bounds for bbox filtering
    bounds_result = conn.execute(f"""
        SELECT
            MIN(ST_XMin("{geom_col}")) as xmin,
            MIN(ST_YMin("{geom_col}")) as ymin,
            MAX(ST_XMax("{geom_col}")) as xmax,
            MAX(ST_YMax("{geom_col}")) as ymax
        FROM input_data
    """).fetchone()
    xmin, ymin, xmax, ymax = bounds_result
    log(f"Bounds: [{xmin:.6f}, {ymin:.6f}, {xmax:.6f}, {ymax:.6f}]")

    # Fetch grids that intersect the bounding box
    log("Fetching grid cells by bounding box...")
    conn.execute(f"""
        CREATE TABLE grid_bbox AS
        SELECT *
        FROM '{grid_source}'
        WHERE "bbox".xmin <= {xmax}
          AND "bbox".xmax >= {xmin}
          AND "bbox".ymin <= {ymax}
          AND "bbox".ymax >= {ymin}
    """)

    bbox_count = conn.execute("SELECT COUNT(*) FROM grid_bbox").fetchone()[0]
    log(f"Found {bbox_count:,} grid cells in bounding box")

    if precise:
        # Compute union of all geometries for precise filtering
        log("Computing geometry union for precise matching...")
        conn.execute(f"""
            CREATE TABLE input_union AS
            SELECT ST_Union_Agg("{geom_col}") as union_geom FROM input_data
        """)

        # Get union geometry as WKT for reporting
        union_wkt = conn.execute("SELECT ST_AsText(union_geom) FROM input_union").fetchone()[0]
        log(f"Union geometry: {union_wkt}")

        # Filter locally using geometry intersection
        log("Filtering grids by geometry intersection...")
        conn.execute("""
            CREATE TABLE grid_result AS
            SELECT g.*
            FROM grid_bbox g, input_union u
            WHERE ST_Intersects(g.geom, u.union_geom)
        """)
    else:
        # Use bbox results directly
        conn.execute("CREATE TABLE grid_result AS SELECT * FROM grid_bbox")

    # Get count
    grid_count = conn.execute("SELECT COUNT(*) FROM grid_result").fetchone()[0]
    log(f"Found {grid_count:,} intersecting grid cells")

    # Write output with zstd compression level 16
    log(f"Writing output to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn.execute(f"""
        COPY grid_result TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION 'zstd', COMPRESSION_LEVEL 16)
    """)

    conn.close()

    return GetGridResult(
        output_path=output_path,
        grid_count=grid_count,
        bounds=(xmin, ymin, xmax, ymax),
    )


def parse_mgrs_tile(tile_id: str) -> tuple[str, str, str] | None:
    """
    Parse an MGRS tile ID into its components.

    Args:
        tile_id: MGRS tile identifier (e.g., "33UUP5081")

    Returns:
        Tuple of (grid_zone_designator, square_id, numeric_part) or None if invalid.
        For "33UUP5081" returns ("33U", "UP", "5081")
    """
    match = _MGRS_PATTERN.match(tile_id.upper().strip())
    if not match:
        return None
    gzd, square_id, numeric = match.groups()
    # Numeric part must have even length (equal easting/northing digits)
    if len(numeric) % 2 != 0:
        return None
    return gzd, square_id, numeric


def _mgrs_numeric_to_meters(numeric: str) -> tuple[int, int, int]:
    """
    Convert MGRS numeric part to easting, northing in meters and tile size.

    Args:
        numeric: The numeric part of MGRS tile ID (e.g., "5081" for 1km tile)

    Returns:
        Tuple of (easting_meters, northing_meters, tile_size_meters)
    """
    if not numeric:
        # 100km tile (precision 0)
        return 0, 0, 100_000

    half_len = len(numeric) // 2
    precision = half_len  # 1=10km, 2=1km, 3=100m, etc.

    easting_str = numeric[:half_len]
    northing_str = numeric[half_len:]

    # Scale factor: for precision 2 (1km), "50" means 50000m
    scale = 10 ** (5 - precision)

    easting = int(easting_str) * scale
    northing = int(northing_str) * scale
    tile_size = _PRECISION_TO_SIZE.get(precision, 1)

    return easting, northing, tile_size


def mgrs_tile_to_polygon(tile_id: str) -> Polygon | None:
    """
    Convert an MGRS tile ID to a Shapely Polygon in WGS84.

    The polygon represents the tile's bounding box. For 1km tiles (precision 2),
    a tile like "33UUP5081" covers a 1km x 1km area.

    Args:
        tile_id: MGRS tile identifier (e.g., "33UUP5081")

    Returns:
        Shapely Polygon in WGS84 (EPSG:4326), or None if conversion fails.

    Note:
        Some tiles near 100km square boundaries may fail to convert due to
        limitations in the mgrs library.
    """
    parsed = parse_mgrs_tile(tile_id)
    if parsed is None:
        return None

    gzd, square_id, numeric = parsed
    easting, northing, tile_size = _mgrs_numeric_to_meters(numeric)

    # Build base MGRS string with full precision (1m)
    base = f"{gzd}{square_id}"

    converter = mgrs.MGRS()

    try:
        # Compute the 4 corners of the tile
        corners_latlon = []
        corner_offsets = [
            (0, 0),  # SW
            (tile_size, 0),  # SE
            (tile_size, tile_size),  # NE
            (0, tile_size),  # NW
        ]

        for de, dn in corner_offsets:
            e = easting + de
            n = northing + dn
            # Format as 5-digit easting and northing for 1m precision
            mgrs_str = f"{base}{e:05d}{n:05d}"
            try:
                lat, lon = converter.toLatLon(mgrs_str)
                corners_latlon.append((lon, lat))
            except Exception:
                # Edge tiles may fail for some corners - try using the short format
                # with adjusted coordinates to get approximate position
                # Fall back to using the tile center and estimating corners
                return _mgrs_tile_to_polygon_fallback(tile_id, tile_size, converter)

        # Close the polygon
        corners_latlon.append(corners_latlon[0])

        return Polygon(corners_latlon)

    except Exception:
        # Some edge tiles fail to convert
        return None


def _mgrs_tile_to_polygon_fallback(
    tile_id: str,
    tile_size: int,
    converter: mgrs.MGRS,
) -> Polygon | None:
    """
    Fallback method to create tile polygon when corner conversion fails.

    Uses the short-format MGRS conversion (which returns SW corner) and
    estimates the tile extent based on approximate degree conversion.
    """
    try:
        # Short format returns the SW corner of the tile
        lat_sw, lon_sw = converter.toLatLon(tile_id)

        # Approximate degrees per meter at this latitude
        # 1 degree latitude ≈ 111,000 meters
        # 1 degree longitude ≈ 111,000 * cos(lat) meters
        import math

        meters_per_deg_lat = 111_000
        meters_per_deg_lon = 111_000 * math.cos(math.radians(lat_sw))

        tile_size_lat = tile_size / meters_per_deg_lat
        tile_size_lon = tile_size / meters_per_deg_lon

        # Create polygon from SW corner
        corners = [
            (lon_sw, lat_sw),  # SW
            (lon_sw + tile_size_lon, lat_sw),  # SE
            (lon_sw + tile_size_lon, lat_sw + tile_size_lat),  # NE
            (lon_sw, lat_sw + tile_size_lat),  # NW
            (lon_sw, lat_sw),  # Close
        ]

        return Polygon(corners)

    except Exception:
        return None


def _generate_mgrs_grid(
    bounds: tuple[float, float, float, float],
    precision: int = 2,
    on_progress: Callable[[str], None] | None = None,
) -> gpd.GeoDataFrame:
    """
    Generate MGRS grid tiles on-the-fly for a bounding box.

    This function generates MGRS tiles by sampling points within the bounding box,
    converting them to MGRS tile IDs, and then computing the polygon for each
    unique tile.

    Args:
        bounds: Bounding box as (xmin, ymin, xmax, ymax) in WGS84
        precision: MGRS precision level (0=100km, 1=10km, 2=1km, 3=100m).
                   Default is 2 (1km tiles).
        on_progress: Optional callback for progress messages.

    Returns:
        GeoDataFrame with columns [tile_id, geometry]
    """
    xmin, ymin, xmax, ymax = bounds

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    log(f"Generating MGRS grid for bounds: [{xmin:.6f}, {ymin:.6f}, {xmax:.6f}, {ymax:.6f}]")
    log(f"Precision: {precision} ({_PRECISION_TO_SIZE.get(precision, '?')}m tiles)")

    converter = mgrs.MGRS()
    tile_size_m = _PRECISION_TO_SIZE.get(precision, 1000)
    tile_size_deg = tile_size_m / 111_000  # Rough m to deg conversion

    # Sample points at half the tile size to ensure we catch all tiles
    step = tile_size_deg / 2

    # Expand sampling bounds by one full tile size to catch edge tiles
    # Tiles whose geometry overlaps the input bbox but whose centroid is outside
    # would otherwise be missed
    sample_xmin = xmin - tile_size_deg
    sample_xmax = xmax + tile_size_deg
    sample_ymin = ymin - tile_size_deg
    sample_ymax = ymax + tile_size_deg

    # Generate sample points with expanded bounds
    lons = np.arange(sample_xmin, sample_xmax + step, step)
    lats = np.arange(sample_ymin, sample_ymax + step, step)

    # Add UTM zone boundaries to sampling points
    # UTM zones occur every 6 degrees: ..., -6, 0, 6, 12, 18, ...
    # At zone boundaries, tiles from both zones can overlap, and sampling
    # just west vs just east of the boundary returns different tiles.
    # We add both sides of each boundary to catch all overlapping tiles.
    zone_boundaries = np.arange(-180, 180, 6)
    boundary_lons = []
    epsilon = 0.0001  # Small offset from boundary
    for boundary in zone_boundaries:
        if sample_xmin < boundary < sample_xmax:
            # Add points just west and just east of the boundary
            boundary_lons.extend([boundary - epsilon, boundary, boundary + epsilon])
    if boundary_lons:
        lons = np.unique(np.concatenate([lons, boundary_lons]))
        lons.sort()

    log(f"Sampling {len(lons)} x {len(lats)} = {len(lons) * len(lats):,} points")

    # Collect unique tile IDs
    tile_ids: set[str] = set()
    for lat in lats:
        for lon in lons:
            try:
                tile_id = converter.toMGRS(lat, lon, MGRSPrecision=precision)
                tile_ids.add(tile_id)
            except Exception:
                # Skip points that fail (e.g., outside valid MGRS range)
                continue

    log(f"Found {len(tile_ids):,} unique tile IDs")

    # Convert tile IDs to polygons
    geometries = []
    valid_tile_ids = []
    failed_count = 0

    for tile_id in sorted(tile_ids):
        polygon = mgrs_tile_to_polygon(tile_id)
        if polygon is not None:
            geometries.append(polygon)
            valid_tile_ids.append(tile_id)
        else:
            failed_count += 1

    if failed_count > 0:
        log(f"Warning: {failed_count} tiles failed to convert to polygons")

    log(f"Created {len(geometries):,} grid polygons")

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {"tile_id": valid_tile_ids, "geometry": geometries},
        crs="EPSG:4326",
    )

    # Filter out tiles that don't intersect the input bounding box
    # We first filter by geometry intersection, then apply a secondary filter
    # for tiles at UTM zone boundaries where our computed polygon may have
    # slight inaccuracies due to zone clipping in the reference grid.
    bounds_box = Polygon(
        [
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
            (xmin, ymin),
        ]
    )

    # First pass: use geometry intersection
    original_count = len(gdf)
    gdf = gdf[gdf.geometry.intersects(bounds_box)].copy()

    # Second pass: filter zone-boundary edge tiles more carefully
    # The remote grid clips tiles at UTM zone boundaries, so tiles whose SW corner
    # is outside the zone boundary may have geometry that doesn't actually overlap
    # with the input bbox in the remote grid's representation.
    def tile_valid_for_zone(tile_id: str) -> bool:
        """Filter tiles at zone boundaries that shouldn't be included."""
        try:
            lat_sw, lon_sw = converter.toLatLon(tile_id)

            # Parse zone from tile ID
            parsed = parse_mgrs_tile(tile_id)
            if not parsed:
                return True

            gzd = parsed[0]
            zone_num = int("".join(c for c in gzd if c.isdigit()))
            zone_west = (zone_num - 1) * 6 - 180

            # Check if this tile straddles the zone boundary
            # (SW corner is west of zone boundary)
            # For zone-boundary tiles, the remote grid has much smaller geometry
            # due to clipping. If the tile's SW corner is outside the input bbox,
            # and the tile straddles a zone boundary, it's likely that the
            # clipped geometry doesn't actually overlap.
            # Simple heuristic: if SW corner is outside the input bbox in the
            # y-dimension, exclude the tile (the clipped portion is too small)
            return not (lon_sw < zone_west and lat_sw < ymin)
        except Exception:
            return True

    gdf = gdf[gdf["tile_id"].apply(tile_valid_for_zone)].copy()

    filtered_count = original_count - len(gdf)
    if filtered_count > 0:
        log(f"Filtered {filtered_count} tiles outside bounds, {len(gdf)} remaining")

    return gdf
