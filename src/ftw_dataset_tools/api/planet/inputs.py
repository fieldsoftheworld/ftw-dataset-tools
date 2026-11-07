"""Read polygon AOIs from fiboa GeoParquet or JSONL."""

import json
import re
from pathlib import Path

from pyproj import Transformer
from shapely import from_wkb
from shapely.geometry import mapping, shape
from shapely.ops import transform

from ftw_dataset_tools.api.planet.state import digest, read_jsonl


def parquet_rows(
    path: Path,
    id_column: str,
    geometry_column: str | None,
    input_crs: str | None,
    columns: tuple[str, ...],
    limit: int | None,
) -> tuple[list[dict], object]:
    """Read WKB geometry and requested attributes with GeoParquet CRS metadata."""
    import geoparquet_io as gpio

    table = gpio.read(path)
    if limit is not None:
        table = table.head(limit)
    arrow = table.to_arrow()
    metadata = json.loads((arrow.schema.metadata or {}).get(b"geo", b"{}"))
    geometry_column = geometry_column or metadata.get("primary_column", "geometry")
    geometry_metadata = metadata.get("columns", {}).get(geometry_column, {})
    if not geometry_metadata and input_crs is None:
        raise ValueError("Geometry metadata is missing; specify --geometry-column and --input-crs")
    crs = input_crs or geometry_metadata.get("crs", "OGC:CRS84")
    if crs is None:
        raise ValueError("Geometry CRS is unknown; specify --input-crs")
    if geometry_metadata.get("encoding", "WKB") != "WKB":
        raise ValueError("Expected WKB geometry encoding")
    for column in (id_column, geometry_column):
        if column not in arrow.column_names:
            raise ValueError(f"Column not found: {column}")
    selected = list(
        dict.fromkeys(
            [id_column, geometry_column, *(name for name in columns if name in arrow.column_names)]
        )
    )
    rows = []
    for record in arrow.select(selected).to_pylist():
        geometry = record.pop(geometry_column)
        if geometry is None:
            raise ValueError(f"Null geometry for {record.get(id_column)}")
        row = {
            name: value.isoformat() if hasattr(value, "isoformat") else value
            for name, value in record.items()
        }
        row["id"] = record[id_column]
        row["geometry"] = mapping(from_wkb(geometry))
        rows.append(row)
    return rows, crs


def read_aois(
    path: Path,
    *,
    id_column: str = "id",
    geometry_column: str | None = None,
    input_crs: str | None = None,
    extra_columns: tuple[str, ...] = (),
    limit: int | None = None,
) -> list[dict]:
    """Return validated WGS84 polygons, retaining source IDs and portable output IDs."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    columns = ("start", "end", "target_date", "mosaics", *extra_columns)
    if path.suffix.lower() in {".parquet", ".geoparquet"}:
        rows, crs = parquet_rows(path, id_column, geometry_column, input_crs, columns, limit)
    else:
        rows = read_jsonl(path)
        if limit is not None:
            rows = rows[:limit]
        rows = [
            dict(row, id=row[id_column], geometry=row[geometry_column or "geometry"])
            for row in rows
        ]
        crs = input_crs or "EPSG:4326"
    projector = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    seen = set()
    for row in rows:
        if row["id"] is None or str(row["id"]) == "":
            raise ValueError("AOI IDs must not be null or empty")
        source_id = str(row["id"])
        identifier = (
            source_id
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", source_id)
            else f"polygon-{digest(source_id)}"
        )
        if identifier in seen:
            raise ValueError(f"Duplicate AOI id: {source_id}")
        seen.add(identifier)
        geometry = transform(projector.transform, shape(row["geometry"]))
        validate_geometry(geometry, source_id)
        row.update(id=identifier, source_id=source_id, geometry=mapping(geometry))
    if not rows:
        raise ValueError("Input contains no AOIs")
    return rows


def validate_geometry(geometry: object, identifier: str) -> None:
    """Reject invalid polygons and ambiguous antimeridian bounds before API calls."""
    if geometry.geom_type not in {"Polygon", "MultiPolygon"} or not geometry.is_valid:
        raise ValueError(f"Invalid polygon: {identifier}")
    if geometry.is_empty or geometry.area <= 0:
        raise ValueError(f"Empty polygon: {identifier}")
    west, south, east, north = geometry.bounds
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError(f"Expected WGS84 coordinates: {identifier}")
    if east - west > 180:
        raise ValueError(f"Split antimeridian-crossing AOIs first: {identifier}")
