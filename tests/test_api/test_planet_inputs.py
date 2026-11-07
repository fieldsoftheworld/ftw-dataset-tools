"""GeoParquet schema, reprojection, and polygon identity regressions."""

import json
from datetime import date

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyproj import CRS, Transformer
from shapely import to_wkb
from shapely.geometry import box, mapping, shape
from shapely.ops import transform

from ftw_dataset_tools.api.planet.inputs import read_aois


def write_fields(path, rows, *, crs="EPSG:4326", geometry_column="geometry"):
    projector = Transformer.from_crs("EPSG:4326", crs or "EPSG:4326", always_xy=True)
    records = []
    for row in rows:
        record = dict(row)
        polygon = transform(projector.transform, shape(record.pop("geometry")))
        record[geometry_column] = to_wkb(polygon)
        records.append(record)
    table = pa.Table.from_pylist(records)
    metadata = {
        "version": "1.1.0",
        "primary_column": geometry_column,
        "columns": {
            geometry_column: {
                "encoding": "WKB",
                "geometry_types": ["Polygon"],
                "crs": CRS(crs).to_json_dict() if crs else None,
            }
        },
    }
    pq.write_table(table.replace_schema_metadata({b"geo": json.dumps(metadata).encode()}), path)


def test_projected_fiboa_custom_columns_dates_and_ids(tmp_path):
    path = tmp_path / "fields.parquet"
    polygon = box(-90.01, 38.60, -90, 38.61)
    write_fields(
        path,
        [
            {
                "field_id": "region/parcel:1",
                "geometry": mapping(polygon),
                "survey_date": date(2024, 6, 15),
            }
        ],
        crs="EPSG:3857",
        geometry_column="boundary",
    )
    rows = read_aois(path, id_column="field_id", extra_columns=("survey_date",))
    assert shape(rows[0]["geometry"]).equals_exact(polygon, 1e-9)
    assert rows[0]["source_id"] == "region/parcel:1"
    assert "/" not in rows[0]["id"]
    assert rows[0]["survey_date"] == "2024-06-15"


def test_unknown_crs_requires_explicit_override(tmp_path):
    path = tmp_path / "fields.parquet"
    write_fields(path, [{"id": "a", "geometry": mapping(box(0, 0, 1, 1))}], crs=None)
    with pytest.raises(ValueError, match="unknown"):
        read_aois(path)
    assert len(read_aois(path, input_crs="EPSG:4326")) == 1


def test_duplicate_ids_rejected_and_limit_applied(tmp_path):
    path = tmp_path / "fields.parquet"
    row = {"id": 4, "geometry": mapping(box(0, 0, 1, 1))}
    write_fields(path, [row, row])
    with pytest.raises(ValueError, match="Duplicate"):
        read_aois(path)
    assert read_aois(path, limit=1)[0]["source_id"] == "4"


def test_jsonl_geometry_validation(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps({"id": "a", "geometry": mapping(box(-179, 0, 179, 1))}) + "\n")
    with pytest.raises(ValueError, match="antimeridian"):
        read_aois(path)
