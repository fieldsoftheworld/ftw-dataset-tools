"""Tests for HCAT3 crop labels on field polygons."""

import csv
import json
from pathlib import Path

import duckdb
import geopandas as gpd
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from ftw_dataset_tools.api import crop_labels
from ftw_dataset_tools.api.crop_labels import CropLabelError, add_crop_labels

CN_LIST = "https://fiboa.org/code/es/cn/crop.csv"
WHEAT, BARLEY, POTATOES = "3301010100", "3301010400", "3301030000"


def _fields(tmp_path: Path, data: dict, *, code_list: str | None = None) -> Path:
    n = len(next(iter(data.values())))
    gdf = gpd.GeoDataFrame(
        {"id": [f"f{i}" for i in range(n)], **data},
        geometry=[box(i, 0, i + 1, 1) for i in range(n)],
        crs="EPSG:4326",
    )
    path = tmp_path / "fields.parquet"
    gdf.to_parquet(path)
    if code_list is not None:
        table = pq.read_table(path)
        collection = {"schemas": {"x": ["https://fiboa.org/specification/v0.3.0/schema.yaml"]}}
        collection["crop:code_list"] = code_list
        metadata = {**table.schema.metadata, b"collection": json.dumps(collection).encode()}
        pq.write_table(table.replace_schema_metadata(metadata), path)
    return path


def _mapping(tmp_path: Path, rows: list[dict], name: str = "mapping.csv") -> Path:
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read(path: Path, columns: str = '"hcat:code", "hcat:name", "hcat:name_en"') -> list[tuple]:
    con = duckdb.connect()
    try:
        return con.execute(f"SELECT id, {columns} FROM '{path}'").fetchall()
    finally:
        con.close()


def _kv(path: Path) -> dict[str, str]:
    return {k.decode(): v.decode() for k, v in pq.read_schema(path).metadata.items()}


def _template(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestBundledCodeLists:
    def test_resolves_fiboa_and_eurocrops_urls(self) -> None:
        assert crop_labels.bundled_code_list(CN_LIST).name == "crop.csv"
        eurocrops = (
            "https://raw.githubusercontent.com/maja601/EuroCrops/refs/heads/main/"
            "csvs/country_mappings/dk_2019.csv"
        )
        assert crop_labels.bundled_code_list(eurocrops).name == "dk_2019.csv"

    def test_unknown_url_is_none(self) -> None:
        assert crop_labels.bundled_code_list("https://fiboa.org/code/de/de_mv.csv") is None
        assert crop_labels.bundled_code_list(None) is None

    def test_every_bundled_list_has_hcat3_and_a_key(self) -> None:
        root = Path(str(crop_labels._CODE_LISTS))
        lists = [p for p in root.rglob("*.csv") if p.name != "HCAT3.csv"]
        assert len(lists) > 50
        for path in lists:
            rows = crop_labels._read_csv(path)
            assert rows, path
            assert "HCAT3_code" in rows[0], path
            assert {"original_code", "original_name"} & set(rows[0]), path
            codes = {row["HCAT3_code"] for row in rows if row["HCAT3_code"]}
            assert all(len(code) == 10 and code.isdigit() for code in codes), path


class TestAddCropLabels:
    def test_native_hcat_is_kept(self, tmp_path: Path) -> None:
        fields = _fields(tmp_path, {"hcat:code": [int(WHEAT), None], "crop:code": ["1", "2"]})
        before = fields.read_bytes()

        result = add_crop_labels(fields)

        assert (result.source, result.fields_labeled, result.fields_total) == ("native", 1, 2)
        assert fields.read_bytes() == before

    def test_bundled_code_list_from_collection_metadata(self, tmp_path: Path) -> None:
        fields = _fields(
            tmp_path,
            {"crop:code": ["R1000", "W1000", "R1000"], "crop:name": ["Papa", "Viña", "Papa"]},
            code_list=CN_LIST,
        )

        result = add_crop_labels(fields)

        assert (result.source, result.fields_labeled, result.unmapped_values) == ("code_list", 3, 0)
        assert _read(fields) == [
            ("f0", int(POTATOES), "potatoes", "Potato"),
            ("f1", 3303060000, "vineyards_wine_vine_rebland_grapes", "Vineyard"),
            ("f2", int(POTATOES), "potatoes", "Potato"),
        ]
        kv = _kv(fields)
        assert crop_labels.HCAT_SCHEMA in json.loads(kv["collection"])["schemas"]["x"]
        assert json.loads(kv["collection"])["crop:code_list"] == CN_LIST
        assert crop_labels.read_provenance(fields)["code_list"] == CN_LIST
        assert not (tmp_path / "fields_crop_mapping_template.csv").exists()

    def test_ambiguous_code_matches_on_name(self, tmp_path: Path) -> None:
        fields = _fields(tmp_path, {"crop:code": ["7", "7"], "crop:name": ["Weizen", "Gerste"]})
        mapping = _mapping(
            tmp_path,
            [
                {"original_code": "7", "original_name": "Weizen", "HCAT3_code": WHEAT},
                {"original_code": "7", "original_name": "Gerste", "HCAT3_code": BARLEY},
            ],
        )

        add_crop_labels(fields, mapping=mapping)

        assert [row[1] for row in _read(fields)] == [int(WHEAT), int(BARLEY)]

    def test_numeric_codes_match_across_formats(self, tmp_path: Path) -> None:
        fields = _fields(tmp_path, {"crop:code": ["01", "2.0"]})
        mapping = _mapping(
            tmp_path,
            [
                {"original_code": "1", "HCAT3_code": WHEAT},
                {"original_code": "02", "HCAT3_code": BARLEY},
            ],
        )

        result = add_crop_labels(fields, mapping=mapping)

        assert result.fields_labeled == 2
        # HCAT3_name is filled from the core list when the mapping leaves it out.
        assert _read(fields)[0][2] == "common_soft_wheat"

    def test_partial_mapping_writes_template_and_keeps_going(self, tmp_path: Path) -> None:
        fields = _fields(tmp_path, {"crop:code": ["A", "B", "B", "C"]})
        mapping = _mapping(tmp_path, [{"original_code": "A", "HCAT3_code": WHEAT}])
        template = tmp_path / "template.csv"

        result = add_crop_labels(fields, mapping=mapping, template_path=template)

        assert (result.fields_labeled, result.unmapped_values) == (1, 2)
        assert result.template_path == template
        assert [row[1] for row in _read(fields)] == [int(WHEAT), None, None, None]
        rows = _template(template)
        assert [(r["original_code"], r["fields"], r["HCAT3_code"]) for r in rows] == [
            ("B", "2", ""),
            ("C", "1", ""),
            ("A", "1", WHEAT),
        ]

    def test_filled_template_is_a_mapping(self, tmp_path: Path) -> None:
        fields = _fields(tmp_path, {"crop:code": ["A", "B"], "crop:name": ["Wheat", "Spud"]})
        template = tmp_path / "template.csv"
        add_crop_labels(fields, template_path=template)
        rows = _template(template)
        assert [r["original_name"] for r in rows] == ["Wheat", "Spud"]
        for row in rows:
            row["HCAT3_code"] = WHEAT if row["original_code"] == "A" else POTATOES
        filled = _mapping(tmp_path, rows, name="filled.csv")

        result = add_crop_labels(fields, mapping=filled, template_path=template)

        assert result.fields_labeled == 2
        assert not template.exists()

    def test_strict_fails_on_unmapped(self, tmp_path: Path) -> None:
        fields = _fields(tmp_path, {"crop:code": ["A", "B"]})
        mapping = _mapping(tmp_path, [{"original_code": "A", "HCAT3_code": WHEAT}])
        before = fields.read_bytes()

        with pytest.raises(CropLabelError, match="1 crop value") as err:
            add_crop_labels(fields, mapping=mapping, strict=True)

        assert err.value.template_path.exists()
        assert fields.read_bytes() == before

    def test_unknown_code_list_skips_with_template(self, tmp_path: Path) -> None:
        fields = _fields(
            tmp_path, {"crop:code": ["DGL"]}, code_list="https://fiboa.org/code/de/de_mv.csv"
        )
        before = fields.read_bytes()

        result = add_crop_labels(fields)

        assert result.source == "skipped"
        assert result.template_path.exists()
        assert fields.read_bytes() == before

    def test_no_crop_column(self, tmp_path: Path) -> None:
        fields = _fields(tmp_path, {"landuse": ["x"]})

        assert add_crop_labels(fields).reason.startswith("no crop column")
        with pytest.raises(CropLabelError, match="no crop column"):
            add_crop_labels(fields, strict=True)

    def test_mapping_with_non_hcat3_code_is_rejected(self, tmp_path: Path) -> None:
        fields = _fields(tmp_path, {"crop:code": ["A"]})
        mapping = _mapping(tmp_path, [{"original_code": "A", "HCAT3_code": "1234"}])

        with pytest.raises(CropLabelError, match="not HCAT3"):
            add_crop_labels(fields, mapping=mapping)

    def test_rerun_replaces_labels_ftwd_added(self, tmp_path: Path) -> None:
        fields = _fields(tmp_path, {"crop:code": ["A"]})
        add_crop_labels(
            fields, mapping=_mapping(tmp_path, [{"original_code": "A", "HCAT3_code": WHEAT}])
        )

        # Not treated as native: ftwd recorded that it wrote these columns.
        result = add_crop_labels(
            fields, mapping=_mapping(tmp_path, [{"original_code": "A", "HCAT3_code": BARLEY}])
        )

        assert result.source == "mapping"
        assert _read(fields)[0][1] == int(BARLEY)

    def test_output_file_leaves_input_alone(self, tmp_path: Path) -> None:
        fields = _fields(tmp_path, {"crop:code": ["A"]})
        out = tmp_path / "out.parquet"
        before = fields.read_bytes()

        add_crop_labels(
            fields, out, mapping=_mapping(tmp_path, [{"original_code": "A", "HCAT3_code": WHEAT}])
        )

        assert fields.read_bytes() == before
        assert _read(out)[0][1] == int(WHEAT)

    def test_name_only_list(self, tmp_path: Path) -> None:
        fields = _fields(tmp_path, {"crop:code": ["9"], "crop_name": ["Winter wheat"]})
        mapping = _mapping(tmp_path, [{"original_name": "WINTER WHEAT", "HCAT3_code": WHEAT}])

        assert add_crop_labels(fields, mapping=mapping).fields_labeled == 1


class TestSummary:
    def test_lines(self) -> None:
        result = crop_labels.CropLabelResult("code_list", 4, 3, 2)
        assert crop_labels.crop_labels_summary(result) == (
            "Crop labels (code_list): 3/4 fields (75.0%), 2 HCAT codes"
        )
        assert crop_labels.crop_labels_summary(None) == "Crop labels: disabled"


class TestTemplateAsMapping:
    def test_template_used_as_mapping_is_not_deleted(self, tmp_path: Path) -> None:
        fields = _fields(tmp_path, {"crop:code": ["A"]})
        template = _mapping(tmp_path, [{"original_code": "A", "HCAT3_code": WHEAT}], "t.csv")

        add_crop_labels(fields, mapping=template, template_path=template)

        assert template.exists()
