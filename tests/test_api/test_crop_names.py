"""Tests for joining harmonized crop names onto a benchmark's field files."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import geopandas as gpd
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from ftw_dataset_tools.api.crop_names import CropNameError, add_crop_names

if TYPE_CHECKING:
    from pathlib import Path


def write(path: Path, rows: list[dict], crs: str = "EPSG:4326") -> Path:
    """A small GeoParquet with one square per row, in the given order."""
    geoms = [box(i, 0, i + 1, 1) for i in range(len(rows))]
    gpd.GeoDataFrame(rows, geometry=geoms, crs=crs).to_parquet(path)
    return path


def source(tmp_path: Path, **overrides) -> Path:
    """A harmonized edition naming crops 10, 20 and 30."""
    rows = [
        {"id": "a", "crop:code": "10", "crop:name": "Weizen", "crop:name_en": "wheat"},
        {"id": "b", "crop:code": "20", "crop:name": "Mais", "crop:name_en": "maize"},
        {"id": "c", "crop:code": "30", "crop:name": None, "crop:name_en": None},
    ]
    for row in rows:
        row.update(overrides.get(row["id"], {}))
    return write(tmp_path / "source.parquet", rows)


def fields(tmp_path: Path, ids=("c", "a", "b"), name: str = "fields.parquet") -> Path:
    """A benchmark fields file, in another row order than the source."""
    codes = {"a": "10", "b": "20", "c": "30"}
    rows = [{"id": i, "crop:code": codes[i], "hcat:code": f"h{codes[i]}"} for i in ids]
    return write(tmp_path / name, rows)


def column(path: Path, name: str) -> list:
    return pq.read_table(path, columns=[name]).column(name).to_pylist()


class TestAddCropNames:
    """add_crop_names joins names by id, in the fields file's own row order."""

    def test_names_follow_each_field_by_id(self, tmp_path: Path) -> None:
        path = fields(tmp_path)
        result = add_crop_names(path, source(tmp_path))
        assert column(path, "id") == ["c", "a", "b"]
        assert column(path, "crop:name") == [None, "Weizen", "Mais"]
        assert column(path, "crop:name_en") == [None, "wheat", "maize"]
        assert (result.rows, result.named, result.changed) == (3, 2, True)
        assert result.columns == ("crop:name", "crop:name_en")

    def test_names_sit_after_the_crop_code_and_keep_the_rest(self, tmp_path: Path) -> None:
        path = fields(tmp_path)
        before = pq.read_schema(path).names
        add_crop_names(path, source(tmp_path))
        after = pq.read_schema(path).names
        at = after.index("crop:code")
        assert after[at + 1 : at + 3] == ["crop:name", "crop:name_en"]
        assert [c for c in after if not c.startswith("crop:name")] == before

    def test_the_output_stays_geoparquet(self, tmp_path: Path) -> None:
        path = fields(tmp_path)
        add_crop_names(path, source(tmp_path))
        geo = json.loads(pq.read_metadata(path).metadata[b"geo"])
        assert geo["primary_column"] == "geometry"
        assert gpd.read_parquet(path).crs.to_epsg() == 4326

    def test_a_projected_crs_is_kept(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "fields.parquet",
            [{"id": i, "crop:code": c} for i, c in (("a", "10"), ("b", "20"))],
            crs="EPSG:32633",
        )
        add_crop_names(path, source(tmp_path))
        assert gpd.read_parquet(path).crs.to_epsg() == 32633

    def test_the_field_embeddings_fingerprint_is_unchanged(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.field_embeddings import fields_fingerprint

        path = fields(tmp_path)
        before = fields_fingerprint(path)
        add_crop_names(path, source(tmp_path))
        assert fields_fingerprint(path) == before

    def test_the_file_layout_and_metadata_are_kept(self, tmp_path: Path) -> None:
        path = tmp_path / "fields.parquet"
        rows = [{"id": f"f{i}", "crop:code": "10"} for i in range(10)]
        geoms = [box(i, 0, i + 1, 1) for i in range(10)]
        gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326").to_parquet(
            path, row_group_size=3, compression="zstd"
        )
        src = write(
            tmp_path / "source.parquet",
            [{**r, "crop:name": "Weizen"} for r in rows],
        )
        before = pq.read_metadata(path)
        add_crop_names(path, src)
        after = pq.read_metadata(path)
        assert after.num_row_groups == before.num_row_groups == 4
        assert after.row_group(0).column(0).compression == "ZSTD"
        for key in (b"geo", b"pandas"):
            assert after.metadata[key] == before.metadata[key]
        assert column(path, "crop:name") == ["Weizen"] * 10

    def test_a_stale_english_name_is_dropped(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "fields.parquet",
            [{"id": "a", "crop:code": "10", "crop:name_en": "old"}],
        )
        src = write(
            tmp_path / "source.parquet", [{"id": "a", "crop:code": "10", "crop:name": "Weizen"}]
        )
        result = add_crop_names(path, src)
        assert result.changed
        assert "crop:name_en" not in pq.read_schema(path).names

    def test_an_empty_fields_file(self, tmp_path: Path) -> None:
        path = tmp_path / "fields.parquet"
        gpd.GeoDataFrame({"id": [], "crop:code": []}, geometry=[], crs="EPSG:4326").astype(
            {"id": "string", "crop:code": "string"}
        ).to_parquet(path)
        src = write(tmp_path / "source.parquet", [{"id": "a", "crop:code": "10", "crop:name": "x"}])
        result = add_crop_names(path, src)
        assert (result.rows, result.named) == (0, 0)

    def test_a_subset_of_the_fields_works(self, tmp_path: Path) -> None:
        path = fields(tmp_path, ids=("b",), name="fields_filtered.parquet")
        add_crop_names(path, source(tmp_path))
        assert column(path, "crop:name") == ["Mais"]

    def test_running_again_changes_nothing(self, tmp_path: Path) -> None:
        path = fields(tmp_path)
        add_crop_names(path, source(tmp_path))
        mtime = path.stat().st_mtime_ns
        result = add_crop_names(path, source(tmp_path))
        assert not result.changed
        assert path.stat().st_mtime_ns == mtime

    def test_stale_names_are_replaced(self, tmp_path: Path) -> None:
        path = fields(tmp_path)
        add_crop_names(path, source(tmp_path, a={"crop:name": "old"}))
        result = add_crop_names(path, source(tmp_path))
        assert result.changed
        assert column(path, "crop:name") == [None, "Weizen", "Mais"]

    def test_writes_elsewhere_when_asked(self, tmp_path: Path) -> None:
        path = fields(tmp_path)
        out = tmp_path / "out.parquet"
        add_crop_names(path, source(tmp_path), out)
        assert "crop:name" not in pq.read_schema(path).names
        assert column(out, "crop:name") == [None, "Weizen", "Mais"]

    def test_only_the_name_columns_the_source_has(self, tmp_path: Path) -> None:
        src = write(
            tmp_path / "source.parquet",
            [
                {"id": i, "crop:code": c, "crop:name": n}
                for i, c, n in (("a", "10", "x"), ("b", "20", "y"), ("c", "30", "z"))
            ],
        )
        path = fields(tmp_path)
        result = add_crop_names(path, src)
        assert result.columns == ("crop:name",)
        assert "crop:name_en" not in pq.read_schema(path).names


class TestRefusals:
    """Every case where a field could take a wrong name is refused before writing."""

    def assert_refused(self, path: Path, src: Path, match: str) -> None:
        before = path.read_bytes()
        with pytest.raises(CropNameError, match=match):
            add_crop_names(path, src)
        assert path.read_bytes() == before

    def test_a_field_missing_from_the_source(self, tmp_path: Path) -> None:
        src = write(
            tmp_path / "source.parquet",
            [
                {"id": "a", "crop:code": "10", "crop:name": "Weizen"},
                {"id": "b", "crop:code": "20", "crop:name": "Mais"},
            ],
        )
        self.assert_refused(fields(tmp_path), src, r"1 field\(s\) .* not in .*\['c'\]")

    def test_a_field_with_another_code_in_the_source(self, tmp_path: Path) -> None:
        src = source(tmp_path, b={"crop:code": "21"})
        self.assert_refused(fields(tmp_path), src, r"another crop:code .*\['b'\]")

    def test_a_source_repeating_an_id(self, tmp_path: Path) -> None:
        src = write(
            tmp_path / "source.parquet",
            [
                {"id": "a", "crop:code": "10", "crop:name": "Weizen"},
                {"id": "a", "crop:code": "10", "crop:name": "Weizen"},
            ],
        )
        self.assert_refused(fields(tmp_path, ids=("a",)), src, "repeats 1 field id")

    def test_a_source_without_names(self, tmp_path: Path) -> None:
        src = write(tmp_path / "source.parquet", [{"id": "a", "crop:code": "10"}])
        self.assert_refused(fields(tmp_path, ids=("a",)), src, "has no crop:name column")

    def test_ids_of_another_type(self, tmp_path: Path) -> None:
        path = write(tmp_path / "fields.parquet", [{"id": 7, "crop:code": "10"}])
        src = write(
            tmp_path / "source.parquet", [{"id": "007", "crop:code": "10", "crop:name": "x"}]
        )
        self.assert_refused(path, src, "id is int64 .* but string")

    def test_a_source_with_only_english_names(self, tmp_path: Path) -> None:
        src = write(
            tmp_path / "source.parquet", [{"id": "a", "crop:code": "10", "crop:name_en": "wheat"}]
        )
        self.assert_refused(fields(tmp_path, ids=("a",)), src, "has no crop:name column")

    def test_a_source_field_without_an_id(self, tmp_path: Path) -> None:
        src = write(
            tmp_path / "source.parquet",
            [
                {"id": "a", "crop:code": "10", "crop:name": "x"},
                {"id": None, "crop:code": "10", "crop:name": "y"},
            ],
        )
        self.assert_refused(fields(tmp_path, ids=("a",)), src, "1 field.* without an id")

    def test_crop_codes_of_another_type(self, tmp_path: Path) -> None:
        src = write(
            tmp_path / "source.parquet",
            [{"id": i, "crop:code": c, "crop:name": "x"} for i, c in (("a", 10.0), ("b", 20.0))],
        )
        self.assert_refused(fields(tmp_path, ids=("a", "b")), src, "another crop:code")

    def test_a_failed_write_leaves_no_temp_file(self, tmp_path: Path, monkeypatch) -> None:
        from ftw_dataset_tools.api import crop_names

        def broken(*_args):
            raise OSError("disk full")

        monkeypatch.setattr(crop_names, "_rewrite", broken)
        path = fields(tmp_path)
        before = path.read_bytes()
        with pytest.raises(OSError, match="disk full"):
            add_crop_names(path, source(tmp_path))
        assert path.read_bytes() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ["fields.parquet", "source.parquet"]

    def test_a_fields_file_without_a_crop_code(self, tmp_path: Path) -> None:
        path = write(tmp_path / "fields.parquet", [{"id": "a"}])
        self.assert_refused(path, source(tmp_path), "has no crop:code column")


class TestByCode:
    """by="code" names each field from its crop code, whatever its id."""

    def other_ids(self, tmp_path: Path) -> Path:
        """A fields file from another release: ids the source does not have."""
        rows = [
            {"id": "x", "crop:code": "20"},
            {"id": "y", "crop:code": "10"},
            {"id": "z", "crop:code": "99"},
        ]
        return write(tmp_path / "fields.parquet", rows)

    def test_names_follow_the_code(self, tmp_path: Path) -> None:
        path = self.other_ids(tmp_path)
        result = add_crop_names(path, source(tmp_path), by="code")
        assert column(path, "crop:name") == ["Mais", "Weizen", None]
        assert column(path, "crop:name_en") == ["maize", "wheat", None]
        assert (result.rows, result.named) == (3, 2)

    def test_ids_are_not_needed(self, tmp_path: Path) -> None:
        path = write(tmp_path / "fields.parquet", [{"crop:code": "10"}])
        add_crop_names(path, source(tmp_path), by="code")
        assert column(path, "crop:name") == ["Weizen"]

    def test_a_code_with_two_names_is_refused(self, tmp_path: Path) -> None:
        src = write(
            tmp_path / "source.parquet",
            [
                {"id": "a", "crop:code": "10", "crop:name": "Weizen"},
                {"id": "b", "crop:code": "10", "crop:name": "Hafer"},
            ],
        )
        path = self.other_ids(tmp_path)
        before = path.read_bytes()
        with pytest.raises(CropNameError, match=r"1 crop code\(s\) .* several names.*\['10'\]"):
            add_crop_names(path, src, by="code")
        assert path.read_bytes() == before

    def test_a_code_named_by_many_rows_once(self, tmp_path: Path) -> None:
        src = write(
            tmp_path / "source.parquet",
            [{"id": i, "crop:code": "10", "crop:name": "Weizen"} for i in "abc"],
        )
        path = self.other_ids(tmp_path)
        add_crop_names(path, src, by="code")
        assert column(path, "crop:name") == [None, "Weizen", None]

    def test_codes_of_another_type_match_as_text(self, tmp_path: Path) -> None:
        path = write(tmp_path / "fields.parquet", [{"id": "x", "crop:code": 10}])
        add_crop_names(path, source(tmp_path), by="code")
        assert column(path, "crop:name") == ["Weizen"]

    def test_an_unknown_mode_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="by must be one of"):
            add_crop_names(fields(tmp_path), source(tmp_path), by="name")
