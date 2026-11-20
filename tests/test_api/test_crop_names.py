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

    def test_text_codes_against_numeric_ones(self, tmp_path: Path) -> None:
        src = write(
            tmp_path / "source.parquet",
            [{"id": i, "crop:code": c, "crop:name": "x"} for i, c in (("a", 10.0), ("b", 20.0))],
        )
        self.assert_refused(fields(tmp_path, ids=("a", "b")), src, "crop:code is string")

    def test_a_crop_code_that_differs_as_a_number(self, tmp_path: Path) -> None:
        src = write(
            tmp_path / "source.parquet",
            [{"id": i, "crop:code": c, "crop:name": "x"} for i, c in (("a", 10.5), ("b", 20.0))],
        )
        path = write(
            tmp_path / "fields.parquet",
            [{"id": "a", "crop:code": 10}, {"id": "b", "crop:code": 20}],
        )
        self.assert_refused(path, src, "1 field.* another crop:code")

    def test_a_failed_write_part_way_leaves_no_temp_file(self, tmp_path: Path, monkeypatch) -> None:
        src = source(tmp_path)
        calls = []
        write_table = pq.ParquetWriter.write_table

        def fail_on_second(self, *args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise OSError("disk full")
            return write_table(self, *args, **kwargs)

        path = tmp_path / "fields.parquet"
        gpd.GeoDataFrame(
            [{"id": i, "crop:code": "10"} for i in ("a", "b", "c")],
            geometry=[box(i, 0, i + 1, 1) for i in range(3)],
            crs="EPSG:4326",
        ).to_parquet(path, row_group_size=1)
        before = path.read_bytes()
        monkeypatch.setattr(pq.ParquetWriter, "write_table", fail_on_second)
        with pytest.raises(OSError, match="disk full"):
            add_crop_names(path, src, by="code")
        assert len(calls) == 2, "the first row group was written before the failure"
        assert path.read_bytes() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ["fields.parquet", "source.parquet"]

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
        with pytest.raises(
            CropNameError, match=r"1 crop code\(s\) .* same crop:name on every row.*\['10'\]"
        ):
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

    def test_text_codes_against_numeric_ones_are_refused(self, tmp_path: Path) -> None:
        path = write(tmp_path / "fields.parquet", [{"id": "x", "crop:code": 10, "crop:name": "k"}])
        before = path.read_bytes()
        with pytest.raises(CropNameError, match="crop:code is int64"):
            add_crop_names(path, source(tmp_path), by="code")
        assert path.read_bytes() == before

    @pytest.mark.parametrize(
        ("source_codes", "fields_codes"),
        [([10.0, 20.0], [10, 20]), ([10, 20], [10.0, float("nan")]), ([10.0, 20.0], [10.0, 20.0])],
    )
    def test_whole_floats_match_integers(self, tmp_path, source_codes, fields_codes) -> None:
        src = write(
            tmp_path / "source.parquet",
            [
                {"id": f"s{i}", "crop:code": c, "crop:name": n}
                for i, (c, n) in enumerate(zip(source_codes, ("Weizen", "Mais"), strict=True))
            ],
        )
        path = write(
            tmp_path / "fields.parquet",
            [
                {"id": f"f{i}", "crop:code": c, "crop:name": "old"}
                for i, c in enumerate(fields_codes)
            ],
        )
        add_crop_names(path, src, by="code")
        nan = fields_codes[1] != fields_codes[1]
        assert column(path, "crop:name") == ["Weizen", None if nan else "Mais"]

    def test_whole_floats_match_integers_by_id(self, tmp_path: Path) -> None:
        src = write(
            tmp_path / "source.parquet",
            [
                {"id": "a", "crop:code": 10, "crop:name": "Weizen"},
                {"id": "b", "crop:code": None, "crop:name": None},
            ],
        )
        path = write(
            tmp_path / "fields.parquet",
            [{"id": "a", "crop:code": 10.0}, {"id": "b", "crop:code": float("nan")}],
        )
        add_crop_names(path, src)
        assert column(path, "crop:name") == ["Weizen", None]

    def test_a_code_unnamed_on_some_rows_is_refused(self, tmp_path: Path) -> None:
        src = write(
            tmp_path / "source.parquet",
            [
                {"id": "a", "crop:code": "10", "crop:name": "Weizen"},
                {"id": "b", "crop:code": "10", "crop:name": None},
            ],
        )
        with pytest.raises(CropNameError, match="unnamed on some"):
            add_crop_names(self.other_ids(tmp_path), src, by="code")

    def test_a_code_whose_english_name_differs_is_refused(self, tmp_path: Path) -> None:
        src = write(
            tmp_path / "source.parquet",
            [
                {"id": "a", "crop:code": "10", "crop:name": "Weizen", "crop:name_en": "wheat"},
                {"id": "b", "crop:code": "10", "crop:name": "Weizen", "crop:name_en": "corn"},
            ],
        )
        with pytest.raises(CropNameError, match=r"crop:name, crop:name_en .*\['10'\]"):
            add_crop_names(self.other_ids(tmp_path), src, by="code")

    def test_unnamed_codes_are_logged(self, tmp_path: Path, caplog) -> None:
        add_crop_names(self.other_ids(tmp_path), source(tmp_path), by="code")
        assert "1 crop code(s) on 1 field(s)" in caplog.text
        assert "['99']" in caplog.text

    def test_all_codes_named_logs_nothing(self, tmp_path: Path, caplog) -> None:
        add_crop_names(fields(tmp_path, ids=("a", "b")), source(tmp_path), by="code")
        assert "no name" not in caplog.text

    def test_an_unnamed_code_loses_its_old_name(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "fields.parquet",
            [{"id": "x", "crop:code": "99", "crop:name": "old"}],
        )
        result = add_crop_names(path, source(tmp_path), by="code")
        assert result.changed
        assert column(path, "crop:name") == [None]

    def test_fields_without_a_code_stay_unnamed(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "fields.parquet",
            [{"id": "x", "crop:code": None}, {"id": "y", "crop:code": "10"}],
        )
        add_crop_names(path, source(tmp_path), by="code")
        assert column(path, "crop:name") == [None, "Weizen"]

    def test_nan_is_no_code(self, tmp_path: Path) -> None:
        src = write(
            tmp_path / "source.parquet",
            [
                {"id": "a", "crop:code": float("nan"), "crop:name": "x"},
                {"id": "b", "crop:code": 10.0, "crop:name": "y"},
            ],
        )
        path = write(
            tmp_path / "fields.parquet",
            [{"id": "x", "crop:code": float("nan")}, {"id": "y", "crop:code": 10.0}],
        )
        add_crop_names(path, src, by="code")
        assert column(path, "crop:name") == [None, "y"]

    def test_a_fields_file_without_a_crop_code(self, tmp_path: Path) -> None:
        path = write(tmp_path / "fields.parquet", [{"id": "a"}])
        with pytest.raises(CropNameError, match="has no crop:code column"):
            add_crop_names(path, source(tmp_path), by="code")

    def test_a_source_without_a_crop_code(self, tmp_path: Path) -> None:
        src = write(tmp_path / "source.parquet", [{"id": "a", "crop:name": "x"}])
        with pytest.raises(CropNameError, match="has no crop:code column"):
            add_crop_names(self.other_ids(tmp_path), src, by="code")

    def test_an_unknown_mode_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="by must be one of"):
            add_crop_names(fields(tmp_path), source(tmp_path), by="name")


class TestFileFormat:
    """The rewrite keeps each column's codec and writes names as text."""

    def test_a_name_column_empty_in_the_source_is_text(self, tmp_path: Path) -> None:
        import pyarrow as pa

        path = fields(tmp_path)
        src = source(
            tmp_path,
            a={"crop:name_en": None},
            b={"crop:name_en": None},
        )
        table = pq.read_table(src)
        index = table.schema.get_field_index("crop:name_en")
        table = table.set_column(index, pa.field("crop:name_en", pa.null()), pa.nulls(3))
        pq.write_table(table, src)
        add_crop_names(path, src)
        assert pq.read_schema(path).field("crop:name_en").type == pa.string()

    def test_each_columns_codec_is_kept(self, tmp_path: Path) -> None:
        path = fields(tmp_path)
        table = pq.read_table(path)
        codecs = {"id": "snappy", "crop:code": "zstd", "hcat:code": "none", "geometry": "gzip"}
        pq.write_table(table, path, compression=codecs)
        add_crop_names(path, source(tmp_path))
        group = pq.ParquetFile(path).metadata.row_group(0)
        written = {
            group.column(i).path_in_schema: group.column(i).compression
            for i in range(group.num_columns)
        }
        assert written == {
            "id": "SNAPPY",
            "crop:code": "ZSTD",
            "crop:name": "ZSTD",
            "crop:name_en": "ZSTD",
            "hcat:code": "UNCOMPRESSED",
            "geometry": "GZIP",
        }

    def test_an_uncompressed_file(self, tmp_path: Path) -> None:
        path = fields(tmp_path)
        pq.write_table(pq.read_table(path), path, compression="none")
        add_crop_names(path, source(tmp_path))
        assert column(path, "crop:name") == [None, "Weizen", "Mais"]


def _table_file(path: Path, columns: dict) -> Path:
    """A plain parquet file (no geometry needed by the join) from typed arrow columns."""
    import pyarrow as pa

    pq.write_table(pa.table(columns), path)
    return path


class TestCodeTypes:
    """Codes of different numeric types match only when they are the same number."""

    def test_integers_beyond_float_precision_stay_apart(self, tmp_path: Path) -> None:
        import pyarrow as pa

        big = 2**53
        src = _table_file(
            tmp_path / "source.parquet",
            {
                "id": ["s"],
                "crop:code": pa.array([big], pa.int64()),
                "crop:name": ["BIG"],
            },
        )
        path = _table_file(
            tmp_path / "fields.parquet",
            {"id": ["f"], "crop:code": pa.array([big + 1], pa.int64())},
        )
        add_crop_names(path, src, by="code")
        assert column(path, "crop:name") == [None]

    def test_integers_beyond_float_precision_differ_by_id(self, tmp_path: Path) -> None:
        import pyarrow as pa

        big = 2**53
        src = _table_file(
            tmp_path / "source.parquet",
            {"id": ["a"], "crop:code": pa.array([big], pa.int64()), "crop:name": ["BIG"]},
        )
        path = _table_file(
            tmp_path / "fields.parquet",
            {"id": ["a"], "crop:code": pa.array([big + 1], pa.int64())},
        )
        with pytest.raises(CropNameError, match="another crop:code"):
            add_crop_names(path, src)

    def test_decimals_match_floats_and_integers(self, tmp_path: Path) -> None:
        from decimal import Decimal

        import pyarrow as pa

        src = _table_file(
            tmp_path / "source.parquet",
            {
                "id": ["s1", "s2"],
                "crop:code": pa.array([Decimal("10.50"), Decimal("20.00")], pa.decimal128(5, 2)),
                "crop:name": ["half", "whole"],
            },
        )
        path = _table_file(
            tmp_path / "fields.parquet",
            {"id": ["f1", "f2"], "crop:code": pa.array([10.5, 20.0], pa.float64())},
        )
        add_crop_names(path, src, by="code")
        assert column(path, "crop:name") == ["half", "whole"]

    def test_a_float_too_large_for_an_integer_does_not_crash(self, tmp_path: Path) -> None:
        import pyarrow as pa

        src = _table_file(
            tmp_path / "source.parquet",
            {"id": ["s"], "crop:code": pa.array([1e39]), "crop:name": ["huge"]},
        )
        path = _table_file(tmp_path / "fields.parquet", {"id": ["f"], "crop:code": [1e39]})
        add_crop_names(path, src, by="code")
        assert column(path, "crop:name") == ["huge"]

    @pytest.mark.parametrize(
        ("fields_id", "source_id"),
        [("large_string", "string"), ("int32", "int64")],
    )
    def test_ids_of_the_same_kind_join(self, tmp_path, fields_id, source_id) -> None:
        import pyarrow as pa

        ids = ["1", "2"] if "string" in fields_id else [1, 2]
        src = _table_file(
            tmp_path / "source.parquet",
            {
                "id": pa.array(ids, getattr(pa, source_id)()),
                "crop:code": ["10", "20"],
                "crop:name": ["Weizen", "Mais"],
            },
        )
        path = _table_file(
            tmp_path / "fields.parquet",
            {"id": pa.array(ids[::-1], getattr(pa, fields_id)()), "crop:code": ["20", "10"]},
        )
        add_crop_names(path, src)
        assert column(path, "crop:name") == ["Mais", "Weizen"]


class TestSchemaRepairs:
    """A name column of the wrong type, or missing from an empty file, is a change."""

    def test_a_wrongly_typed_name_column_is_rewritten(self, tmp_path: Path) -> None:
        import pyarrow as pa

        src = _table_file(
            tmp_path / "source.parquet",
            {"id": ["a"], "crop:code": ["10"], "crop:name": pa.array([None], pa.string())},
        )
        path = _table_file(
            tmp_path / "fields.parquet",
            {"id": ["a"], "crop:code": ["10"], "crop:name": pa.array([None], pa.int32())},
        )
        result = add_crop_names(path, src)
        assert result.changed
        assert pq.read_schema(path).field("crop:name").type == pa.string()

    def test_a_file_without_row_groups_gains_the_columns(self, tmp_path: Path) -> None:
        import pyarrow as pa

        src = source(tmp_path)
        path = tmp_path / "fields.parquet"
        schema = pa.schema([("id", pa.string()), ("crop:code", pa.string())])
        pq.ParquetWriter(path, schema).close()
        assert pq.ParquetFile(path).metadata.num_row_groups == 0
        result = add_crop_names(path, src)
        assert result.changed
        assert "crop:name" in pq.read_schema(path).names


class TestFractionalCodes:
    """Codes with a fraction match across float widths and decimals."""

    @pytest.mark.parametrize(
        ("source_type", "fields_type", "value"),
        [
            ("float32", "float64", "0.6"),
            ("float64", "float32", "0.1"),
            ("decimal", "float64", "0.0015"),
            ("decimal", "float32", "0.1"),
            ("decimal", "decimal", "0.00001"),
        ],
    )
    def test_the_same_value_matches(self, tmp_path, source_type, fields_type, value) -> None:
        from decimal import Decimal

        import pyarrow as pa

        def codes(kind: str) -> pa.Array:
            if kind == "decimal":
                return pa.array([Decimal(value)], pa.decimal128(10, 6))
            return pa.array([float(value)], getattr(pa, kind)())

        src = _table_file(
            tmp_path / "source.parquet",
            {"id": ["a"], "crop:code": codes(source_type), "crop:name": ["x"]},
        )
        path = _table_file(
            tmp_path / "fields.parquet", {"id": ["a"], "crop:code": codes(fields_type)}
        )
        add_crop_names(path, src, by="code")
        assert column(path, "crop:name") == ["x"]
        add_crop_names(path, src)
        assert column(path, "crop:name") == ["x"]


class TestDifferentCodesStayApart:
    """Two different numbers never match, however close."""

    @pytest.mark.parametrize(
        ("source_code", "fields_code"),
        [
            ("1234567890.0000000001", "1234567890.0000000002"),
            ("123456789012345.6700000000", "123456789012345.6710000000"),
        ],
    )
    def test_decimals_beyond_float_precision(self, tmp_path, source_code, fields_code) -> None:
        from decimal import Decimal

        import pyarrow as pa

        def codes(value: str) -> pa.Array:
            return pa.array([Decimal(value)], pa.decimal128(38, 10))

        src = _table_file(
            tmp_path / "source.parquet",
            {"id": ["a"], "crop:code": codes(source_code), "crop:name": ["wheat"]},
        )
        path = _table_file(
            tmp_path / "fields.parquet", {"id": ["a"], "crop:code": codes(fields_code)}
        )
        add_crop_names(path, src, by="code")
        assert column(path, "crop:name") == [None]
        with pytest.raises(CropNameError, match="another crop:code"):
            add_crop_names(path, src)

    def test_a_decimal_beyond_float_precision_still_matches_itself(self, tmp_path: Path) -> None:
        from decimal import Decimal

        import pyarrow as pa

        code = pa.array([Decimal("1234567890.0000000001")], pa.decimal128(38, 10))
        src = _table_file(
            tmp_path / "source.parquet", {"id": ["a"], "crop:code": code, "crop:name": ["wheat"]}
        )
        path = _table_file(tmp_path / "fields.parquet", {"id": ["a"], "crop:code": code})
        add_crop_names(path, src, by="code")
        assert column(path, "crop:name") == ["wheat"]

    def test_float32_and_float64_of_different_numbers(self, tmp_path: Path) -> None:
        import pyarrow as pa

        src = _table_file(
            tmp_path / "source.parquet",
            {"id": ["a"], "crop:code": pa.array([0.6], pa.float32()), "crop:name": ["x"]},
        )
        path = _table_file(
            tmp_path / "fields.parquet",
            {"id": ["a"], "crop:code": pa.array([0.6000000238418579], pa.float64())},
        )
        add_crop_names(path, src, by="code")
        assert column(path, "crop:name") == [None]


class TestDecimalEdges:
    """Decimals are compared by their exact digits, never through a float."""

    @staticmethod
    def _decimal_file(path: Path, value: str, precision: int, scale: int, **extra) -> Path:
        from decimal import Decimal

        import pyarrow as pa

        code = pa.array([Decimal(value)], pa.decimal128(precision, scale))
        return _table_file(path, {"id": ["a"], "crop:code": code, **extra})

    @pytest.mark.parametrize(
        ("source", "fields_code"),
        [
            (("82114535650721.59", 38, 2), ("82114535650721.60", 38, 2)),
            (("0.1", 30, 25), ("0.1000000000000000117440512", 30, 25)),
            (
                ("0.000000000000000000000000000001", 38, 30),
                ("0.000000000000000000000000000002", 38, 30),
            ),
        ],
    )
    def test_different_decimals_stay_apart(self, tmp_path, source, fields_code) -> None:
        src = self._decimal_file(tmp_path / "source.parquet", *source, **{"crop:name": ["x"]})
        path = self._decimal_file(tmp_path / "fields.parquet", *fields_code)
        add_crop_names(path, src, by="code")
        assert column(path, "crop:name") == [None]

    @pytest.mark.parametrize(
        ("value", "precision", "scale"),
        [("9.99999999999999999", 18, 17), ("9999999999999999.99", 18, 2), ("-0.10", 5, 2)],
    )
    def test_decimals_at_the_edge_of_their_type_match_themselves(
        self, tmp_path, value, precision, scale
    ) -> None:
        src = self._decimal_file(
            tmp_path / "source.parquet", value, precision, scale, **{"crop:name": ["x"]}
        )
        path = self._decimal_file(tmp_path / "fields.parquet", value, precision, scale)
        add_crop_names(path, src, by="code")
        assert column(path, "crop:name") == ["x"]

    def test_the_same_decimal_at_another_scale_matches(self, tmp_path: Path) -> None:
        src = self._decimal_file(
            tmp_path / "source.parquet", "0.00001", 38, 30, **{"crop:name": ["x"]}
        )
        path = self._decimal_file(tmp_path / "fields.parquet", "0.00001", 10, 6)
        add_crop_names(path, src, by="code")
        assert column(path, "crop:name") == ["x"]


class TestWideAndUnusualTypes:
    """Types DuckDB cannot compare exactly are refused; text views count as text."""

    def test_decimals_wider_than_38_digits_are_refused(self, tmp_path: Path) -> None:
        from decimal import Decimal

        import pyarrow as pa

        code = pa.array([Decimal("0.1")], pa.decimal256(40, 20))
        src = _table_file(
            tmp_path / "source.parquet", {"id": ["a"], "crop:code": code, "crop:name": ["x"]}
        )
        path = _table_file(tmp_path / "fields.parquet", {"id": ["a"], "crop:code": code})
        before = path.read_bytes()
        for by in ("id", "code"):
            with pytest.raises(CropNameError, match="wider than 38 digits"):
                add_crop_names(path, src, by=by)
        assert path.read_bytes() == before

    def test_a_decimal_without_integer_digits_matches_a_float(self, tmp_path: Path) -> None:
        from decimal import Decimal

        import pyarrow as pa

        src = _table_file(
            tmp_path / "source.parquet",
            {
                "id": ["a", "b"],
                "crop:code": pa.array([Decimal("0.500"), Decimal("-0.250")], pa.decimal128(3, 3)),
                "crop:name": ["half", "minus a quarter"],
            },
        )
        path = _table_file(
            tmp_path / "fields.parquet", {"id": ["b", "a"], "crop:code": [-0.25, 0.5]}
        )
        add_crop_names(path, src, by="code")
        assert column(path, "crop:name") == ["minus a quarter", "half"]

    def test_string_view_ids_and_codes_are_text(self, tmp_path: Path) -> None:
        import pyarrow as pa

        src = _table_file(
            tmp_path / "source.parquet",
            {"id": ["a"], "crop:code": ["10"], "crop:name": ["Weizen"]},
        )
        path = _table_file(
            tmp_path / "fields.parquet",
            {
                "id": pa.array(["a"], pa.string_view()),
                "crop:code": pa.array(["10"], pa.string_view()),
            },
        )
        add_crop_names(path, src)
        assert column(path, "crop:name") == ["Weizen"]


class TestNameText:
    """Names are written without surrounding spaces, and a blank name counts as none."""

    def test_padded_names_are_trimmed_by_id(self, tmp_path: Path) -> None:
        src = source(tmp_path, a={"crop:name": "Weizen "}, b={"crop:name": "  Mais"})
        path = fields(tmp_path)
        add_crop_names(path, src)
        assert column(path, "crop:name") == [None, "Weizen", "Mais"]

    def test_blank_names_are_none(self, tmp_path: Path) -> None:
        src = source(tmp_path, a={"crop:name": ""}, b={"crop:name": "   "})
        path = fields(tmp_path)
        result = add_crop_names(path, src)
        assert column(path, "crop:name") == [None, None, None]
        assert result.named == 0

    def test_tabs_line_breaks_and_unicode_spaces_are_trimmed(self, tmp_path: Path) -> None:
        src = source(tmp_path, a={"crop:name": "\tWeizen\n"}, b={"crop:name": "Mais\t\u00a0"})
        path = fields(tmp_path)
        add_crop_names(path, src)
        assert column(path, "crop:name") == [None, "Weizen", "Mais"]

    def test_a_bom_and_zero_width_spaces_are_trimmed(self, tmp_path: Path) -> None:
        src = source(tmp_path, a={"crop:name": "\ufeffWeizen\u200b"}, b={"crop:name": "\u200b"})
        path = fields(tmp_path)
        result = add_crop_names(path, src)
        assert column(path, "crop:name") == [None, "Weizen", None]
        assert result.named == 1

    def test_inner_spaces_are_kept(self, tmp_path: Path) -> None:
        src = source(tmp_path, a={"crop:name": " Winter  Weizen "})
        path = fields(tmp_path)
        add_crop_names(path, src)
        assert column(path, "crop:name")[1] == "Winter  Weizen"

    def test_a_code_padded_on_some_rows_has_one_name(self, tmp_path: Path) -> None:
        """By code, "Weizen " and "Weizen" are the same name, not two to refuse."""
        src = write(
            tmp_path / "source.parquet",
            [
                {"id": "a", "crop:code": "10", "crop:name": "Weizen "},
                {"id": "b", "crop:code": "10", "crop:name": "Weizen"},
            ],
        )
        path = write(tmp_path / "fields.parquet", [{"id": "x", "crop:code": "10"}])
        add_crop_names(path, src, by="code")
        assert column(path, "crop:name") == ["Weizen"]

    def test_a_code_blank_on_some_rows_is_still_refused(self, tmp_path: Path) -> None:
        src = write(
            tmp_path / "source.parquet",
            [
                {"id": "a", "crop:code": "10", "crop:name": " "},
                {"id": "b", "crop:code": "10", "crop:name": "Weizen"},
            ],
        )
        path = write(tmp_path / "fields.parquet", [{"id": "x", "crop:code": "10"}])
        with pytest.raises(CropNameError, match="same"):
            add_crop_names(path, src, by="code")

    def test_trimming_an_existing_name_is_a_change(self, tmp_path: Path) -> None:
        rows = [{"id": "a", "crop:code": "10", "crop:name": "Weizen "}]
        path = write(tmp_path / "fields.parquet", rows)
        src = source(tmp_path, a={"crop:name": "Weizen "})
        result = add_crop_names(path, src)
        assert result.changed
        assert column(path, "crop:name") == ["Weizen"]
        assert not add_crop_names(path, src).changed


class TestTextCodesAsCompared:
    """Text codes are compared trimmed, with a blank code as no code."""

    def test_a_blank_code_matches_an_empty_one_by_id(self, tmp_path: Path) -> None:
        """pt 2023 published " " for no code, the rebuilt edition ""."""
        src = write(tmp_path / "source.parquet", [{"id": "a", "crop:code": "", "crop:name": None}])
        path = write(tmp_path / "fields.parquet", [{"id": "a", "crop:code": " "}])
        result = add_crop_names(path, src)
        assert column(path, "crop:name") == [None]
        assert column(path, "crop:code") == [" "], "the code itself is left as it was"
        assert result.named == 0

    def test_padded_codes_match_by_id_and_by_code(self, tmp_path: Path) -> None:
        src = source(tmp_path, a={"crop:code": "10 "})
        path = write(tmp_path / "fields.parquet", [{"id": "a", "crop:code": " 10"}])
        add_crop_names(path, src)
        assert column(path, "crop:name") == ["Weizen"]
        other = write(tmp_path / "other.parquet", [{"id": "x", "crop:code": "\t10"}])
        add_crop_names(other, src, by="code")
        assert column(other, "crop:name") == ["Weizen"]

    def test_a_code_and_no_code_still_differ(self, tmp_path: Path) -> None:
        src = source(tmp_path, a={"crop:code": " "})
        path = write(tmp_path / "fields.parquet", [{"id": "a", "crop:code": "10"}])
        with pytest.raises(CropNameError, match="another crop:code"):
            add_crop_names(path, src)

    def test_leading_zeros_still_count(self, tmp_path: Path) -> None:
        """'22' and '022' are different text codes; nothing here knows they mean the same."""
        src = source(tmp_path, a={"crop:code": "010"})
        path = write(tmp_path / "fields.parquet", [{"id": "a", "crop:code": "10"}])
        with pytest.raises(CropNameError, match="another crop:code"):
            add_crop_names(path, src)

    def test_more_invisible_padding_is_trimmed_from_names(self, tmp_path: Path) -> None:
        src = source(tmp_path, a={"crop:name": "\x0bWeizen\x85"}, b={"crop:name": "⁠Mais‎"})
        path = fields(tmp_path)
        add_crop_names(path, src)
        assert column(path, "crop:name") == [None, "Weizen", "Mais"]
