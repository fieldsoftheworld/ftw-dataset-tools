"""Tests for per-chip HCAT crop composition."""

from pathlib import Path

import duckdb
import geopandas as gpd
import pytest
from shapely.geometry import box


def _fields(tmp_path: Path, *, names: bool = True, hcat: bool = True) -> Path:
    data = {"id": [1, 2, 3, 4]}
    if hcat:
        data["hcat:code"] = [3301010101, 3302000000, 3301010101, 3301160000]
        if names:
            data["hcat:name_en"] = ["Winter wheat", "Pasture", "Winter wheat", "Soy"]
    gdf = gpd.GeoDataFrame(
        data,
        # chip A = box(0,0,2,2): wheat 1.0 + pasture 1.0 + wheat 1.0 (area 3 of 4); chip B empty
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2), box(10, 10, 11, 11)],
        crs="EPSG:4326",
    )
    path = tmp_path / "fields.parquet"
    gdf.to_parquet(path)
    return path


def _chips(tmp_path: Path) -> Path:
    gdf = gpd.GeoDataFrame(
        {"id": ["ftw-33UXP0001", "ftw-33UXP0002"], "field_coverage_pct": [75.0, 0.0]},
        geometry=[box(0, 0, 2, 2), box(5, 5, 6, 6)],
        crs="EPSG:4326",
    )
    path = tmp_path / "chips.parquet"
    gdf.to_parquet(path)
    return path


def _write(path: Path, data: dict, geometries: list, *, bbox: bool = False) -> Path:
    """Write a small GeoParquet fixture, optionally through the bbox-adding writer."""
    gdf = gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:4326")
    if bbox:
        from ftw_dataset_tools.api.geo import write_geoparquet

        write_geoparquet(path, gdf=gdf)
    else:
        gdf.to_parquet(path)
    return path


def _read(chips: Path) -> list[tuple]:
    con = duckdb.connect()
    try:
        return con.execute(
            "SELECT id, hcat_dominant_code, hcat_dominant_name_en, hcat_dominant_pct, hcat_top "
            f"FROM read_parquet('{chips}') ORDER BY id"
        ).fetchall()
    finally:
        con.close()


class TestDetectHcatColumns:
    def test_code_and_name_en(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.crop_stats import detect_hcat_columns

        assert detect_hcat_columns(_fields(tmp_path)) == ("hcat:code", "hcat:name_en")

    def test_code_only(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.crop_stats import detect_hcat_columns

        assert detect_hcat_columns(_fields(tmp_path, names=False)) == ("hcat:code", None)

    def test_none_without_code(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.crop_stats import detect_hcat_columns

        assert detect_hcat_columns(_fields(tmp_path, hcat=False)) is None


class TestAddCropStats:
    def test_composition_is_area_weighted(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.crop_stats import add_crop_stats

        chips = _chips(tmp_path)
        result = add_crop_stats(chips, _fields(tmp_path))

        assert result.skipped is False
        assert result.chips_total == 2
        assert result.chips_with_crops == 1
        assert result.distinct_codes == 2

        rows = _read(chips)
        assert rows[0][1] == 3301010101
        assert rows[0][2] == "Winter wheat"
        assert rows[0][3] == pytest.approx(66.67, abs=0.01)
        assert [entry["code"] for entry in rows[0][4]] == [3301010101, 3302000000]
        assert rows[0][4][1]["pct"] == pytest.approx(33.33, abs=0.01)
        assert rows[1][1] is None and rows[1][4] is None

    def test_geoparquet_metadata_survives(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.crop_stats import add_crop_stats
        from ftw_dataset_tools.api.geo import detect_crs, detect_geometry_column

        chips = _chips(tmp_path)
        add_crop_stats(chips, _fields(tmp_path))

        assert detect_geometry_column(chips) == "geometry"
        assert detect_crs(chips, "geometry").authority_code == "EPSG:4326"
        assert gpd.read_parquet(chips).shape[0] == 2

    def test_rerun_replaces_columns(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.crop_stats import add_crop_stats

        chips = _chips(tmp_path)
        add_crop_stats(chips, _fields(tmp_path))
        add_crop_stats(chips, _fields(tmp_path))

        cols = list(gpd.read_parquet(chips).columns)
        assert cols.count("hcat_dominant_code") == 1

    def test_skips_without_hcat(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.crop_stats import add_crop_stats

        chips = _chips(tmp_path)
        messages: list[str] = []
        result = add_crop_stats(chips, _fields(tmp_path, hcat=False), on_progress=messages.append)

        assert result.skipped is True
        assert "hcat:code" in (result.reason or "")
        assert any("skipping crop composition" in m for m in messages)
        assert "hcat_dominant_code" not in gpd.read_parquet(chips).columns

    def test_name_falls_back_to_hcat_name(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.crop_stats import add_crop_stats

        fields = _fields(tmp_path, names=False)
        gdf = gpd.read_parquet(fields)
        gdf["hcat:name"] = [
            "winter_common_soft_wheat",
            "pasture",
            "winter_common_soft_wheat",
            "soy",
        ]
        gdf.to_parquet(fields)
        chips = _chips(tmp_path)

        add_crop_stats(chips, fields)

        assert (
            gpd.read_parquet(chips)["hcat_dominant_name_en"].iloc[0] == "winter_common_soft_wheat"
        )

    def test_name_is_the_lowest_of_conflicting_names(self, tmp_path: Path) -> None:
        """Two rows sharing a code must not make the published name arbitrary."""
        from ftw_dataset_tools.api.crop_stats import add_crop_stats

        fields = _write(
            tmp_path / "fields.parquet",
            {"hcat:code": [1, 1], "hcat:name_en": ["Beta wheat", "Alpha wheat"]},
            [box(0, 0, 1, 2), box(1, 0, 2, 2)],
        )
        chips = _chips(tmp_path)

        add_crop_stats(chips, fields)

        assert _read(chips)[0][2] == "Alpha wheat"


class TestCodeCasting:
    def test_varchar_codes_are_cast(self, tmp_path: Path) -> None:
        """fiboa types hcat:code as a string; numeric strings must still work."""
        from ftw_dataset_tools.api.crop_stats import add_crop_stats

        fields = _write(
            tmp_path / "fields.parquet",
            {"hcat:code": ["3301010101", "3302000000"]},
            [box(0, 0, 1, 2), box(1, 0, 2, 2)],
        )
        chips = _chips(tmp_path)

        result = add_crop_stats(chips, fields)

        assert result.skipped is False
        assert _read(chips)[0][1] == 3301010101

    def test_non_numeric_codes_skip_without_raising(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.crop_stats import add_crop_stats

        fields = _write(
            tmp_path / "fields.parquet",
            {"hcat:code": ["AAA", "BBB"]},
            [box(0, 0, 1, 2), box(1, 0, 2, 2)],
        )
        chips = _chips(tmp_path)
        messages: list[str] = []

        result = add_crop_stats(chips, fields, on_progress=messages.append)

        assert result.skipped is True
        assert result.reason == "hcat:code has no numeric values"
        assert result.chips_total == 2
        assert any("skipping crop composition" in m for m in messages)
        assert "hcat_dominant_code" not in gpd.read_parquet(chips).columns

    def test_uncastable_rows_count_as_uncoded(self, tmp_path: Path) -> None:
        """One bad code must not discard the codes that do parse."""
        from ftw_dataset_tools.api.crop_stats import add_crop_stats

        fields = _write(
            tmp_path / "fields.parquet",
            {"hcat:code": ["3301010101", "not-a-code"]},
            [box(0, 0, 1, 2), box(1, 0, 2, 2)],
        )
        chips = _chips(tmp_path)

        add_crop_stats(chips, fields)

        row = _read(chips)[0]
        assert row[1] == 3301010101
        # half the chip's field area is uncoded, so the coded half is 50%
        assert row[3] == pytest.approx(50.0, abs=0.01)


class TestDenominator:
    def test_uncoded_fields_stay_in_the_denominator(self, tmp_path: Path) -> None:
        """pct is a share of all field-covered area, per spec 3.5."""
        from ftw_dataset_tools.api.crop_stats import add_crop_stats

        fields = _write(
            tmp_path / "fields.parquet",
            {"hcat:code": ["3301010101", None], "hcat:name_en": ["Winter wheat", None]},
            [box(0, 0, 1, 2), box(1, 0, 2, 2)],
        )
        chips = _chips(tmp_path)

        add_crop_stats(chips, fields)

        row = _read(chips)[0]
        assert row[1] == 3301010101
        assert row[3] == pytest.approx(50.0, abs=0.01)
        assert row[4][0]["pct"] == pytest.approx(50.0, abs=0.01)

    def test_overlapping_fields_are_not_double_counted(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.crop_stats import add_crop_stats

        fields = _write(
            tmp_path / "fields.parquet",
            {"hcat:code": [1, 1, 2]},
            [box(0, 0, 1, 2), box(0, 0, 1, 2), box(1, 0, 2, 2)],
        )
        chips = _chips(tmp_path)

        add_crop_stats(chips, fields)

        row = _read(chips)[0]
        assert row[3] == pytest.approx(50.0, abs=0.01)
        assert [entry["pct"] for entry in row[4]] == [
            pytest.approx(50.0, abs=0.01),
            pytest.approx(50.0, abs=0.01),
        ]

    def test_top_n_truncates_without_renormalising(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.crop_stats import add_crop_stats

        fields = _write(
            tmp_path / "fields.parquet",
            {"hcat:code": [1, 2, 3]},
            [box(0, 0, 2, 1), box(0, 1, 1, 2), box(1, 1, 2, 2)],
        )
        chips = _chips(tmp_path)

        result = add_crop_stats(chips, fields, top_n=2)

        row = _read(chips)[0]
        assert [entry["code"] for entry in row[4]] == [1, 2]
        # code 1 covers 2 of the 4 covered units; the top-2 basis would be 66.7
        assert row[3] == pytest.approx(50.0, abs=0.01)
        assert result.distinct_codes == 3

    def test_distinct_codes_counts_codes_below_the_top_list(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.crop_stats import add_crop_stats

        codes = [1, 2, 3, 4, 5, 6, 7]
        geoms = [box(i * 0.25, 0, (i + 1) * 0.25, 2) for i in range(len(codes))]
        fields = _write(tmp_path / "fields.parquet", {"hcat:code": codes}, geoms)
        chips = _chips(tmp_path)

        result = add_crop_stats(chips, fields)

        assert len(_read(chips)[0][4]) == 5
        assert result.distinct_codes == 7

    def test_field_spanning_two_chips_is_clipped_per_chip(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.crop_stats import add_crop_stats

        chips = _write(
            tmp_path / "chips.parquet",
            {"id": ["ftw-33UXP0001", "ftw-33UXP0002"]},
            [box(0, 0, 2, 2), box(2, 0, 4, 2)],
        )
        fields = _write(
            tmp_path / "fields.parquet",
            {"hcat:code": [1, 2, 3]},
            # code 1 straddles both chips; codes 2 and 3 sit in one chip each
            [box(1, 0, 3, 1), box(0, 1, 2, 2), box(3, 0, 4, 2)],
        )

        add_crop_stats(chips, fields)

        rows = _read(chips)
        assert rows[0][1] == 2 and rows[0][3] == pytest.approx(66.67, abs=0.01)
        assert rows[1][1] == 3 and rows[1][3] == pytest.approx(66.67, abs=0.01)
        assert [e["pct"] for e in rows[0][4]] == [
            pytest.approx(66.67, abs=0.01),
            pytest.approx(33.33, abs=0.01),
        ]


class TestPathsAndBbox:
    def _ranked_query(self, **bbox_cols: str | None) -> str:
        from ftw_dataset_tools.api.crop_stats import build_ranked_query

        return build_ranked_query(
            "chips_table",
            "fields_table",
            chips_geom_col="geometry",
            fields_geom_col="geometry",
            chips_id_col="id",
            code_col="hcat:code",
            name_col=None,
            **bbox_cols,
        )

    def test_join_is_bbox_prefiltered_when_both_files_have_bbox(self) -> None:
        query = self._ranked_query(chips_bbox_col="bbox", fields_bbox_col="bbox")

        assert 'g."bbox".xmin <= f."bbox".xmax' in query
        assert "ST_Intersects" in query

    def test_join_falls_back_without_bbox(self) -> None:
        query = self._ranked_query(chips_bbox_col="bbox", fields_bbox_col=None)

        assert ".xmin" not in query
        assert "ST_Intersects" in query

    def test_bbox_columns_are_detected_and_preserved(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.crop_stats import add_crop_stats
        from ftw_dataset_tools.api.field_stats import detect_bbox_column

        chips = _write(
            tmp_path / "chips.parquet",
            {"id": ["ftw-33UXP0001"]},
            [box(0, 0, 2, 2)],
            bbox=True,
        )
        fields = _write(
            tmp_path / "fields.parquet",
            {"hcat:code": [1, 2]},
            [box(0, 0, 1, 2), box(1, 0, 2, 2)],
            bbox=True,
        )
        con = duckdb.connect()
        assert detect_bbox_column(con, chips, "geometry") == "bbox"
        assert detect_bbox_column(con, fields, "geometry") == "bbox"

        result = add_crop_stats(chips, fields)

        assert result.chips_with_crops == 1
        assert _read(chips)[0][3] == pytest.approx(50.0, abs=0.01)
        assert "bbox" in [
            row[0] for row in con.execute(f"DESCRIBE SELECT * FROM '{chips}'").fetchall()
        ]
        con.close()

    def test_path_containing_a_single_quote(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.crop_stats import add_crop_stats

        odd_dir = tmp_path / "o'brien"
        odd_dir.mkdir()
        chips = _chips(odd_dir)
        fields = _fields(odd_dir)

        result = add_crop_stats(chips, fields)

        assert result.chips_with_crops == 1
