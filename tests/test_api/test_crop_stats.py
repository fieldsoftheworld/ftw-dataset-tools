"""Tests for per-chip HCAT crop composition."""

from pathlib import Path

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
        import duckdb

        from ftw_dataset_tools.api.crop_stats import add_crop_stats

        chips = _chips(tmp_path)
        result = add_crop_stats(chips, _fields(tmp_path))

        assert result.skipped is False
        assert result.chips_total == 2
        assert result.chips_with_crops == 1
        assert result.distinct_codes == 2

        con = duckdb.connect()
        rows = con.execute(
            f"SELECT id, hcat_dominant_code, hcat_dominant_name_en, hcat_dominant_pct, hcat_top "
            f"FROM read_parquet('{chips}') ORDER BY id"
        ).fetchall()
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
