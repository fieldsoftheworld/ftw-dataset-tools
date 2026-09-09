"""Tests for PMTiles building."""

import shutil
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

needs_tippecanoe = pytest.mark.skipif(
    shutil.which("tippecanoe") is None, reason="tippecanoe not installed"
)


def _chips(tmp_path: Path) -> Path:
    gdf = gpd.GeoDataFrame(
        {
            "id": ["ftw-33UXP0001", "ftw-33UXP0002"],
            "split": ["train", "test"],
            "field_coverage_pct": [12.5, 80.0],
        },
        geometry=[box(10.0, 50.0, 10.02, 50.02), box(10.02, 50.0, 10.04, 50.02)],
        crs="EPSG:4326",
    )
    path = tmp_path / "chips.parquet"
    gdf.to_parquet(path)
    return path


class TestExportGeojsonseq:
    def test_keeps_only_present_attributes(self, tmp_path: Path) -> None:
        import json

        from ftw_dataset_tools.api.tiles import CHIPS_TILES, export_geojsonseq

        out = tmp_path / "chips.geojsonseq"
        kept = export_geojsonseq(_chips(tmp_path), out, CHIPS_TILES.attributes)

        assert kept == ["id", "split", "field_coverage_pct"]
        lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        assert len(lines) == 2
        assert set(lines[0]["properties"]) == {"id", "split", "field_coverage_pct"}
        assert lines[0]["geometry"]["type"] == "Polygon"

    def test_reprojects_to_4326(self, tmp_path: Path) -> None:
        import json

        from ftw_dataset_tools.api.tiles import export_geojsonseq

        gdf = gpd.GeoDataFrame(
            {"id": [1]}, geometry=[box(500000, 5500000, 500100, 5500100)], crs="EPSG:32633"
        )
        src = tmp_path / "utm.parquet"
        gdf.to_parquet(src)
        out = tmp_path / "utm.geojsonseq"

        export_geojsonseq(src, out, ("id",))

        lon, lat = json.loads(out.read_text().splitlines()[0])["geometry"]["coordinates"][0][0]
        assert 14 < lon < 16 and 49 < lat < 50

    def test_nan_attribute_serialises_as_null(self, tmp_path: Path) -> None:
        import json
        import math

        from ftw_dataset_tools.api.tiles import export_geojsonseq

        gdf = gpd.GeoDataFrame(
            {"id": ["a", "b"], "field_coverage_pct": [math.nan, 5.0]},
            geometry=[box(10.0, 50.0, 10.02, 50.02), box(10.02, 50.0, 10.04, 50.02)],
            crs="EPSG:4326",
        )
        src = tmp_path / "nan.parquet"
        gdf.to_parquet(src)
        out = tmp_path / "nan.geojsonseq"

        export_geojsonseq(src, out, ("id", "field_coverage_pct"))

        raw_lines = [line for line in out.read_text().splitlines() if line.strip()]
        assert "NaN" not in raw_lines[0]
        first = json.loads(raw_lines[0])
        assert first["properties"]["field_coverage_pct"] is None


class TestBuildPmtiles:
    def test_raises_without_tippecanoe(self, tmp_path: Path, monkeypatch) -> None:
        from ftw_dataset_tools.api import tiles

        monkeypatch.setattr(tiles.shutil, "which", lambda _name: None)
        with pytest.raises(RuntimeError, match="tippecanoe"):
            tiles.build_pmtiles(_chips(tmp_path), tmp_path / "chips.pmtiles", tiles.CHIPS_TILES)

    @needs_tippecanoe
    def test_builds_archive_and_cleans_up(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.tiles import CHIPS_TILES, build_pmtiles

        out = build_pmtiles(_chips(tmp_path), tmp_path / "chips.pmtiles", CHIPS_TILES)

        assert out.exists() and out.stat().st_size > 0
        assert out.read_bytes()[:7] == b"PMTiles"
        assert not list(tmp_path.glob("*.geojsonseq"))
