"""Tests for style generation."""

import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box


def _chips(tmp_path: Path, *, hcat: bool = True) -> Path:
    data = {
        "id": [f"ftw-33UXP000{i}" for i in range(6)],
        "split": ["train", "train", "train", "val", "test", "test"],
        "field_coverage_pct": [1.0, 12.0, 35.0, 60.0, 88.0, 99.0],
    }
    if hcat:
        data["hcat_dominant_code"] = [
            3301010101,
            3301010101,
            3302000000,
            3302000000,
            3301160000,
            None,
        ]
        data["hcat_dominant_name_en"] = [
            "Winter wheat",
            "Winter wheat",
            "Pasture",
            "Pasture",
            "Soy",
            None,
        ]
    gdf = gpd.GeoDataFrame(data, geometry=[box(i, 0, i + 1, 1) for i in range(6)], crs="EPSG:4326")
    path = tmp_path / "chips.parquet"
    gdf.to_parquet(path)
    return path


def _fields(tmp_path: Path, *, geom_col: str = "geometry") -> Path:
    gdf = gpd.GeoDataFrame(
        {
            "id": [1, 2, 3],
            "hcat:code": [3301010101, 3302000000, 3301010101],
            "hcat:name_en": ["Winter wheat", "Pasture", "Winter wheat"],
            geom_col: [box(0, 0, 2, 1), box(0, 1, 1, 2), box(1, 1, 2, 2)],
        },
        geometry=geom_col,
        crs="EPSG:4326",
    )
    path = tmp_path / "fields.parquet"
    gdf.to_parquet(path)
    return path


def _fill_match_values(style: dict) -> list:
    layer = next(lyr for lyr in style["layers"] if lyr["type"] == "fill")
    expr = layer["paint"]["fill-color"]
    assert expr[0] in ("match", "step")
    return expr


class TestPalette:
    def test_loads_by_code(self) -> None:
        from ftw_dataset_tools.api.styles import load_palette

        palette = load_palette()
        assert palette["3301010101"]["color"] == "#cda737"
        assert len(palette) > 300

    def test_distinct_color_avoids_generic_and_close(self) -> None:
        from ftw_dataset_tools.api.styles import distinct_color

        assert distinct_color("#cc8c32", set()) != "#cc8c32"  # generic arable default
        first = distinct_color("#1f77b4", set())
        assert first == "#1f77b4"
        # too close to an existing legend colour
        assert distinct_color("#1f78b5", {"#1f77b4"}) != "#1f78b5"


class TestMeasurements:
    def test_split_counts_and_coverage_stops(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.styles import coverage_quantiles, split_counts

        chips = _chips(tmp_path)
        assert split_counts(chips) == {"train": 3, "val": 1, "test": 2}
        stops = coverage_quantiles(chips)
        assert stops == sorted(stops) and len(set(stops)) == len(stops) and 1 <= len(stops) <= 4
        assert all(0 < s < 100 for s in stops)

    def test_top_codes_by_count_and_area(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.styles import top_codes

        by_count = top_codes(
            _chips(tmp_path), "hcat_dominant_code", "hcat_dominant_name_en", weight="count"
        )
        assert [c for c, _, _ in by_count] == [3301010101, 3302000000, 3301160000]
        by_area = top_codes(_fields(tmp_path), "hcat:code", "hcat:name_en", weight="area")
        assert [c for c, _, _ in by_area] == [3301010101, 3302000000]
        assert by_area[0][1] == "Winter wheat"

    def test_top_codes_by_area_detects_non_default_geometry_column(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.styles import top_codes

        fields = _fields(tmp_path, geom_col="geom")
        by_area = top_codes(fields, "hcat:code", "hcat:name_en", weight="area")
        assert [c for c, _, _ in by_area] == [3301010101, 3302000000]
        assert by_area[0][1] == "Winter wheat"


class TestBuilders:
    def test_split_style_has_legend_for_present_splits_only(self) -> None:
        from ftw_dataset_tools.api.styles import split_style

        style_id, style, legend = split_style(
            {"train": 3, "test": 2}, tiles_href="./chips.pmtiles", layer="chips"
        )

        assert style_id == "split"
        expr = _fill_match_values(style)
        assert expr[0] == "match" and expr[1] == ["get", "split"]
        assert "train" in expr and "test" in expr and "val" not in expr
        assert [row["label"] for row in legend] == ["train", "test"]
        assert style["sources"]["data"]["url"] == "pmtiles://./chips.pmtiles"

    def test_coverage_style_is_step(self) -> None:
        from ftw_dataset_tools.api.styles import coverage_style

        _, style, legend = coverage_style(
            [10.0, 40.0, 80.0], tiles_href="./chips.pmtiles", layer="chips"
        )

        expr = _fill_match_values(style)
        assert expr[0] == "step" and expr[1] == ["get", "field_coverage_pct"]
        assert list(expr[3::2]) == [10.0, 40.0, 80.0]
        assert len(legend) == 4

    def test_crop_styles_use_palette_and_other(self) -> None:
        from ftw_dataset_tools.api.styles import crops_style, dominant_crop_style

        rows = [(3301010101, "Winter wheat", 2.0), (3302000000, "Pasture", 1.0)]
        _, dom, legend = dominant_crop_style(rows, tiles_href="./chips.pmtiles", layer="chips")
        expr = _fill_match_values(dom)
        assert expr[1][1] == ["get", "hcat_dominant_code"]  # inner match on the code
        assert 3301010101 in expr[1] and "#cda737" in expr
        assert legend[-1]["label"] == "Other"

        _, crops, _ = crops_style(rows, tiles_href="./fields.pmtiles", layer="fields")
        assert _fill_match_values(crops)[1][1] == ["get", "hcat:code"]

    def test_outline_style_has_no_fill_legend(self) -> None:
        from ftw_dataset_tools.api.styles import outline_style

        style_id, style, legend = outline_style("lu", tiles_href="./fields.pmtiles", layer="fields")

        assert style_id == "outline" and legend == []
        assert all(
            lyr["type"] != "fill" or "match" not in str(lyr["paint"]) for lyr in style["layers"]
        )


class TestWriteStyles:
    def test_writes_applicable_styles_and_marks_default(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.styles import write_styles

        results = write_styles(
            tmp_path,
            "lu",
            _chips(tmp_path),
            _fields(tmp_path),
            chips_tiles="./chips.pmtiles",
            fields_tiles="./fields.pmtiles",
        )

        ids = [r.style_id for r in results]
        assert ids == ["split", "field-coverage", "dominant-crop", "crops", "outline"]
        assert [r.default for r in results] == [True, False, False, False, False]
        for r in results:
            assert r.path == tmp_path / "styles" / f"{r.style_id}.json"
            assert json.loads(r.path.read_text())["version"] == 8

    def test_skips_crop_styles_without_hcat_and_fields_styles_without_tiles(
        self, tmp_path: Path
    ) -> None:
        from ftw_dataset_tools.api.styles import write_styles

        results = write_styles(
            tmp_path,
            "lu",
            _chips(tmp_path, hcat=False),
            _fields(tmp_path),
            chips_tiles="./chips.pmtiles",
            fields_tiles=None,
        )

        assert [r.style_id for r in results] == ["split", "field-coverage"]
