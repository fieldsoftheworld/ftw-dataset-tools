"""CLI tests for the ftwd inspect-fields command."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import geopandas as gpd
from click.testing import CliRunner
from shapely.geometry import box

from ftw_dataset_tools.commands.inspect_fields import inspect_fields

if TYPE_CHECKING:
    from pathlib import Path


def _fields(tmp_path: Path) -> Path:
    gdf = gpd.GeoDataFrame(
        {"id": ["a", "b", "c"], "crop": ["wheat", "wheat", "maize"]},
        geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2), box(2, 2, 3, 3)],
        crs="EPSG:4326",
    )
    path = tmp_path / "fields.parquet"
    gdf.to_parquet(path)
    return path


class TestInspectFieldsCommand:
    def test_help(self) -> None:
        result = CliRunner().invoke(inspect_fields, ["--help"])
        assert result.exit_code == 0
        assert "Summarize a fields Parquet file" in result.output

    def test_text_output(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(inspect_fields, [str(_fields(tmp_path))])
        assert result.exit_code == 0, result.output
        assert "Rows: 3" in result.output
        assert "crop" in result.output
        assert "wheat" in result.output
        assert "Class-filter candidates:" in result.output

    def test_json_output(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(inspect_fields, [str(_fields(tmp_path)), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["num_rows"] == 3
        assert "crop" in data["class_filter_candidates"]

    def test_missing_file_errors(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(inspect_fields, [str(tmp_path / "nope.parquet")])
        assert result.exit_code != 0
