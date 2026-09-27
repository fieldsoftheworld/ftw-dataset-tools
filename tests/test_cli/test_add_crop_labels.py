"""Tests for the add-crop-labels CLI command."""

from pathlib import Path

import duckdb
import geopandas as gpd
from click.testing import CliRunner
from shapely.geometry import box

from ftw_dataset_tools.cli import cli


def _fields(tmp_path: Path) -> Path:
    path = tmp_path / "fields.parquet"
    gpd.GeoDataFrame(
        {"id": [1, 2], "crop:code": ["A", "B"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    ).to_parquet(path)
    return path


class TestAddCropLabelsCommand:
    def test_default_output_with_mapping(self, tmp_path: Path) -> None:
        mapping = tmp_path / "m.csv"
        mapping.write_text("original_code,HCAT3_code\nA,3301010100\nB,3301030000\n")

        result = CliRunner().invoke(
            cli, ["add-crop-labels", str(_fields(tmp_path)), "--mapping", str(mapping)]
        )

        assert result.exit_code == 0, result.output
        out = tmp_path / "fields_hcat.parquet"
        codes = duckdb.connect().execute(f"""SELECT "hcat:code" FROM '{out}' ORDER BY id""")
        assert codes.fetchall() == [(3301010100,), (3301030000,)]

    def test_template_only(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            cli, ["add-crop-labels", str(_fields(tmp_path)), "--template-only"]
        )

        assert result.exit_code == 0, result.output
        assert (tmp_path / "fields_hcat_crop_mapping_template.csv").exists()
        assert not (tmp_path / "fields_hcat.parquet").exists()

    def test_strict_error_is_clean(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(cli, ["add-crop-labels", str(_fields(tmp_path)), "--strict"])

        assert result.exit_code != 0
        assert "Traceback" not in result.output
