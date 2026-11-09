"""Tests for the create-ftw-grid CLI command."""

from pathlib import Path

from click.testing import CliRunner

from ftw_dataset_tools.cli import cli


class TestCreateFTWGridCommand:
    """Tests for create-ftw-grid command."""

    def test_help(self) -> None:
        """Test --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-ftw-grid", "--help"])
        assert result.exit_code == 0

    def test_missing_input(self) -> None:
        """Test error for missing input argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-ftw-grid"])
        assert result.exit_code != 0

    def test_valid_input(self, sample_mgrs_1km_geoparquet: Path, tmp_path: Path) -> None:
        """Test create-ftw-grid with valid MGRS input."""
        output_file = tmp_path / "ftw_grid.parquet"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "create-ftw-grid",
                str(sample_mgrs_1km_geoparquet),
                "--output",
                str(output_file),
                "--km-size",
                "2",
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()

    def test_km_size_option(self, sample_mgrs_1km_geoparquet: Path, tmp_path: Path) -> None:
        """Test --km-size option with valid value."""
        output_file = tmp_path / "ftw_grid_5km.parquet"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "create-ftw-grid",
                str(sample_mgrs_1km_geoparquet),
                "--output",
                str(output_file),
                "--km-size",
                "5",
            ],
        )
        assert result.exit_code == 0

    def test_invalid_km_size(self, sample_mgrs_1km_geoparquet: Path, tmp_path: Path) -> None:
        """Test --km-size option with invalid value."""
        output_file = tmp_path / "ftw_grid_3km.parquet"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "create-ftw-grid",
                str(sample_mgrs_1km_geoparquet),
                "--output",
                str(output_file),
                "--km-size",
                "3",
            ],
        )
        assert result.exit_code != 0
        assert "km_size must divide 100" in result.output


def _write_cells(path: Path, mgrs: list[str]) -> None:
    """1km cells in 32U with empty EASTING/NORTHING, so ids come from the MGRS code."""
    import geopandas as gpd
    from shapely.geometry import box

    gpd.GeoDataFrame(
        {"MGRS": mgrs, "GZD": ["32U"] * len(mgrs), "EASTING": [None] * len(mgrs)},
        geometry=[box(9.0, 48.0, 9.01, 48.01)] * len(mgrs),
        crs="EPSG:4326",
    ).assign(NORTHING=[None] * len(mgrs)).to_parquet(path)


class TestCreateFTWGridBadCoordinates:
    """Cells without easting/northing are refused, and the CLI says so."""

    def test_single_file_error_is_clean(self, tmp_path: Path) -> None:
        src = tmp_path / "mgrs.parquet"
        _write_cells(src, ["32UPE0000", "32UPE"])

        result = CliRunner().invoke(cli, ["create-ftw-grid", str(src), "-o", str(tmp_path / "o")])

        assert result.exit_code == 1
        assert "no valid 100km square or easting/northing" in result.output
        assert "Traceback" not in result.output

    def test_folder_lists_skipped_files(self, tmp_path: Path) -> None:
        (tmp_path / "in" / "a").mkdir(parents=True)
        (tmp_path / "in" / "b").mkdir()
        _write_cells(tmp_path / "in" / "a" / "good.parquet", ["32UPE0000"])
        _write_cells(tmp_path / "in" / "b" / "bad.parquet", ["32UPE"])

        result = CliRunner().invoke(
            cli, ["create-ftw-grid", str(tmp_path / "in"), "-o", str(tmp_path / "out")]
        )

        assert result.exit_code == 0
        assert "Skipped 1 file(s)" in result.output
        assert "bad.parquet" in result.output
