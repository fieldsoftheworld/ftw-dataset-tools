"""Tests for the add-land-cover CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import pytest
from click.testing import CliRunner

from ftw_dataset_tools.api import land_cover
from ftw_dataset_tools.cli import cli
from tests.test_api.test_land_cover import LocalSource, two_chip_setup

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def local_io(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Chips over a local tile, with IO swapped for a source that has 2017 and 2021 maps."""
    chips, tile = two_chip_setup(tmp_path)
    source = LocalSource({"33U": {2017: str(tile), 2021: str(tile)}})
    monkeypatch.setattr(land_cover, "IOLulcSource", lambda: source)
    return chips


class TestAddLandCoverCommand:
    def test_help(self) -> None:
        result = CliRunner().invoke(cli, ["add-land-cover", "--help"])

        assert result.exit_code == 0
        assert "CHIPS_FILE" in result.output
        assert "--year" in result.output
        assert "--only stac" in result.output

    def test_writes_the_columns_and_prints_the_summary(self, local_io: Path) -> None:
        result = CliRunner().invoke(cli, ["add-land-cover", str(local_io), "--year", "2021"])

        assert result.exit_code == 0, result.output
        assert "Land cover: 2/2 chips from IO 2021" in result.output
        assert "Warning" not in result.output
        gdf = gpd.read_parquet(local_io)
        assert set(land_cover.OUTPUT_COLUMNS) <= set(gdf.columns)
        assert gdf["landcover_year_exact"].tolist() == [True, True]

    def test_warns_when_the_nearest_year_is_used(self, local_io: Path) -> None:
        result = CliRunner().invoke(cli, ["add-land-cover", str(local_io), "--year", "2025"])

        assert result.exit_code == 0, result.output
        assert "Warning: 2 chips use the nearest IO year" in result.output
        assert "(2 nearest-year for dataset year 2025)" in result.output
        assert gpd.read_parquet(local_io)["landcover_year"].tolist() == [2021, 2021]

    def test_year_is_required(self, local_io: Path) -> None:
        result = CliRunner().invoke(cli, ["add-land-cover", str(local_io)])

        assert result.exit_code != 0
        assert "--year" in result.output

    def test_missing_file(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            cli, ["add-land-cover", str(tmp_path / "nope.parquet"), "--year", "2021"]
        )

        assert result.exit_code != 0

    def test_missing_id_column_is_a_clean_error(self, local_io: Path) -> None:
        result = CliRunner().invoke(
            cli, ["add-land-cover", str(local_io), "--year", "2021", "--id-col", "chip"]
        )

        assert result.exit_code == 1
        assert "Error: Chips file has no 'chip' column" in result.output
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_unreachable_source_is_a_clean_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        chips, _tile = two_chip_setup(tmp_path)

        class Offline:
            def index(self, _bbox: tuple) -> dict:
                raise OSError("Planetary Computer is unreachable")

        monkeypatch.setattr(land_cover, "IOLulcSource", Offline)

        result = CliRunner().invoke(cli, ["add-land-cover", str(chips), "--year", "2021"])

        assert result.exit_code == 1
        assert "Planetary Computer is unreachable" in result.output
