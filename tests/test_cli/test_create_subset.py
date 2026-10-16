"""Tests for the create-subset and create-splits CLI commands."""

from pathlib import Path

import geopandas as gpd
from click.testing import CliRunner
from shapely.geometry import box

from ftw_dataset_tools.cli import cli


def _chips(path: Path, ids: list[str]) -> Path:
    gpd.GeoDataFrame(
        {"id": ids}, geometry=[box(i, 0, i + 1, 1) for i in range(len(ids))], crs="EPSG:4326"
    ).to_parquet(path)
    return path


class TestCreateSubset:
    def test_keeps_a_scale(self, tmp_path: Path) -> None:
        ids = [f"ftw-33UXP{e:02d}{n:02d}" for e in range(0, 36, 2) for n in range(0, 36, 2)]
        path = _chips(tmp_path / "chips.parquet", ids)

        result = CliRunner().invoke(cli, ["create-subset", str(path), "--percent", "50"])

        assert result.exit_code == 0, result.output
        kept = len(gpd.read_parquet(path))
        assert f"Kept {kept:,} of 324 chips" in result.output
        assert 0 < kept < 324

    def test_invalid_ids_fail_cleanly(self, tmp_path: Path) -> None:
        path = _chips(tmp_path / "chips.parquet", ["bad-id"])

        result = CliRunner().invoke(cli, ["create-subset", str(path), "--percent", "50"])

        assert result.exit_code != 0
        assert "Invalid chip ID format" in result.output

    def test_percent_out_of_range(self, tmp_path: Path) -> None:
        path = _chips(tmp_path / "chips.parquet", ["ftw-33UXP0000"])

        result = CliRunner().invoke(cli, ["create-subset", str(path), "--percent", "150"])

        assert result.exit_code != 0


def test_create_splits_km_size(tmp_path: Path) -> None:
    path = _chips(tmp_path / "chips.parquet", ["ftw-33UXP0000", "ftw-33UXP1200"])

    result = CliRunner().invoke(
        cli,
        [
            "create-splits",
            str(path),
            "--split-type",
            "block3x3",
            "--split-percents",
            "50",
            "0",
            "50",
            "--km-size",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert set(gpd.read_parquet(path)["split"]) == {"train", "test"}


def test_create_subset_sub_km_ids(tmp_path: Path) -> None:
    ids = [f"ftw-33UXP{e:03d}{n:03d}" for e in range(0, 90, 5) for n in range(0, 90, 5)]
    path = _chips(tmp_path / "chips.parquet", ids)

    result = CliRunner().invoke(
        cli, ["create-subset", str(path), "--percent", "50", "--km-size", "0.5"]
    )

    assert result.exit_code == 0, result.output
    assert 0 < len(gpd.read_parquet(path)) < len(ids)


def test_create_splits_sub_km_size(tmp_path: Path) -> None:
    path = _chips(tmp_path / "chips.parquet", ["ftw-33UXP000000", "ftw-33UXP060000"])

    result = CliRunner().invoke(
        cli,
        [
            "create-splits",
            str(path),
            "--split-type",
            "block3x3",
            "--split-percents",
            "50",
            "0",
            "50",
            "--km-size",
            "0.5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert set(gpd.read_parquet(path)["split"]) == {"train", "test"}


def test_km_size_in_metres_rejected(tmp_path: Path) -> None:
    path = _chips(tmp_path / "chips.parquet", ["ftw-33UXP0000"])

    result = CliRunner().invoke(
        cli, ["create-subset", str(path), "--percent", "50", "--km-size", "500"]
    )

    assert result.exit_code != 0
    assert "kilometres" in result.output
