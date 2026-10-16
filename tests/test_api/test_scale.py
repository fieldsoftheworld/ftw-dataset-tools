"""Tests for hash-based nested chip scales."""

import shutil
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from ftw_dataset_tools.api.blocks import chip_block_ids
from ftw_dataset_tools.api.scale import SCORE_COLUMN, apply_scale, validate_scale

SQUARES = ("36NXF", "36NXG")


def _write_chips(path: Path, ids: list[str]) -> Path:
    gdf = gpd.GeoDataFrame(
        {
            "id": ids,
            "field_coverage_pct": [50.0] * len(ids),
            "geometry": [box(i, 0, i + 1, 1) for i in range(len(ids))],
        },
        crs="EPSG:4326",
    )
    gdf.to_parquet(path)
    return path


@pytest.fixture
def chips_file(tmp_path: Path) -> Path:
    # Two squares, each a 9x9 grid of 2 km cells: 9 blocks per square.
    ids = [
        f"ftw-{sq}{e:02d}{n:02d}"
        for sq in SQUARES
        for e in range(0, 18, 2)
        for n in range(0, 18, 2)
    ]
    return _write_chips(tmp_path / "chips.parquet", ids)


def _ids(path: Path) -> set[str]:
    return set(gpd.read_parquet(path)["id"])


class TestApplyScale:
    def test_smaller_scales_are_subsets(self, chips_file: Path, tmp_path: Path) -> None:
        kept = []
        for percent in (20, 50, 80):
            path = shutil.copy(chips_file, tmp_path / f"chips_{percent}.parquet")
            apply_scale(path, percent=percent)
            kept.append(_ids(Path(path)))
        assert kept[0] <= kept[1] <= kept[2]
        assert len(kept[0]) < len(kept[2])

    def test_keeps_whole_blocks(self, chips_file: Path) -> None:
        result = apply_scale(chips_file, percent=50, min_blocks_per_square=0)
        gdf = gpd.read_parquet(chips_file)
        assert result.kept_chips == len(gdf) and result.total_chips == 162
        assert len(gdf) % 9 == 0
        assert (gdf[SCORE_COLUMN] < 0.5).all()
        assert (chip_block_ids(gdf["id"], 2).value_counts() == 9).all()

    def test_floor_keeps_blocks_in_every_square(self, chips_file: Path) -> None:
        apply_scale(chips_file, percent=0, min_blocks_per_square=1)
        gdf = gpd.read_parquet(chips_file)
        assert len(gdf) == 18
        assert set(gdf["id"].str[4:9]) == set(SQUARES)

    def test_full_scale_keeps_every_chip_and_columns(self, chips_file: Path) -> None:
        result = apply_scale(chips_file)
        gdf = gpd.read_parquet(chips_file)
        assert result.kept_chips == result.total_chips == len(gdf) == 162
        assert {"field_coverage_pct", "geometry", SCORE_COLUMN} <= set(gdf.columns)
        assert gdf[SCORE_COLUMN].between(0, 1, inclusive="left").all()

    def test_rerun_replaces_score_column(self, chips_file: Path) -> None:
        apply_scale(chips_file, percent=60)
        first = gpd.read_parquet(chips_file)
        apply_scale(chips_file, percent=30)
        second = gpd.read_parquet(chips_file)
        assert list(second.columns).count(SCORE_COLUMN) == 1
        assert set(second["id"]) <= set(first["id"])

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            apply_scale(tmp_path / "missing.parquet", percent=10)

    def test_missing_id_column(self, tmp_path: Path) -> None:
        path = tmp_path / "chips.parquet"
        gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326").to_parquet(path)
        with pytest.raises(ValueError, match="'id' column"):
            apply_scale(path, percent=10)

    def test_empty_file(self, tmp_path: Path) -> None:
        path = _write_chips(tmp_path / "chips.parquet", [])
        with pytest.raises(ValueError, match="empty"):
            apply_scale(path, percent=10)


class TestValidateScale:
    @pytest.mark.parametrize("percent", [-1, 100.5, True, "10"])
    def test_rejects_bad_percent(self, percent: object) -> None:
        with pytest.raises(ValueError, match="between 0 and 100"):
            validate_scale(percent, 1)  # type: ignore[arg-type]

    @pytest.mark.parametrize("min_blocks", [-1, 1.5, True])
    def test_rejects_bad_floor(self, min_blocks: object) -> None:
        with pytest.raises(ValueError, match="non-negative integer"):
            validate_scale(10, min_blocks)  # type: ignore[arg-type]

    def test_accepts_bounds(self) -> None:
        validate_scale(0, 0)
        validate_scale(100.0, 3)
