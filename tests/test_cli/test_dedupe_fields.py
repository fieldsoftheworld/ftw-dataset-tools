"""Tests for the dedupe-fields command and the commands that require unique field ids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import pytest
import yaml
from click.testing import CliRunner
from shapely.geometry import box

from ftw_dataset_tools.api import field_ids
from ftw_dataset_tools.cli import cli

if TYPE_CHECKING:
    from pathlib import Path

LEFT = box(10.0, 50.0, 10.001, 50.001)
FAR = box(10.01, 50.0, 10.011, 50.001)


def _fields(path: Path, ids: list, crops: list[str], shapes: list) -> Path:
    gdf = gpd.GeoDataFrame({"id": ids, "crop:name": crops}, geometry=shapes, crs="EPSG:4326")
    gdf.to_parquet(path)
    return path


@pytest.fixture
def duplicated(tmp_path: Path) -> Path:
    """Two declarations of one parcel and two different fields sharing an id."""
    return _fields(
        tmp_path / "dup.parquet",
        [1, 1, 2, 2],
        ["A", "B", "x", "y"],
        [LEFT, LEFT, LEFT, FAR],
    )


class TestDedupeFieldsCommand:
    def test_writes_unique_ids(self, duplicated: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        result = CliRunner().invoke(cli, ["dedupe-fields", str(duplicated), str(out)])
        assert result.exit_code == 0, result.output
        assert "1 same_parcel (-1 rows), 1 renumbered (-0 rows); 4 -> 3 fields" in result.output
        assert "Added column(s): source_id, other_crop_names" in result.output
        assert not field_ids.find_duplicate_ids(out)
        written = gpd.read_parquet(out)
        assert written["crop:name"].tolist()[0] == "A"
        assert sorted(written["id"]) == ["1", "2_1", "2_2"]

    def test_class_filter_picks_the_declaration(self, duplicated: Path, tmp_path: Path) -> None:
        filter_file = tmp_path / "filter.yaml"
        filter_file.write_text(
            yaml.safe_dump({"column": "crop:name", "include": ["B", "x", "y"], "exclude": ["A"]})
        )
        out = tmp_path / "out.parquet"
        result = CliRunner().invoke(
            cli,
            ["dedupe-fields", str(duplicated), str(out), "--class-filter", str(filter_file)],
        )
        assert result.exit_code == 0, result.output
        assert gpd.read_parquet(out)["crop:name"].tolist()[0] == "B"

    def test_a_bad_class_filter_is_an_error(self, duplicated: Path, tmp_path: Path) -> None:
        filter_file = tmp_path / "filter.yaml"
        filter_file.write_text("column: crop:name\n")
        result = CliRunner().invoke(
            cli,
            [
                "dedupe-fields",
                str(duplicated),
                str(tmp_path / "out.parquet"),
                "--class-filter",
                str(filter_file),
            ],
        )
        assert result.exit_code == 1
        assert "Error:" in result.output

    @pytest.mark.parametrize("spelling", ["same", "relative", "symlink"])
    def test_the_output_must_not_be_the_input(
        self, duplicated: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str
    ) -> None:
        before = duplicated.read_bytes()
        if spelling == "relative":
            monkeypatch.chdir(tmp_path)
            output = "./sub/../dup.parquet"
            (tmp_path / "sub").mkdir()
        elif spelling == "symlink":
            output = str(tmp_path / "link.parquet")
            (tmp_path / "link.parquet").symlink_to(duplicated)
        else:
            output = str(duplicated)
        result = CliRunner().invoke(cli, ["dedupe-fields", str(duplicated), output])
        assert result.exit_code == 2, result.output
        assert "must not be FIELDS_FILE" in result.output
        assert duplicated.read_bytes() == before

    def test_a_missing_id_column_is_an_error(self, duplicated: Path, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            cli,
            ["dedupe-fields", str(duplicated), str(tmp_path / "o.parquet"), "--id-col", "fid"],
        )
        assert result.exit_code == 1
        assert "has no 'fid' column" in result.output


class TestCommandsRequireUniqueIds:
    """Each standalone command refuses duplicate ids before doing any work."""

    @pytest.mark.parametrize(
        "args",
        [
            ["create-chips", "{fields}"],
            ["create-boundaries", "{fields}", "-o", "{tmp}"],
            ["create-boundaries", "{dir}", "-o", "{tmp}"],
            [
                "create-masks",
                "{fields}",
                "{fields}",
                "{fields}",
                "--field-dataset",
                "t",
                "--year",
                "2024",
                "-o",
                "{tmp}",
            ],
            ["add-field-embeddings", "{fields}", "{tmp}/e.parquet", "--year", "2024"],
        ],
        ids=["create-chips", "create-boundaries", "create-boundaries-dir", "create-masks", "emb"],
    )
    def test_duplicate_ids_point_to_dedupe_fields(
        self, duplicated: Path, tmp_path: Path, args: list[str]
    ) -> None:
        argv = [
            a.format(fields=duplicated, dir=duplicated.parent, tmp=tmp_path / "out") for a in args
        ]
        result = CliRunner().invoke(cli, argv)
        assert result.exit_code == 1, result.output
        assert "dup.parquet: 2 'id' value(s) are shared by 4 rows" in result.output
        assert "ftwd dedupe-fields" in result.output
        assert not (tmp_path / "out").exists()

    def test_unique_ids_pass_the_check(self, tmp_path: Path) -> None:
        fields = _fields(tmp_path / "ok.parquet", [1, 2], ["A", "B"], [LEFT, FAR])
        result = CliRunner().invoke(
            cli, ["create-boundaries", str(fields), "-o", str(tmp_path / "out")]
        )
        assert result.exit_code == 0, result.output
