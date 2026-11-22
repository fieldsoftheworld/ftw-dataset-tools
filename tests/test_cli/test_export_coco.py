"""Tests for the export-coco command."""

from pathlib import Path

import numpy as np
from click.testing import CliRunner

from ftw_dataset_tools.cli import cli
from tests.test_api.test_coco import write_chip, write_table


def test_export_coco_splits_from_table(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "collection.json").write_text('{"id": "at_2024"}')
    write_chip(ds, "ftw-a", np.ones((2, 2), dtype=np.uint32), None)
    table = write_table(
        tmp_path / "chips.parquet", [("at_2024", "ftw-a", "val")], "split_admin2_v1"
    )

    result = CliRunner().invoke(cli, ["export-coco", str(ds), "--splits", str(table)])

    assert result.exit_code == 0, result.output
    assert "val: 1 images" in result.output


def test_export_coco_bad_split_column(tmp_path: Path) -> None:
    (tmp_path / "collection.json").write_text("{}")
    write_chip(tmp_path, "ftw-a", np.ones((2, 2), dtype=np.uint32), None)
    table = write_table(tmp_path / "chips.parquet", [("x", "ftw-a", "val")], "split_admin2_v1")

    result = CliRunner().invoke(
        cli, ["export-coco", str(tmp_path), "--splits", str(table), "--split-column", "nope"]
    )

    assert result.exit_code == 1
    assert "has no nope column" in result.output


def test_export_coco(tmp_path: Path) -> None:
    (tmp_path / "collection.json").write_text("{}")
    write_chip(tmp_path, "ftw-a", np.ones((2, 2), dtype=np.uint32), "train")
    write_chip(tmp_path, "ftw-b", None, "train")

    result = CliRunner().invoke(cli, ["export-coco", str(tmp_path), "--min-area", "1"])

    assert result.exit_code == 0, result.output
    assert "train: 1 images, 1 annotations" in result.output
    assert "Skipped 1 chips" in result.output
    assert (tmp_path / "coco" / "instances_train.json").exists()


def test_no_collection(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli, ["export-coco", str(tmp_path)])

    assert result.exit_code != 0
    assert "No collection.json" in result.output


def test_no_instance_masks(tmp_path: Path) -> None:
    (tmp_path / "collection.json").write_text("{}")

    result = CliRunner().invoke(cli, ["export-coco", str(tmp_path)])

    assert result.exit_code != 0
    assert "No chips with instance masks" in result.output
