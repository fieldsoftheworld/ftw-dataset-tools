"""Tests for the export-coco command."""

from pathlib import Path

import numpy as np
from click.testing import CliRunner

from ftw_dataset_tools.cli import cli
from tests.test_api.test_coco import write_chip


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
