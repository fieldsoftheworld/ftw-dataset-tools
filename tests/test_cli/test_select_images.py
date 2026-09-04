"""Tests for the select-images CLI command."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pystac
from click.testing import CliRunner

from ftw_dataset_tools.cli import cli

if TYPE_CHECKING:
    from pathlib import Path


def _write_minimal_collection(path: Path) -> None:
    """Write a minimal valid STAC Collection JSON to `path`."""
    collection = {
        "type": "Collection",
        "id": "test-dataset",
        "stac_version": "1.0.0",
        "description": "Test dataset",
        "license": "proprietary",
        "extent": {
            "spatial": {"bbox": [[-180.0, -90.0, 180.0, 90.0]]},
            "temporal": {"interval": [["2024-01-01T00:00:00Z", "2024-12-31T00:00:00Z"]]},
        },
        "links": [],
    }
    path.write_text(json.dumps(collection))


def _write_chip_item(item_dir: Path, item_id: str) -> Path:
    """Save a minimal parent chip STAC item under `item_dir`, return its path."""
    item_dir.mkdir(parents=True)
    item = pystac.Item(
        id=item_id,
        geometry={
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        },
        bbox=(0.0, 0.0, 1.0, 1.0),
        datetime=datetime(2024, 1, 1, tzinfo=UTC),
        properties={},
    )
    item_path = item_dir / f"{item_id}.json"
    item.set_self_href(str(item_path))
    item.save_object(dest_href=str(item_path))
    return item_path


class TestSelectImagesSingleChipMode:
    """Tests for single-chip-mode collection directory discovery.

    The item now lives at <dataset>/chips/<square>/<item_id>/<item_id>.json
    (two levels deeper than the old flat layout), so the command has to walk
    up to the dataset root to find collection.json. `select_scenes_for_chip`
    (the function the command calls to actually select imagery) does not
    receive `catalog_dir` at all -- it only takes chip_id/bbox/year/etc -- so
    monkeypatching it can't observe what `catalog_dir` the command resolved.
    `--show-stats` is the command's own built-in mechanism for surfacing the
    resolved `catalog_dir` in observable output, and returns before any
    selection/network work, so it's used here instead.
    """

    def test_resolves_catalog_dir_to_dataset_root(self, tmp_path: Path) -> None:
        """Single-chip mode locates collection.json at the dataset root."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        _write_minimal_collection(dataset_dir / "collection.json")

        item_dir = dataset_dir / "chips" / "33UXP" / "ftw-item1"
        item_path = _write_chip_item(item_dir, "ftw-item1")

        result = CliRunner().invoke(
            cli,
            ["select-images", str(item_path), "--year", "2024", "--show-stats"],
        )

        assert result.exit_code == 0, result.output
        assert f"Imagery Selection Statistics for {dataset_dir.resolve()}" in result.output

    def test_directory_without_collection_json_fails_clearly(self, tmp_path: Path) -> None:
        """Pointing at a directory with no collection.json is a clear error."""
        empty_dir = tmp_path / "not-a-dataset"
        empty_dir.mkdir()

        result = CliRunner().invoke(cli, ["select-images", str(empty_dir)])

        assert result.exit_code != 0
        assert "collection.json" in result.output
