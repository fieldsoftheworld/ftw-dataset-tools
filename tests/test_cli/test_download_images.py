"""Tests for the download-images CLI command."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import patch

import pystac
from click.testing import CliRunner

from ftw_dataset_tools.api.imagery.image_download import DownloadResult
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


def _write_staged_child_item(dataset_dir: Path) -> Path:
    """Stage a planting child item whose `root` link points at a missing file.

    Mirrors the real staging tree, where hierarchical links already carry the
    *published* root: resolving that root raises, so anything that saves the
    item through ``Item.save_object`` fails for every scene.
    """
    item_dir = dataset_dir / "chips" / "31UFR" / "ftw-chip"
    item_dir.mkdir(parents=True)
    path = item_dir / "ftw-chip_planting_s2.json"

    item = pystac.Item(
        id="ftw-chip_planting_s2",
        geometry={
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        },
        bbox=[0.0, 0.0, 1.0, 1.0],
        datetime=datetime(2024, 3, 1, tzinfo=UTC),
        properties={"eo:cloud_cover": 1.5},
    )
    item.add_link(pystac.Link(rel="root", target="../../../does-not-exist.json"))
    for band in ("red", "green", "blue", "nir"):
        item.add_asset(
            band,
            pystac.Asset(href=f"https://example.com/scene/{band}.tif", roles=["data"]),
        )

    path.write_text(
        json.dumps(item.to_dict(include_self_link=False, transform_hrefs=False), indent=2) + "\n"
    )
    return path


def _fake_download(output_path: Path) -> DownloadResult:
    """A successful download that leaves a (tiny) file behind."""
    output_path.write_bytes(b"not-really-a-geotiff")
    return DownloadResult(
        output_path=output_path,
        scene_id="S2_FAKE",
        season="planting",
        bands=["red", "green", "blue", "nir"],
        width=2,
        height=2,
        crs="EPSG:4326",
        success=True,
    )


class TestDownloadImagesKeepRemoteRefs:
    """`--keep-remote-refs` writes the child item without resolving its root.

    The legacy branch saved with ``item.save_object(str(item_path))``. The first
    positional parameter of ``STACObject.save_object`` is ``include_self_link``
    (a bool), not ``dest_href``, so the path was swallowed as a truthy flag, the
    destination stayed unset, and pystac fell back to the self href -- resolving
    the root on the way and failing on every scene in a staged catalog.
    """

    def test_writes_clipped_asset_to_the_item_path(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        _write_minimal_collection(dataset_dir / "collection.json")
        item_path = _write_staged_child_item(dataset_dir)

        with patch(
            "ftw_dataset_tools.commands.download_images.download_and_clip_scene"
        ) as mock_download:
            mock_download.side_effect = lambda **kwargs: _fake_download(kwargs["output_path"])

            result = CliRunner().invoke(
                cli,
                ["download-images", str(dataset_dir), "--keep-remote-refs"],
            )

        assert result.exit_code == 0, result.output
        assert "Downloaded: 1" in result.output
        assert "Failed: 0" in result.output

        written = json.loads(item_path.read_text())
        assert written["assets"]["clipped"]["href"] == "./ftw-chip_planting_image_s2.tif"
        # Remote band references are kept, which is the point of the flag.
        assert written["assets"]["red"]["href"] == "https://example.com/scene/red.tif"
        # The unresolvable root link survives the write untouched.
        root = next(link for link in written["links"] if link["rel"] == "root")
        assert root["href"] == "../../../does-not-exist.json"

    def test_unreadable_item_is_reported_as_a_failure(self, tmp_path: Path) -> None:
        """A chip whose JSON cannot be parsed is counted, not silently dropped."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        _write_minimal_collection(dataset_dir / "collection.json")
        _write_staged_child_item(dataset_dir)

        broken_dir = dataset_dir / "chips" / "31UFR" / "ftw-broken"
        broken_dir.mkdir(parents=True)
        (broken_dir / "ftw-broken_planting_s2.json").write_text("{ not json")

        with patch(
            "ftw_dataset_tools.commands.download_images.download_and_clip_scene"
        ) as mock_download:
            mock_download.side_effect = lambda **kwargs: _fake_download(kwargs["output_path"])

            result = CliRunner().invoke(
                cli,
                ["download-images", str(dataset_dir), "--keep-remote-refs"],
            )

        assert result.exit_code == 0, result.output
        assert "Failed: 1" in result.output
        assert "ftw-broken_planting_s2" in result.output
