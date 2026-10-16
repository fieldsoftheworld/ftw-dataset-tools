"""Tests for the convert-previews CLI command."""

from __future__ import annotations

import datetime
import json
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pystac
import pytest
from click.testing import CliRunner

from ftw_dataset_tools.api.imagery import preview_conversion
from ftw_dataset_tools.api.imagery.parallel import MAX_WORKERS
from ftw_dataset_tools.api.imagery.preview_conversion import ConversionResult
from ftw_dataset_tools.cli import cli
from tests.test_api.test_imagery.test_preview_conversion import _clipped_chip

if TYPE_CHECKING:
    from pathlib import Path

LEGACY_PREVIEWS = ("_overlay.jpg", "_planting_image_s2.jpg", "_harvest_image_s2.jpg")


def _collection(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "collection.json").write_text(
        json.dumps(
            {
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
        )
    )
    return dataset_dir


def _chip(dataset_dir: Path, chip_id: str, *, sources: bool) -> Path:
    """A chip with legacy .jpg previews, optionally with imagery to re-render them from.

    Only the dry run is exercised here, and it decides from what is on disk rather
    than from pixels, so the mask and imagery are placeholders.
    """
    chip_dir = dataset_dir / "chips" / "33TXM" / chip_id
    chip_dir.mkdir(parents=True)

    for suffix in LEGACY_PREVIEWS:
        (chip_dir / f"{chip_id}{suffix}").write_bytes(b"jpeg")
    if sources:
        (chip_dir / f"{chip_id}_semantic_3_class.tif").write_bytes(b"tif")
        for season in ("planting", "harvest"):
            (chip_dir / f"{chip_id}_{season}_image_s2.tif").write_bytes(b"tif")

    item = pystac.Item(
        id=chip_id,
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        bbox=[0.0, 0.0, 1.0, 1.0],
        datetime=datetime.datetime(2024, 6, 1, tzinfo=datetime.UTC),
        properties={},
    )
    item.add_asset(
        "thumbnail",
        pystac.Asset(href=f"./{chip_id}_overlay.jpg", media_type="image/jpeg", roles=["thumbnail"]),
    )
    item_path = chip_dir / f"{chip_id}.json"
    item.set_self_href(str(item_path))
    item.save_object(include_self_link=False, dest_href=str(item_path))
    return chip_dir


class TestConvertPreviewsDryRun:
    """--dry-run reports what a run would do and touches nothing."""

    def test_reports_the_counts(self, tmp_path: Path) -> None:
        dataset_dir = _collection(tmp_path)
        _chip(dataset_dir, "chip_a", sources=True)

        result = CliRunner().invoke(cli, ["convert-previews", str(dataset_dir), "--dry-run"])

        assert result.exit_code == 0, result.output
        assert (
            "Would convert 1 chips (3 previews, 3 JPEGs removed); 0 skipped, 0 failed."
            in result.output
        )

    def test_writes_nothing(self, tmp_path: Path) -> None:
        dataset_dir = _collection(tmp_path)
        chip_dir = _chip(dataset_dir, "chip_a", sources=True)
        before = {path.name: path.read_bytes() for path in sorted(chip_dir.iterdir())}

        result = CliRunner().invoke(cli, ["convert-previews", str(dataset_dir), "--dry-run"])

        assert result.exit_code == 0, result.output
        after = {path.name: path.read_bytes() for path in sorted(chip_dir.iterdir())}
        assert after == before

    def test_reports_a_chip_with_no_source_as_skipped(self, tmp_path: Path) -> None:
        """A chip the real run cannot convert must not be counted as one that would."""
        dataset_dir = _collection(tmp_path)
        _chip(dataset_dir, "chip_c", sources=False)

        result = CliRunner().invoke(
            cli, ["convert-previews", str(dataset_dir), "--dry-run", "--verbose"]
        )

        assert result.exit_code == 0, result.output
        assert (
            "Would convert 0 chips (0 previews, 0 JPEGs removed); 1 skipped, 0 failed."
            in result.output
        )
        assert "skipped chip_c: No imagery to re-render the preview from" in result.output

    def test_an_empty_catalog_converts_nothing(self, tmp_path: Path) -> None:
        dataset_dir = _collection(tmp_path)

        result = CliRunner().invoke(cli, ["convert-previews", str(dataset_dir), "--dry-run"])

        assert result.exit_code == 0, result.output
        assert "Would convert 0 chips" in result.output


class TestConvertPreviewsErrors:
    """Bad invocations are rejected, and a chip that cannot be read fails the run."""

    def test_missing_catalog_dir_is_rejected(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(cli, ["convert-previews", str(tmp_path / "nope")])

        assert result.exit_code == 2
        assert "CATALOG_DIR" in result.output

    def test_a_file_in_place_of_the_catalog_dir_is_rejected(self, tmp_path: Path) -> None:
        not_a_dir = tmp_path / "collection.json"
        not_a_dir.write_text("{}")

        result = CliRunner().invoke(cli, ["convert-previews", str(not_a_dir)])

        assert result.exit_code == 2

    def test_workers_outside_the_allowed_range_are_rejected(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.imagery.parallel import MAX_WORKERS

        dataset_dir = _collection(tmp_path)

        for value in ("0", "-1", str(MAX_WORKERS + 1)):
            result = CliRunner().invoke(
                cli, ["convert-previews", str(dataset_dir), "--workers", value]
            )
            assert result.exit_code == 2, value
            assert "--workers" in result.output, value

    def test_an_unreadable_chip_fails_the_run(self, tmp_path: Path) -> None:
        dataset_dir = _collection(tmp_path)
        chip_dir = dataset_dir / "chips" / "33TXM" / "chip_broken"
        chip_dir.mkdir(parents=True)
        (chip_dir / "chip_broken.json").write_text("{ not json")

        result = CliRunner().invoke(cli, ["convert-previews", str(dataset_dir), "--verbose"])

        assert result.exit_code == 1
        assert "failed chip_broken" in result.output
        assert "1 chips failed to convert" in result.output


def test_help():
    result = CliRunner().invoke(cli, ["convert-previews", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.output
    assert "--workers" in result.output


@pytest.mark.parametrize("workers", ["0", str(MAX_WORKERS + 1)])
def test_rejects_invalid_workers(tmp_path, workers):
    result = CliRunner().invoke(cli, ["convert-previews", str(tmp_path), "--workers", workers])
    assert result.exit_code == 2
    assert "--workers" in result.output


def test_forwards_options_and_reports_dry_run_skips(tmp_path, monkeypatch):
    convert = Mock(
        return_value=ConversionResult(
            skipped=1, skipped_details=[{"chip": "chip_a", "reason": "No imagery"}]
        )
    )
    monkeypatch.setattr(preview_conversion, "convert_previews_for_catalog", convert)

    result = CliRunner().invoke(
        cli, ["convert-previews", str(tmp_path), "--dry-run", "--workers", "2", "-v"]
    )

    assert result.exit_code == 0
    convert.assert_called_once_with(tmp_path, dry_run=True, workers=2)
    assert "Would convert 0 chips" in result.output
    assert "1 skipped, 0 failed" in result.output
    assert "skipped chip_a: No imagery" in result.output


@pytest.mark.parametrize("dry_run", [False, True])
def test_failure_is_a_click_error(tmp_path, monkeypatch, dry_run):
    monkeypatch.setattr(
        preview_conversion,
        "convert_previews_for_catalog",
        Mock(
            return_value=ConversionResult(
                failed=1, failed_details=[{"chip": "chip_a", "error": "Unreadable child"}]
            )
        ),
    )
    args = ["convert-previews", str(tmp_path), "-v"]
    if dry_run:
        args.append("--dry-run")

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 1
    assert "failed chip_a: Unreadable child" in result.output
    assert "Error: 1 chips failed to convert" in result.output


def test_real_conversion_and_dry_run(tmp_path):
    out = _collection(tmp_path)
    chip = _clipped_chip(out, "chip_a")
    before = {p: p.read_bytes() for p in chip.iterdir()}

    dry = CliRunner().invoke(cli, ["convert-previews", str(out), "--dry-run"])

    assert dry.exit_code == 0, dry.output
    assert "Would convert 1 chips (3 previews, 3 JPEGs removed)" in dry.output
    assert {p: p.read_bytes() for p in chip.iterdir()} == before

    actual = CliRunner().invoke(cli, ["convert-previews", str(out), "--workers", "1"])

    assert actual.exit_code == 0, actual.output
    assert "1 chips converted, 3 previews written, 3 JPEGs removed" in actual.output
    assert (chip / "chip_a_overlay.webp").exists()
    assert not (chip / "chip_a_overlay.jpg").exists()
