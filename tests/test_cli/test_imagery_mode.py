"""Tests for --imagery-mode on the CLI, in config and in the pipeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pystac
import pytest
from click.testing import CliRunner

from ftw_dataset_tools.api import pipeline
from ftw_dataset_tools.api.config import ConfigError, DatasetConfig
from ftw_dataset_tools.cli import cli

if TYPE_CHECKING:
    from pathlib import Path

_WORKFLOW = "ftw_dataset_tools.api.imagery.selection_workflow"
_YEAR_AVAILABLE = "ftw_dataset_tools.api.imagery.mosaic_selection.year_available"


def _catalog(tmp_path: Path, chip_properties: dict | None = None, links: tuple = ()) -> Path:
    (tmp_path / "collection.json").write_text(
        json.dumps(
            {
                "type": "Collection",
                "id": "test",
                "stac_version": "1.0.0",
                "description": "test",
                "license": "proprietary",
                "extent": {
                    "spatial": {"bbox": [[-180.0, -90.0, 180.0, 90.0]]},
                    "temporal": {"interval": [["2024-01-01T00:00:00Z", None]]},
                },
                "links": [],
            }
        )
    )
    chip_dir = tmp_path / "chips" / "32UNA" / "chip_001_2024"
    chip_dir.mkdir(parents=True)
    item = pystac.Item(
        id="chip_001_2024",
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        bbox=(0.0, 0.0, 1.0, 1.0),
        datetime=datetime(2024, 1, 1, tzinfo=UTC),
        properties=chip_properties or {},
    )
    for rel in links:
        item.add_link(pystac.Link(rel=rel, target=f"./chip_001_2024_{rel[4:]}_s2.json"))
    path = chip_dir / "chip_001_2024.json"
    item.set_self_href(str(path))
    item.save_object(dest_href=str(path))
    return tmp_path


def _success() -> MagicMock:
    return MagicMock(success=True, skipped_reason=None, candidates_checked=4)


class TestSelectImagesMosaics:
    def test_scene_only_option_is_rejected(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            cli,
            [
                "select-images",
                str(_catalog(tmp_path)),
                "--imagery-mode",
                "mosaics",
                "--cloud-cover-chip",
                "5",
            ],
        )
        assert result.exit_code == 2
        assert "--cloud-cover-chip" in result.output
        assert "only applies to --imagery-mode scenes" in result.output

    def test_without_year_warns_and_uses_2025(self, tmp_path: Path) -> None:
        with (
            patch(_YEAR_AVAILABLE, return_value=True),
            patch(f"{_WORKFLOW}.run_chip_selection", return_value=_success()) as run,
            patch(f"{_WORKFLOW}.ensure_crop_calendar_exists") as warm,
        ):
            result = CliRunner().invoke(
                cli, ["select-images", str(_catalog(tmp_path)), "--imagery-mode", "mosaics"]
            )
        assert result.exit_code == 0, result.output
        assert "no --year given; using 2025" in result.output
        job = run.call_args.args[0]
        assert job.year == 2025  # not the chip id's 2024
        assert run.call_args.kwargs["imagery_mode"] == "mosaics"
        warm.assert_not_called()

    def test_unavailable_year_is_rejected(self, tmp_path: Path) -> None:
        with patch(_YEAR_AVAILABLE, return_value=False):
            result = CliRunner().invoke(
                cli,
                [
                    "select-images",
                    str(_catalog(tmp_path)),
                    "--imagery-mode",
                    "mosaics",
                    "--year",
                    "2023",
                ],
            )
        assert result.exit_code == 2
        assert "Mosaics for 2023 are not available" in result.output

    def test_other_year_in_workspace_errors_and_points_to_force(self, tmp_path: Path) -> None:
        catalog = _catalog(
            tmp_path,
            {"ftw:imagery_mode": "mosaics", "ftw:requested_year": 2024},
            links=("ftw:q1", "ftw:q2", "ftw:q3", "ftw:q4"),
        )
        with (
            patch(_YEAR_AVAILABLE, return_value=True),
            patch(f"{_WORKFLOW}.run_chip_selection", return_value=_success()) as run,
        ):
            result = CliRunner().invoke(
                cli,
                ["select-images", str(catalog), "--imagery-mode", "mosaics", "--year", "2025"],
            )
        assert result.exit_code == 1
        assert "has mosaics for 2024" in result.output
        assert "--force" in result.output
        run.assert_not_called()

    def test_force_replaces_the_other_year(self, tmp_path: Path) -> None:
        catalog = _catalog(
            tmp_path,
            {"ftw:imagery_mode": "mosaics", "ftw:requested_year": 2024},
            links=("ftw:q1", "ftw:q2", "ftw:q3", "ftw:q4"),
        )
        with (
            patch(_YEAR_AVAILABLE, return_value=True),
            patch(f"{_WORKFLOW}.run_chip_selection", return_value=_success()) as run,
        ):
            result = CliRunner().invoke(
                cli,
                [
                    "select-images",
                    str(catalog),
                    "--imagery-mode",
                    "mosaics",
                    "--year",
                    "2025",
                    "--force",
                ],
            )
        assert result.exit_code == 0, result.output
        assert "Cleared imagery from a different run: 1" in result.output
        assert run.call_args.args[0].year == 2025

    def test_scenes_over_mosaics_errors(self, tmp_path: Path) -> None:
        catalog = _catalog(
            tmp_path,
            {"ftw:imagery_mode": "mosaics", "ftw:requested_year": 2024},
            links=("ftw:q1", "ftw:q2", "ftw:q3", "ftw:q4"),
        )
        result = CliRunner().invoke(cli, ["select-images", str(catalog)])
        assert result.exit_code == 1
        assert "has mosaics imagery" in result.output

    def test_on_missing_fail_names_the_chip(self, tmp_path: Path) -> None:
        skipped = MagicMock(success=False, skipped_reason="No complete mosaic year (tried 2024)")
        with (
            patch(_YEAR_AVAILABLE, return_value=True),
            patch(f"{_WORKFLOW}.run_chip_selection", return_value=skipped),
        ):
            result = CliRunner().invoke(
                cli,
                [
                    "select-images",
                    str(_catalog(tmp_path)),
                    "--imagery-mode",
                    "mosaics",
                    "--year",
                    "2024",
                    "--on-missing",
                    "fail",
                ],
            )
        assert result.exit_code == 1
        assert "No imagery for chip_001_2024: No complete mosaic year" in result.output

    def test_clear_selections_removes_quarter_items(self, tmp_path: Path) -> None:
        catalog = _catalog(
            tmp_path,
            {"ftw:imagery_mode": "mosaics", "ftw:requested_year": 2024},
            links=("ftw:q1", "ftw:q2", "ftw:q3", "ftw:q4"),
        )
        chip_dir = catalog / "chips" / "32UNA" / "chip_001_2024"
        for quarter in ("q1", "q2", "q3", "q4"):
            (chip_dir / f"chip_001_2024_{quarter}_s2.json").write_text("{}")

        result = CliRunner().invoke(
            cli, ["select-images", str(catalog), "--clear-selections"], input="y\n"
        )

        assert result.exit_code == 0, result.output
        assert "The imagery STAC items of 1 chips" in result.output
        assert "STAC items deleted: 4" in result.output
        assert not list(chip_dir.glob("*_q*_s2.json"))


class TestCreateDatasetMosaics:
    def test_scene_only_option_fails_before_any_work(self, tmp_path: Path) -> None:
        fields = tmp_path / "fields.parquet"
        fields.write_bytes(b"")
        with patch("ftw_dataset_tools.commands.create_dataset.dataset.create_dataset") as create:
            result = CliRunner().invoke(
                cli,
                [
                    "create-dataset",
                    str(fields),
                    "--split-type",
                    "random-uniform",
                    "--imagery-mode",
                    "mosaics",
                    "--buffer-days",
                    "3",
                ],
            )
        assert result.exit_code == 2
        assert "--buffer-days" in result.output
        create.assert_not_called()

    def test_skip_images_needs_no_mosaic_checks(self, tmp_path: Path) -> None:
        fields = tmp_path / "fields.parquet"
        fields.write_bytes(b"")
        with (
            patch("ftw_dataset_tools.commands.create_dataset.dataset.create_dataset") as create,
            patch(_YEAR_AVAILABLE) as available,
        ):
            create.side_effect = RuntimeError("stop")
            CliRunner().invoke(
                cli,
                [
                    "create-dataset",
                    str(fields),
                    "--split-type",
                    "random-uniform",
                    "--imagery-mode",
                    "mosaics",
                    "--skip-images",
                ],
            )
        available.assert_not_called()
        assert create.call_args.kwargs["on_imagery"] is None


class TestImageryModeConfig:
    def _config(self, select: dict, download: dict | None = None) -> DatasetConfig:
        stages: dict = {"select_images": select}
        if download is not None:
            stages["download_images"] = download
        return DatasetConfig.from_dict({"fields_file": "f.parquet", "stages": stages})

    def test_defaults_to_scenes(self) -> None:
        assert DatasetConfig.from_dict({"fields_file": "f"}).stages.select_images.imagery_mode == (
            "scenes"
        )

    def test_mosaics_accepted(self) -> None:
        assert self._config({"imagery_mode": "mosaics"}).stages.select_images.imagery_mode == (
            "mosaics"
        )

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ConfigError, match="imagery_mode"):
            self._config({"imagery_mode": "monthly"})

    def test_scene_only_key_rejected_in_mosaic_mode(self) -> None:
        with pytest.raises(ConfigError, match="cloud_cover_chip only applies"):
            self._config({"imagery_mode": "mosaics", "cloud_cover_chip": 5.0})

    def test_preview_download_rejected_as_expected_for_mosaics(self) -> None:
        with pytest.raises(ConfigError, match="This is expected"):
            self._config({"imagery_mode": "mosaics"}, {"mode": "preview"})


class TestPipelineMosaicYear:
    def test_configured_year_is_used(self, caplog: pytest.LogCaptureFixture) -> None:
        assert pipeline.mosaic_year(2022) == 2022
        assert not caplog.records

    def test_missing_year_warns_and_falls_back(self, caplog: pytest.LogCaptureFixture) -> None:
        assert pipeline.mosaic_year(None) == 2025
        assert any(
            r.levelname == "WARNING" and "no year given; using 2025" in r.message
            for r in caplog.records
        )
