"""Tests for --imagery-mode on the CLI, in config and in the pipeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pystac
import pytest
from click.testing import CliRunner

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

    def test_without_year_uses_the_chip_year(self, tmp_path: Path) -> None:
        with (
            patch(_YEAR_AVAILABLE, return_value=True),
            patch(f"{_WORKFLOW}.run_chip_selection", return_value=_success()) as run,
            patch(f"{_WORKFLOW}.ensure_crop_calendar_exists") as warm,
        ):
            result = CliRunner().invoke(
                cli, ["select-images", str(_catalog(tmp_path)), "--imagery-mode", "mosaics"]
            )
        assert result.exit_code == 0, result.output
        job = run.call_args.args[0]
        assert job.year == 2024  # the chip id's year, as its scenes would use; not 2025
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

    def test_scenes_join_a_legacy_mosaics_chip(self, tmp_path: Path) -> None:
        catalog = _catalog(
            tmp_path,
            {"ftw:imagery_mode": "mosaics", "ftw:requested_year": 2024},
            links=("ftw:q1", "ftw:q2", "ftw:q3", "ftw:q4"),
        )
        with (
            patch(f"{_WORKFLOW}.run_chip_selection", return_value=_success()) as run,
            patch(f"{_WORKFLOW}.ensure_crop_calendar_exists"),
            patch(f"{_WORKFLOW}._build_scene_index", return_value=None),
        ):
            result = CliRunner().invoke(cli, ["select-images", str(catalog)])
        assert result.exit_code == 0, result.output
        assert run.call_args.kwargs["imagery_mode"] == "scenes"

    def test_both_modes_run_scenes_then_mosaics(self, tmp_path: Path) -> None:
        with (
            patch(_YEAR_AVAILABLE, return_value=True),
            patch(f"{_WORKFLOW}.run_chip_selection", return_value=_success()) as run,
            patch(f"{_WORKFLOW}.ensure_crop_calendar_exists"),
            patch(f"{_WORKFLOW}._build_scene_index", return_value=None),
        ):
            result = CliRunner().invoke(
                cli,
                [
                    "select-images",
                    str(_catalog(tmp_path)),
                    "--imagery-mode",
                    "mosaics",
                    "--imagery-mode",
                    "scenes",
                    "--year",
                    "2024",
                ],
            )
        assert result.exit_code == 0, result.output
        assert [c.kwargs["imagery_mode"] for c in run.call_args_list] == ["scenes", "mosaics"]
        assert "Imagery selection (scenes): 1 ok" in result.output
        assert "Scenes summary:" in result.output
        assert "Mosaics summary:" in result.output

    def test_both_modes_skip_mosaics_for_a_year_without_them(self, tmp_path: Path) -> None:
        report = tmp_path / "report.json"  # outside chips/, so not read as a chip
        with (
            patch(_YEAR_AVAILABLE, return_value=False),
            patch(f"{_WORKFLOW}.run_chip_selection", return_value=_success()) as run,
            patch(f"{_WORKFLOW}.ensure_crop_calendar_exists"),
            patch(f"{_WORKFLOW}._build_scene_index", return_value=None),
        ):
            result = CliRunner().invoke(
                cli,
                [
                    "select-images",
                    str(_catalog(tmp_path)),
                    "--imagery-mode",
                    "scenes",
                    "--imagery-mode",
                    "mosaics",
                    "--year",
                    "2023",
                    "--output-report",
                    str(report),
                ],
            )
        assert result.exit_code == 0, result.output
        assert [c.kwargs["imagery_mode"] for c in run.call_args_list] == ["scenes"]
        assert "Imagery selection (mosaics): 0 ok, 1 skipped" in result.output
        skipped = json.loads(report.read_text())["skipped"]
        assert skipped[0]["mode"] == "mosaics"
        assert skipped[0]["reason"].startswith("Mosaics for 2023 are not available")

    def test_both_modes_allow_scene_options(self, tmp_path: Path) -> None:
        with (
            patch(_YEAR_AVAILABLE, return_value=True),
            patch(f"{_WORKFLOW}.run_chip_selection", return_value=_success()) as run,
            patch(f"{_WORKFLOW}.ensure_crop_calendar_exists"),
            patch(f"{_WORKFLOW}._build_scene_index", return_value=None),
        ):
            result = CliRunner().invoke(
                cli,
                [
                    "select-images",
                    str(_catalog(tmp_path)),
                    "--imagery-mode",
                    "scenes",
                    "--imagery-mode",
                    "mosaics",
                    "--cloud-cover-chip",
                    "5",
                ],
            )
        assert result.exit_code == 0, result.output
        assert run.call_args_list[0].kwargs["cloud_cover_chip"] == 5.0

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
        assert "The scenes and mosaics imagery STAC items of 1 chips" in result.output
        assert "STAC items deleted: 4" in result.output
        assert not list(chip_dir.glob("*_q*_s2.json"))

    def test_clear_selections_respects_an_explicit_mode(self, tmp_path: Path) -> None:
        slots = ("planting", "harvest", "q1", "q2", "q3", "q4")
        catalog = _catalog(tmp_path, links=tuple(f"ftw:{slot}" for slot in slots))
        chip_dir = catalog / "chips" / "32UNA" / "chip_001_2024"
        for slot in slots:
            (chip_dir / f"chip_001_2024_{slot}_s2.json").write_text("{}")

        result = CliRunner().invoke(
            cli,
            ["select-images", str(catalog), "--clear-selections", "--imagery-mode", "mosaics"],
            input="y\n",
        )

        assert result.exit_code == 0, result.output
        assert "The mosaics imagery STAC items of 1 chips" in result.output
        assert not list(chip_dir.glob("*_q*_s2.json"))
        assert (chip_dir / "chip_001_2024_planting_s2.json").exists()
        saved = json.loads((chip_dir / "chip_001_2024.json").read_text())
        assert {link["rel"] for link in saved["links"]} >= {"ftw:planting", "ftw:harvest"}

    def test_summary_counts_chips_with_this_mode(self, tmp_path: Path) -> None:
        # The chip holds mosaics only; the scenes summary must not count it.
        catalog = _catalog(tmp_path, links=("ftw:q1", "ftw:q2", "ftw:q3", "ftw:q4"))
        with (
            patch(f"{_WORKFLOW}.run_chip_selection", return_value=_success()),
            patch(f"{_WORKFLOW}.ensure_crop_calendar_exists"),
            patch(f"{_WORKFLOW}._build_scene_index", return_value=None),
        ):
            result = CliRunner().invoke(cli, ["select-images", str(catalog)])
        assert result.exit_code == 0, result.output
        assert "Total with scenes: 0/1" in result.output


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
        config = DatasetConfig.from_dict({"fields_file": "f"})
        assert config.stages.select_images.imagery_mode == ("scenes",)
        assert config.config_dict()["stages"]["select_images"]["imagery_mode"] == "scenes"

    def test_mosaics_accepted(self) -> None:
        select = self._config({"imagery_mode": "mosaics"}).stages.select_images
        assert select.imagery_mode == ("mosaics",)
        assert select.imagery_modes == ("mosaics",)

    def test_list_of_both_runs_scenes_first(self) -> None:
        select = self._config({"imagery_mode": ["mosaics", "scenes"]}).stages.select_images
        assert select.imagery_mode == ("scenes", "mosaics")

    def test_single_item_list_is_one_mode(self) -> None:
        config = self._config({"imagery_mode": ["mosaics"]})
        assert config.stages.select_images.imagery_mode == ("mosaics",)
        assert config.config_dict()["stages"]["select_images"]["imagery_mode"] == "mosaics"

    def test_repeats_collapse(self) -> None:
        select = self._config({"imagery_mode": ["scenes", "scenes"]}).stages.select_images
        assert select.imagery_mode == ("scenes",)

    def test_resolved_config_writes_both_as_a_list(self) -> None:
        import yaml

        config = self._config({"imagery_mode": ["scenes", "mosaics"]})
        dumped = yaml.safe_dump(config.provenance_dict()["config"])
        reloaded = yaml.safe_load(dumped)
        assert reloaded["stages"]["select_images"]["imagery_mode"] == ["scenes", "mosaics"]
        # The resolved section loads back to the same modes.
        again = self._config(reloaded["stages"]["select_images"])
        assert again.stages.select_images.imagery_mode == ("scenes", "mosaics")

    def test_unknown_mode_rejected(self) -> None:
        with pytest.raises(ConfigError, match="imagery_mode"):
            self._config({"imagery_mode": "monthly"})

    def test_unknown_mode_in_list_rejected(self) -> None:
        with pytest.raises(ConfigError, match="'monthly'"):
            self._config({"imagery_mode": ["scenes", "monthly"]})

    @pytest.mark.parametrize("value", [[], 3, {"scenes": True}])
    def test_malformed_value_rejected(self, value: object) -> None:
        with pytest.raises(ConfigError, match=r"stages\.select_images\.imagery_mode"):
            self._config({"imagery_mode": value})

    def test_scene_only_key_rejected_in_mosaic_mode(self) -> None:
        with pytest.raises(ConfigError, match="cloud_cover_chip only applies"):
            self._config({"imagery_mode": "mosaics", "cloud_cover_chip": 5.0})

    def test_scene_only_key_allowed_with_both_modes(self) -> None:
        config = self._config({"imagery_mode": ["scenes", "mosaics"], "cloud_cover_chip": 5.0})
        assert config.stages.select_images.cloud_cover_chip == 5.0

    def test_preview_download_rejected_as_expected_for_mosaics(self) -> None:
        with pytest.raises(ConfigError, match="This is expected"):
            self._config({"imagery_mode": "mosaics"}, {"mode": "preview"})

    def test_preview_download_allowed_with_both_modes(self) -> None:
        config = self._config({"imagery_mode": ["scenes", "mosaics"]}, {"mode": "preview"})
        assert config.stages.download_images.mode == "preview"

    def test_example_config_loads(self) -> None:
        from pathlib import Path

        from ftw_dataset_tools.api.config import load_config

        example = Path(__file__).parents[2] / "configs" / "examples" / "config.yaml"
        assert load_config(example).stages.select_images.imagery_mode == ("scenes",)
