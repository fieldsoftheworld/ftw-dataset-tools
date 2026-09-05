"""CLI tests for the ftwd run command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import yaml
from click.testing import CliRunner

from ftw_dataset_tools.api.crop_stats import CropStatsResult
from ftw_dataset_tools.api.field_stats import FieldStatsResult
from ftw_dataset_tools.api.pipeline import PipelineContext
from ftw_dataset_tools.commands.run import _print_summary, run

if TYPE_CHECKING:
    from pathlib import Path


def _write_config(path: Path, fields_file: Path, **overrides: object) -> Path:
    data: dict = {
        "fields_file": str(fields_file),
        "year": 2023,
        "stages": {"splits": {"split_type": "random-uniform"}},
    }
    data.update(overrides)  # type: ignore[arg-type]
    path.write_text(yaml.safe_dump(data))
    return path


class TestRunCommand:
    def test_help(self) -> None:
        result = CliRunner().invoke(run, ["--help"])
        assert result.exit_code == 0
        assert "config-driven" in result.output or "config file" in result.output

    def test_list_stages(self, sample_geoparquet_4326: Path, tmp_path: Path) -> None:
        config_path = _write_config(tmp_path / "c.yaml", sample_geoparquet_4326)
        result = CliRunner().invoke(run, [str(config_path), "--list-stages"])
        assert result.exit_code == 0
        assert "reproject" in result.output
        assert "download_images" in result.output

    def test_dry_run_shows_resolved_config_and_stages(
        self, sample_geoparquet_4326: Path, tmp_path: Path
    ) -> None:
        config_path = _write_config(tmp_path / "c.yaml", sample_geoparquet_4326)
        result = CliRunner().invoke(run, [str(config_path), "--dry-run"])
        assert result.exit_code == 0
        assert "Config (resolved):" in result.output
        assert "Stages that would run:" in result.output
        assert "reproject" in result.output
        # download_images disabled by default -> not in the run list
        assert "select_images" in result.output

    def test_missing_config_file(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(run, [str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0

    def test_invalid_config_reports_error(self, tmp_path: Path) -> None:
        config_path = tmp_path / "bad.yaml"
        config_path.write_text(yaml.safe_dump({"fields_file": "f.parquet", "bogus": 1}))
        result = CliRunner().invoke(run, [str(config_path), "--dry-run"])
        assert result.exit_code != 0
        assert "Unknown key" in result.output

    def test_only_conflicts_with_from(self, sample_geoparquet_4326: Path, tmp_path: Path) -> None:
        config_path = _write_config(tmp_path / "c.yaml", sample_geoparquet_4326)
        result = CliRunner().invoke(run, [str(config_path), "--only", "masks", "--from", "chips"])
        assert result.exit_code != 0
        assert "cannot be combined" in result.output

    def test_unknown_stage_reports_error(
        self, sample_geoparquet_4326: Path, tmp_path: Path
    ) -> None:
        config_path = _write_config(tmp_path / "c.yaml", sample_geoparquet_4326)
        result = CliRunner().invoke(run, [str(config_path), "--only", "bogus"])
        assert result.exit_code != 0
        assert "Unknown stage" in result.output

    def test_run_through_reproject_writes_provenance(
        self, sample_geoparquet_4326: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        config_path = _write_config(
            tmp_path / "c.yaml", sample_geoparquet_4326, output_dir=str(out), name="ds"
        )
        result = CliRunner().invoke(run, [str(config_path), "--through", "reproject"])
        assert result.exit_code == 0, result.output
        assert (out / "ds_fields.parquet").exists()
        assert (out / "ftwd-config.resolved.yaml").exists()


class TestRunDryRunRemote:
    def test_dry_run_url_config_reports_source_without_fetching(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from ftw_dataset_tools.api import pipeline
        from ftw_dataset_tools.cli import cli

        def boom(*_args, **_kwargs):
            raise AssertionError("fetch_source must not be called")

        monkeypatch.setattr(pipeline, "fetch_source", boom)
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            "fields_file: https://x/lu.parquet\n"
            f"output_dir: {tmp_path / 'out'}\n"
            "year: 2024\n"
            "stages:\n"
            "  splits:\n"
            "    split_type: random-uniform\n"
        )

        result = CliRunner().invoke(cli, ["run", str(cfg), "--dry-run"])

        assert result.exit_code == 0, result.output
        assert "https://x/lu.parquet" in result.output


class TestPrintSummaryCropComposition:
    def _base_ctx(self, tmp_path: Path) -> Mock:
        ctx = Mock(spec=PipelineContext)
        ctx.output_dir = tmp_path
        ctx.chips_result = FieldStatsResult(
            output_path=tmp_path / "chips.parquet",
            total_cells=10,
            cells_with_coverage=7,
            average_coverage=50.0,
            max_coverage=90.0,
        )
        ctx.splits_result = None
        ctx.masks_results = {}
        ctx.stac_result = None
        ctx.selection_result = None
        ctx.download_result = None
        return ctx

    def test_populated_crop_composition(self, tmp_path: Path, capsys) -> None:
        ctx = self._base_ctx(tmp_path)
        ctx.crop_stats_result = CropStatsResult(
            chips_total=10, chips_with_crops=7, distinct_codes=3, skipped=False
        )

        _print_summary(ctx)

        assert "Crop composition: 7/10 chips, 3 HCAT codes" in capsys.readouterr().out

    def test_skipped_crop_composition(self, tmp_path: Path, capsys) -> None:
        ctx = self._base_ctx(tmp_path)
        ctx.crop_stats_result = CropStatsResult(
            chips_total=10,
            chips_with_crops=0,
            distinct_codes=0,
            skipped=True,
            reason="fields carry no hcat:code column",
        )

        _print_summary(ctx)

        out = capsys.readouterr().out
        assert "Crop composition: skipped (fields carry no hcat:code column)" in out

    def test_disabled_crop_composition(self, tmp_path: Path, capsys) -> None:
        ctx = self._base_ctx(tmp_path)
        ctx.crop_stats_result = None

        _print_summary(ctx)

        assert "Crop composition: disabled" in capsys.readouterr().out

    def test_no_crop_line_when_chips_stage_did_not_run(self, tmp_path: Path, capsys) -> None:
        """Resuming past chips says nothing about a composition it never touched."""
        ctx = self._base_ctx(tmp_path)
        ctx.chips_result = None
        ctx.crop_stats_result = None

        _print_summary(ctx)

        assert "Crop composition" not in capsys.readouterr().out


class TestRunSourceFetchError:
    def test_fetch_failure_prints_clean_error(self, tmp_path: Path, monkeypatch) -> None:
        from ftw_dataset_tools.api import pipeline
        from ftw_dataset_tools.api.source import SourceFetchError

        def boom(*_args, **_kwargs):
            raise SourceFetchError("could not fetch https://x/y.parquet: HTTP Error 403")

        monkeypatch.setattr(pipeline, "fetch_source", boom)
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            "fields_file: https://x/y.parquet\n"
            f"output_dir: {tmp_path / 'out'}\n"
            "year: 2024\n"
            "stages:\n"
            "  splits:\n"
            "    split_type: random-uniform\n"
        )

        result = CliRunner().invoke(run, [str(cfg)])

        assert result.exit_code != 0
        assert "could not fetch" in result.output
        assert "Traceback" not in result.output
