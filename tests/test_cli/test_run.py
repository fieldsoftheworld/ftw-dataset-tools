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
        ctx.docs_result = None
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


class TestPrintSummaryMasksSkippedAndReused:
    def _ctx(self, tmp_path: Path, mask_result) -> Mock:
        ctx = Mock(spec=PipelineContext)
        ctx.output_dir = tmp_path
        ctx.chips_result = None
        ctx.splits_result = None
        ctx.masks_results = {"semantic_2class": mask_result}
        ctx.crop_stats_result = None
        ctx.stac_result = None
        ctx.selection_result = None
        ctx.download_result = None
        ctx.docs_result = None
        return ctx

    def test_skipped_line_printed_when_present(self, tmp_path: Path, capsys) -> None:
        from ftw_dataset_tools.api.masks import CreateMasksResult

        mask_result = CreateMasksResult(
            masks_created=[],
            masks_skipped=[("g1", "ValueError: boom")],
            field_dataset="ds",
        )
        _print_summary(self._ctx(tmp_path, mask_result))

        out = capsys.readouterr().out
        assert "Masks skipped: 1 (see log for reasons)" in out

    def test_no_skipped_line_when_none_skipped(self, tmp_path: Path, capsys) -> None:
        from ftw_dataset_tools.api.masks import CreateMasksResult

        mask_result = CreateMasksResult(masks_created=[], masks_skipped=[], field_dataset="ds")
        _print_summary(self._ctx(tmp_path, mask_result))

        assert "Masks skipped" not in capsys.readouterr().out

    def test_reused_line_printed_when_present(self, tmp_path: Path, capsys) -> None:
        from ftw_dataset_tools.api.masks import CreateMasksResult

        mask_result = CreateMasksResult(
            masks_created=[], masks_skipped=[], field_dataset="ds", masks_existing=5
        )
        _print_summary(self._ctx(tmp_path, mask_result))

        assert "Masks reused: 5" in capsys.readouterr().out

    def test_no_reused_line_when_nothing_existing(self, tmp_path: Path, capsys) -> None:
        from ftw_dataset_tools.api.masks import CreateMasksResult

        mask_result = CreateMasksResult(masks_created=[], masks_skipped=[], field_dataset="ds")
        _print_summary(self._ctx(tmp_path, mask_result))

        assert "Masks reused" not in capsys.readouterr().out

    def test_restarts_line_printed_when_present(self, tmp_path: Path, capsys) -> None:
        from ftw_dataset_tools.api.masks import CreateMasksResult

        mask_result = CreateMasksResult(
            masks_created=[], masks_skipped=[], field_dataset="ds", pool_restarts=2
        )
        _print_summary(self._ctx(tmp_path, mask_result))

        assert "Worker pool restarts: 2" in capsys.readouterr().out

    def test_no_restarts_line_when_zero(self, tmp_path: Path, capsys) -> None:
        from ftw_dataset_tools.api.masks import CreateMasksResult

        mask_result = CreateMasksResult(masks_created=[], masks_skipped=[], field_dataset="ds")
        _print_summary(self._ctx(tmp_path, mask_result))

        assert "Worker pool restarts" not in capsys.readouterr().out


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


class TestRunPmtilesTrueWithoutTippecanoe:
    def test_missing_tippecanoe_prints_clean_error(self, tmp_path: Path, monkeypatch) -> None:
        from ftw_dataset_tools.api import tiles
        from tests.test_api.test_stac import TestCollectionAssetMetadata

        TestCollectionAssetMetadata()._build_catalog(tmp_path)
        monkeypatch.setattr(tiles, "tippecanoe_available", lambda: False)

        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "fields_file": str(tmp_path / "ds_fields.parquet"),
                    "output_dir": str(tmp_path),
                    "name": "ds",
                    "year": 2024,
                    "stages": {"docs": {"pmtiles": True}},
                }
            )
        )

        result = CliRunner().invoke(run, [str(cfg), "--only", "docs"])

        assert result.exit_code != 0
        # Distinctive phrase from the RuntimeError message, not just "tippecanoe"
        # (which would also match the tmp_path directory pytest names after this test).
        assert "tippecanoe is not installed" in result.output
        assert "Traceback" not in result.output


class TestPrintSummaryDocs:
    """The docs stage gets a summary line like every other stage."""

    def _ctx(self, tmp_path: Path, docs_result, pmtiles: str | bool = "auto") -> Mock:
        from ftw_dataset_tools.api.config import DatasetConfig

        ctx = Mock(spec=PipelineContext)
        ctx.output_dir = tmp_path
        ctx.chips_result = None
        ctx.crop_stats_result = None
        ctx.splits_result = None
        ctx.masks_results = {}
        ctx.stac_result = None
        ctx.selection_result = None
        ctx.download_result = None
        ctx.docs_result = docs_result
        ctx.config = DatasetConfig.from_dict(
            {"fields_file": "f.parquet", "stages": {"docs": {"pmtiles": pmtiles}}}
        )
        return ctx

    def _result(self, tmp_path: Path, **kwargs):
        from ftw_dataset_tools.api.pipeline import DocsStageResult
        from ftw_dataset_tools.api.styles import StyleResult

        defaults = {
            "tiles": {
                "chips_tiles": tmp_path / "chips.pmtiles",
                "fields_tiles": tmp_path / "fields.pmtiles",
            },
            "styles": [
                StyleResult(f"s{i}", tmp_path / f"s{i}.json", f"S{i}", [], i == 0) for i in range(5)
            ],
            "docs": [tmp_path / "README.md", tmp_path / "AGENTS.md"],
            "tippecanoe_used": True,
        }
        defaults.update(kwargs)
        return DocsStageResult(**defaults)

    def test_full_docs_line(self, tmp_path: Path, capsys) -> None:
        _print_summary(self._ctx(tmp_path, self._result(tmp_path)))

        assert "Docs: README.md, AGENTS.md; 2 PMTiles, 5 styles" in capsys.readouterr().out

    def test_missing_tippecanoe_under_auto_is_called_out(self, tmp_path: Path, capsys) -> None:
        result = self._result(tmp_path, tiles={}, styles=[], tippecanoe_used=False)

        _print_summary(self._ctx(tmp_path, result))

        out = capsys.readouterr().out
        assert "Docs: README.md, AGENTS.md; 0 PMTiles, 0 styles (tippecanoe not found)" in out

    def test_pmtiles_disabled_is_not_reported_as_missing(self, tmp_path: Path, capsys) -> None:
        result = self._result(tmp_path, tiles={}, styles=[], tippecanoe_used=False)

        _print_summary(self._ctx(tmp_path, result, pmtiles=False))

        out = capsys.readouterr().out
        assert "Docs: README.md, AGENTS.md; 0 PMTiles, 0 styles" in out
        assert "tippecanoe not found" not in out

    def test_docs_disabled_line_lists_only_the_counts(self, tmp_path: Path, capsys) -> None:
        """readme and agents both off: no document names, and no stray separator."""
        result = self._result(tmp_path, docs=[])

        _print_summary(self._ctx(tmp_path, result))

        out = capsys.readouterr().out
        assert "  Docs: 2 PMTiles, 5 styles" in out
        assert "Docs: ; " not in out

    def test_no_docs_line_when_the_stage_did_not_run(self, tmp_path: Path, capsys) -> None:
        _print_summary(self._ctx(tmp_path, None))

        assert "Docs:" not in capsys.readouterr().out


class TestPrintSummaryImagery:
    """The imagery stages report their failures, not just their successes."""

    def _ctx(self, tmp_path: Path, selection=None, download=None) -> Mock:
        ctx = Mock(spec=PipelineContext)
        ctx.output_dir = tmp_path
        ctx.chips_result = None
        ctx.crop_stats_result = None
        ctx.splits_result = None
        ctx.masks_results = {}
        ctx.stac_result = None
        ctx.selection_result = selection
        ctx.download_result = download
        ctx.docs_result = None
        return ctx

    def _selection(self, **kwargs):
        from ftw_dataset_tools.api.imagery.selection_workflow import SelectionWorkflowResult

        return SelectionWorkflowResult(**kwargs)

    def _download(self, **kwargs):
        from ftw_dataset_tools.api.imagery.download_workflow import DownloadWorkflowResult

        return DownloadWorkflowResult(**kwargs)

    def test_counts_every_outcome(self, tmp_path: Path, capsys) -> None:
        result = self._selection(successful=770, skipped=4, failed=1)

        _print_summary(self._ctx(tmp_path, selection=result))

        assert "Imagery selection: 770 ok, 4 skipped, 1 failed" in capsys.readouterr().out

    def test_names_the_first_error(self, tmp_path: Path, capsys) -> None:
        result = self._selection(
            failed=775,
            failed_details=[
                {"chip": "ftw-31UFR9620_2025", "error": "does not resolve to a STAC object"},
                {"chip": "other", "error": "something else"},
            ],
        )

        _print_summary(self._ctx(tmp_path, selection=result))

        out = capsys.readouterr().out
        assert "Imagery selection: 0 ok, 0 skipped, 775 failed" in out
        assert "(first error: does not resolve to a STAC object)" in out
        assert "something else" not in out

    def test_download_gets_the_same_line(self, tmp_path: Path, capsys) -> None:
        result = self._download(
            successful=2, failed=1, failed_details=[{"item": "chip_planting_s2", "error": "404"}]
        )

        _print_summary(self._ctx(tmp_path, download=result))

        out = capsys.readouterr().out
        assert "Imagery download: 2 ok, 0 skipped, 1 failed (first error: 404)" in out

    def test_no_lines_when_the_stages_did_not_run(self, tmp_path: Path, capsys) -> None:
        _print_summary(self._ctx(tmp_path))

        assert "Imagery" not in capsys.readouterr().out
