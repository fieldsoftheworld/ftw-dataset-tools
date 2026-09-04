"""CLI tests for the ftwd run command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml
from click.testing import CliRunner

from ftw_dataset_tools.commands.run import run

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
