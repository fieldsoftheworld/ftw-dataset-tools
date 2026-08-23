"""Tests for the create-dataset CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from click.testing import CliRunner

from ftw_dataset_tools.cli import cli
from ftw_dataset_tools.commands import create_dataset as create_dataset_module

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from ftw_dataset_tools.api.dataset import CreateDatasetResult


class TestCreateDatasetCommand:
    """Tests for create-dataset command."""

    def test_help(self) -> None:
        """Test --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-dataset", "--help"])
        assert result.exit_code == 0
        assert "Create a complete training dataset" in result.output

    def test_missing_input(self) -> None:
        """Test error for missing input argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-dataset"])
        assert result.exit_code != 0

    def test_nonexistent_file(self) -> None:
        """Test error for nonexistent input file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-dataset", "/nonexistent/fields.parquet"])
        assert result.exit_code != 0


class TestCreateDatasetMaskTypes:
    """Tests for --mask-types parsing and validation."""

    def test_decode_types_listed_in_help(self) -> None:
        """The DECODE layers are discoverable from --help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-dataset", "--help"])

        assert result.exit_code == 0
        assert "decode_boundary" in result.output
        assert "decode_distance" in result.output

    def test_decode_types_not_on_by_default(self) -> None:
        """The shown default stays the three standard mask types."""
        from ftw_dataset_tools.api.config import DEFAULT_MASK_TYPES

        assert "decode_boundary" not in DEFAULT_MASK_TYPES
        assert "decode_distance" not in DEFAULT_MASK_TYPES

    def test_invalid_mask_type_rejected(self, sample_fields_geoparquet: Path) -> None:
        """An unknown mask type fails before any work starts."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "create-dataset",
                str(sample_fields_geoparquet),
                "--split-type",
                "random-uniform",
                "--mask-types",
                "decode_bounary",  # typo
            ],
        )

        assert result.exit_code != 0
        assert "Invalid mask type" in result.output
        # The error lists the valid options, including the DECODE ones.
        assert "decode_boundary" in result.output

    def test_decode_mask_types_accepted(
        self, sample_fields_geoparquet: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Requesting only the DECODE layers passes validation and reaches the pipeline."""
        captured: dict[str, Any] = {}

        def fake_create_dataset(**kwargs: Any) -> CreateDatasetResult:
            captured.update(kwargs)
            raise SystemExit(0)

        monkeypatch.setattr(create_dataset_module.dataset, "create_dataset", fake_create_dataset)

        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "create-dataset",
                str(sample_fields_geoparquet),
                "--split-type",
                "random-uniform",
                "--mask-types",
                "decode_boundary,decode_distance",
            ],
        )

        assert captured["mask_types"] == ["decode_boundary", "decode_distance"]

    def test_mask_types_are_whitespace_tolerant(
        self, sample_fields_geoparquet: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spaces around the commas do not break validation."""
        captured: dict[str, Any] = {}

        def fake_create_dataset(**kwargs: Any) -> CreateDatasetResult:
            captured.update(kwargs)
            raise SystemExit(0)

        monkeypatch.setattr(create_dataset_module.dataset, "create_dataset", fake_create_dataset)

        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "create-dataset",
                str(sample_fields_geoparquet),
                "--split-type",
                "random-uniform",
                "--mask-types",
                "semantic_2_class, decode_distance",
            ],
        )

        assert captured["mask_types"] == ["semantic_2_class", "decode_distance"]
