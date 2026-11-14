"""Tests for the ftwd add-field-embeddings command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
from click.testing import CliRunner
from shapely.geometry import box

from ftw_dataset_tools.api import field_embeddings
from ftw_dataset_tools.api.source import SourceFetchError
from ftw_dataset_tools.cli import cli

if TYPE_CHECKING:
    from pathlib import Path


def _fields(tmp_path: Path) -> Path:
    path = tmp_path / "fields.parquet"
    gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326").to_parquet(path)
    return path


class TestAddFieldEmbeddingsCommand:
    def test_passes_the_options_through(self, tmp_path: Path, monkeypatch) -> None:
        seen = {}

        def fake(fields, out, **kwargs):
            seen.update(fields=fields, out=out, **kwargs)
            kwargs["on_progress"](1, 1)
            return field_embeddings.FieldEmbeddingsResult(1, 1, 0, ("aef",), 2021)

        monkeypatch.setattr(field_embeddings, "add_field_embeddings", fake)
        fields = _fields(tmp_path)

        result = CliRunner().invoke(
            cli,
            [
                "add-field-embeddings",
                str(fields),
                str(tmp_path / "o.parquet"),
                "--year",
                "2021",
                "--id-col",
                "fid",
                "--workers",
                "3",
            ],
        )

        assert result.exit_code == 0, result.output
        assert (seen["year"], seen["id_col"], seen["workers"]) == (2021, "fid", 3)

    def test_errors_become_click_errors(self, tmp_path: Path, monkeypatch) -> None:
        def fail(*_args, **_kwargs):
            raise SourceFetchError("bucket unreachable")

        monkeypatch.setattr(field_embeddings, "add_field_embeddings", fail)

        result = CliRunner().invoke(
            cli,
            [
                "add-field-embeddings",
                str(_fields(tmp_path)),
                str(tmp_path / "o.parquet"),
                "--year",
                "2021",
            ],
        )

        assert result.exit_code == 1
        assert "bucket unreachable" in result.output

    def test_years_outside_both_products_are_refused(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(
            cli,
            [
                "add-field-embeddings",
                str(_fields(tmp_path)),
                str(tmp_path / "o.parquet"),
                "--year",
                "2016",
            ],
        )

        assert result.exit_code == 2
