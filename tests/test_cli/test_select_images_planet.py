"""select-images --source planetscope option handling and config validation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pystac
import pytest
from click.testing import CliRunner

from ftw_dataset_tools.api.config import ConfigError, DatasetConfig
from ftw_dataset_tools.api.imagery import selection_workflow
from ftw_dataset_tools.api.imagery.planet_api import PlanetUnavailableError
from ftw_dataset_tools.cli import cli

if TYPE_CHECKING:
    from pathlib import Path


def _dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "collection.json").write_text(
        json.dumps(
            {
                "type": "Collection",
                "id": "test",
                "stac_version": "1.0.0",
                "description": "t",
                "license": "proprietary",
                "extent": {
                    "spatial": {"bbox": [[0, 0, 1, 1]]},
                    "temporal": {"interval": [["2024-01-01T00:00:00Z", None]]},
                },
                "links": [],
            }
        )
    )
    chip_dir = dataset / "chips" / "33UXP" / "ftw-33UXP0101_2024"
    chip_dir.mkdir(parents=True)
    item = pystac.Item(
        "ftw-33UXP0101_2024",
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        [0.0, 0.0, 1.0, 1.0],
        datetime(2024, 1, 1, tzinfo=UTC),
        {},
    )
    item.set_self_href(str(chip_dir / f"{item.id}.json"))
    item.save_object(dest_href=str(chip_dir / f"{item.id}.json"))
    return dataset


def _run(*args: str) -> object:
    return CliRunner().invoke(cli, ["select-images", *args])


class TestOptionChecks:
    def test_planet_options_need_the_planetscope_source(self, tmp_path: Path) -> None:
        result = _run(str(_dataset(tmp_path)), "--no-planet-wait")

        assert result.exit_code == 2
        assert "only applies to the planetscope source" in result.output

    def test_planetscope_mosaics_are_rejected(self, tmp_path: Path) -> None:
        result = _run(
            str(_dataset(tmp_path)), "--source", "planetscope", "--imagery-mode", "mosaics"
        )

        assert result.exit_code == 2
        assert "scenes only for now" in result.output

    def test_search_backend_is_sentinel2_only(self, tmp_path: Path) -> None:
        result = _run(
            str(_dataset(tmp_path)), "--source", "planetscope", "--search-backend", "earth-search"
        )

        assert result.exit_code == 2
        assert "only applies to the sentinel-2 source" in result.output

    def test_clear_selections_is_sentinel2_only_for_now(self, tmp_path: Path) -> None:
        result = _run(str(_dataset(tmp_path)), "--source", "planetscope", "--clear-selections")

        assert result.exit_code == 2
        assert "Sentinel-2 selections only" in result.output


class TestPlanetRun:
    def test_options_reach_the_workflow(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict = {}

        def fake_select(*_args, **kwargs):
            seen.update(kwargs)
            return selection_workflow.SelectionWorkflowResult()

        monkeypatch.setattr(
            "ftw_dataset_tools.commands.select_images.select_imagery_for_catalog", fake_select
        )

        result = _run(
            str(_dataset(tmp_path)),
            "--source",
            "planetscope",
            "--no-planet-wait",
            "--planet-timeout",
            "5",
        )

        assert result.exit_code == 0, result.output
        assert seen["source"] == "planetscope"
        assert seen["planet_wait"] is False
        assert seen["planet_timeout_minutes"] == 5.0

    def test_missing_api_key_is_a_clean_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_key(*_args, **_kwargs):
            raise PlanetUnavailableError("Set PL_API_KEY to use the planetscope source.")

        monkeypatch.setattr(selection_workflow, "select_planet_scenes", no_key)

        result = _run(str(_dataset(tmp_path)), "--source", "planetscope", "--year", "2024")

        assert result.exit_code == 1
        assert "Set PL_API_KEY" in result.output
        assert "Traceback" not in result.output


class TestConfig:
    @staticmethod
    def _config(select: dict) -> DatasetConfig:
        return DatasetConfig.from_dict(
            {"fields_file": "f.parquet", "stages": {"select_images": select}}
        )

    def test_defaults_to_sentinel2(self) -> None:
        select = DatasetConfig.from_dict({"fields_file": "f.parquet"}).stages.select_images
        assert (select.source, select.planet_wait, select.planet_timeout_minutes) == (
            "sentinel-2",
            True,
            120.0,
        )

    def test_planetscope_is_accepted(self) -> None:
        assert self._config({"source": "planetscope"}).stages.select_images.source == "planetscope"

    @pytest.mark.parametrize(
        ("select", "message"),
        [
            ({"source": "landsat"}, "source must be one of"),
            ({"source": "planetscope", "imagery_mode": "mosaics"}, "imagery_mode 'scenes' only"),
            ({"planet_timeout_minutes": -1}, "non-negative"),
            ({"planet_wait": "yes"}, "planet_wait must be true or false"),
        ],
    )
    def test_invalid_values(self, select: dict, message: str) -> None:
        with pytest.raises(ConfigError, match=message):
            self._config(select)


class TestCreateDatasetImageSource:
    def test_planet_options_reach_the_dataset_and_selection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_fields_geoparquet: Path
    ) -> None:
        from ftw_dataset_tools.api import dataset
        from ftw_dataset_tools.commands import create_dataset as command

        seen: dict = {}

        def fake_create_dataset(**kwargs):
            seen["image_source"] = kwargs["image_source"]
            kwargs["on_imagery"](tmp_path)
            raise SystemExit(0)

        def fake_select(**kwargs):
            seen.update(kwargs)
            return selection_workflow.SelectionWorkflowResult()

        monkeypatch.setattr(dataset, "create_dataset", fake_create_dataset)
        monkeypatch.setattr(command, "select_imagery_for_catalog", fake_select)

        CliRunner().invoke(
            cli,
            [
                "create-dataset",
                str(sample_fields_geoparquet),
                "--split-type",
                "random-uniform",
                "--year",
                "2024",
                "--image-source",
                "planetscope",
                "--no-planet-wait",
            ],
        )

        assert seen["image_source"] == "planetscope"
        assert seen["source"] == "planetscope"
        assert seen["planet_wait"] is False

    def test_planetscope_mosaics_are_rejected(self, sample_fields_geoparquet: Path) -> None:
        result = CliRunner().invoke(
            cli,
            [
                "create-dataset",
                str(sample_fields_geoparquet),
                "--split-type",
                "random-uniform",
                "--image-source",
                "planetscope",
                "--imagery-mode",
                "mosaics",
            ],
        )

        assert result.exit_code != 0
        assert "scenes only for now" in result.output
