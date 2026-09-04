"""Tests for the config-driven workflow schema and loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from ftw_dataset_tools.api import config as config_module
from ftw_dataset_tools.api.config import ConfigError, DatasetConfig

if TYPE_CHECKING:
    from pathlib import Path


class TestFromDict:
    """Tests for DatasetConfig.from_dict parsing and defaults."""

    def test_minimal_config_fills_defaults(self) -> None:
        config = DatasetConfig.from_dict({"fields_file": "fields.parquet"})
        assert config.fields_file == "fields.parquet"
        assert config.output_dir is None
        assert config.name is None
        assert config.skip_reproject is False
        # Stage defaults
        assert config.stages.chips.min_coverage == 0.01
        assert config.stages.splits.split_percents == (80, 10, 10)
        assert config.stages.masks.mask_types == [
            "instance",
            "semantic_2_class",
            "semantic_3_class",
        ]
        assert config.stages.select_images.enabled is True
        assert config.stages.download_images.enabled is False

    def test_full_config_overrides(self) -> None:
        config = DatasetConfig.from_dict(
            {
                "version": 1,
                "fields_file": "f.parquet",
                "output_dir": "./out",
                "name": "austria",
                "year": 2023,
                "skip_reproject": True,
                "stages": {
                    "chips": {"min_coverage": 0.5, "drop_border_chips": True},
                    "splits": {"split_type": "block3x3", "split_percents": [70, 20, 10]},
                    "masks": {"mask_types": ["semantic_2_class"], "resolution": 5.0},
                    "select_images": {"enabled": False, "buffer_days": 30},
                    "download_images": {"enabled": True, "bands": ["red", "green"]},
                },
            }
        )
        assert config.name == "austria"
        assert config.year == 2023
        assert config.skip_reproject is True
        assert config.stages.chips.min_coverage == 0.5
        assert config.stages.splits.split_type == "block3x3"
        assert config.stages.splits.split_percents == (70, 20, 10)
        assert config.stages.masks.mask_types == ["semantic_2_class"]
        assert config.stages.masks.resolution == 5.0
        assert config.stages.select_images.enabled is False
        assert config.stages.select_images.buffer_days == 30
        assert config.stages.download_images.bands == ["red", "green"]

    def test_missing_fields_file_raises(self) -> None:
        with pytest.raises(ConfigError, match="must specify 'fields_file'"):
            DatasetConfig.from_dict({"year": 2023})

    def test_unknown_top_level_key_raises(self) -> None:
        with pytest.raises(ConfigError, match="Unknown key"):
            DatasetConfig.from_dict({"fields_file": "f.parquet", "bogus": 1})

    def test_unknown_stage_key_raises(self) -> None:
        with pytest.raises(ConfigError, match=r"Unknown key.*stages"):
            DatasetConfig.from_dict({"fields_file": "f.parquet", "stages": {"bogus": {}}})

    def test_unknown_stage_option_raises(self) -> None:
        with pytest.raises(ConfigError, match=r"stages\.chips"):
            DatasetConfig.from_dict(
                {"fields_file": "f.parquet", "stages": {"chips": {"bad_option": 1}}}
            )

    def test_invalid_split_type_raises(self) -> None:
        with pytest.raises(ConfigError, match="Invalid split_type"):
            DatasetConfig.from_dict(
                {"fields_file": "f.parquet", "stages": {"splits": {"split_type": "nope"}}}
            )

    def test_invalid_split_percents_raises(self) -> None:
        with pytest.raises(ConfigError, match="split_percents"):
            DatasetConfig.from_dict(
                {"fields_file": "f.parquet", "stages": {"splits": {"split_percents": [50, 10, 10]}}}
            )

    def test_invalid_mask_type_raises(self) -> None:
        with pytest.raises(ConfigError, match="Invalid mask type"):
            DatasetConfig.from_dict(
                {"fields_file": "f.parquet", "stages": {"masks": {"mask_types": ["nope"]}}}
            )

    def test_decode_mask_types_accepted(self) -> None:
        config = DatasetConfig.from_dict(
            {
                "fields_file": "f.parquet",
                "stages": {"masks": {"mask_types": ["decode_boundary", "decode_distance"]}},
            }
        )
        assert config.stages.masks.mask_types == ["decode_boundary", "decode_distance"]

    def test_root_must_be_mapping(self) -> None:
        with pytest.raises(ConfigError, match="mapping"):
            DatasetConfig.from_dict([1, 2, 3])  # type: ignore[arg-type]


class TestFromKwargs:
    """Tests for building a config from create_dataset kwargs."""

    def test_maps_kwargs_and_disables_imagery(self) -> None:
        config = DatasetConfig.from_kwargs(
            fields_file="f.parquet",
            output_dir="./out",
            field_dataset="ds",
            split_type="random-uniform",
            split_percents=(80, 10, 10),
            min_coverage=0.02,
            resolution=20.0,
            num_workers=4,
            skip_reproject=False,
            year=2022,
            mask_types=["instance"],
            presence_only=True,
            drop_border_chips=True,
        )
        assert config.name == "ds"
        assert config.stages.chips.min_coverage == 0.02
        assert config.stages.chips.drop_border_chips is True
        assert config.stages.masks.workers == 4
        assert config.stages.masks.presence_only is True
        assert config.stages.masks.mask_types == ["instance"]
        # create_dataset never runs imagery itself
        assert config.stages.select_images.enabled is False
        assert config.stages.download_images.enabled is False

    def test_none_mask_types_defaults_to_standard_set(self) -> None:
        config = DatasetConfig.from_kwargs(
            fields_file="f.parquet",
            output_dir=None,
            field_dataset=None,
            split_type="random-uniform",
            split_percents=(80, 10, 10),
            min_coverage=0.01,
            resolution=10.0,
            num_workers=None,
            skip_reproject=False,
            year=None,
            mask_types=None,
            presence_only=False,
            drop_border_chips=False,
        )
        assert config.stages.masks.mask_types == list(config_module.DEFAULT_MASK_TYPES)
        # The DECODE layers are valid but opt-in, so they are not in the default set.
        assert "decode_boundary" not in config.stages.masks.mask_types
        assert "decode_distance" not in config.stages.masks.mask_types


class TestProvenance:
    """Tests for provenance / resolved-config output."""

    def test_provenance_dict_shape(self) -> None:
        config = DatasetConfig.from_dict({"fields_file": "f.parquet", "year": 2023})
        prov = config.provenance_dict()
        assert prov["ftwd_version"]
        assert prov["config_schema_version"] == config_module.CONFIG_SCHEMA_VERSION
        assert "generated_at" in prov
        assert prov["config"]["fields_file"] == "f.parquet"
        # split_percents serialized as a list (YAML-friendly), not a tuple.
        assert prov["config"]["stages"]["splits"]["split_percents"] == [80, 10, 10]

    def test_write_provenance_file_roundtrips(self, tmp_path: Path) -> None:
        config = DatasetConfig.from_dict({"fields_file": "f.parquet", "year": 2023})
        prov = config.provenance_dict()
        out = config_module.write_provenance_file(prov, tmp_path)
        assert out.exists()
        assert out.name == "ftwd-config.resolved.yaml"
        loaded = yaml.safe_load(out.read_text())
        assert loaded["config"]["fields_file"] == "f.parquet"


class TestLoadConfig:
    """Tests for loading a config from a YAML file."""

    def test_load_valid_file(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "fields_file": "f.parquet",
                    "year": 2023,
                    "stages": {"splits": {"split_type": "random-uniform"}},
                }
            )
        )
        config = config_module.load_config(config_path)
        assert config.year == 2023
        assert config.stages.splits.split_type == "random-uniform"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            config_module.load_config(tmp_path / "nope.yaml")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        config_path = tmp_path / "empty.yaml"
        config_path.write_text("")
        with pytest.raises(ConfigError, match="empty"):
            config_module.load_config(config_path)

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        config_path = tmp_path / "bad.yaml"
        config_path.write_text("fields_file: [unclosed")
        with pytest.raises(ConfigError, match="parse"):
            config_module.load_config(config_path)

    def test_class_filter_resolved_relative_to_config(self, tmp_path: Path) -> None:
        (tmp_path / "filter.yaml").write_text(
            yaml.safe_dump({"column": "crop", "include": ["wheat"], "exclude": ["water"]})
        )
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "fields_file": "f.parquet",
                    "year": 2023,
                    "stages": {"masks": {"class_filter": "filter.yaml"}},
                }
            )
        )
        config = config_module.load_config(config_path)
        assert config.class_filter is not None
        assert config.class_filter.column == "crop"
        assert config.class_filter.include == ["wheat"]

    def test_no_class_filter_leaves_none(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({"fields_file": "f.parquet", "year": 2023}))
        assert config_module.load_config(config_path).class_filter is None

    def test_top_level_class_filter_key_rejected(self) -> None:
        # class_filter is resolved from stages.masks, not a valid top-level key.
        with pytest.raises(ConfigError, match="Unknown key"):
            DatasetConfig.from_dict({"fields_file": "f.parquet", "class_filter": "x.yaml"})

    def test_grid_file_resolved_relative_to_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "fields_file": "f.parquet",
                    "year": 2023,
                    "stages": {"chips": {"grid_file": "grid.parquet"}},
                }
            )
        )
        config = config_module.load_config(config_path)
        assert config.stages.chips.grid_file == str((tmp_path / "grid.parquet").resolve())

    def test_grid_source_passthrough(self) -> None:
        config = DatasetConfig.from_dict(
            {"fields_file": "f.parquet", "stages": {"chips": {"grid_source": "s3://x/*.parquet"}}}
        )
        assert config.stages.chips.grid_source == "s3://x/*.parquet"
        assert config.stages.chips.grid_file is None
