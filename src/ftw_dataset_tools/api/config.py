"""Config-driven workflow schema and loading for FTW dataset creation.

This module defines the YAML config schema for ``ftwd run`` as a set of
dataclasses, plus helpers to load/validate a config file, fill in defaults, and
produce a fully-resolved provenance record.

The config is the single source of truth for a dataset build: every setting that
the ``create-dataset`` pipeline accepts has a home here. Loading a config resolves
all defaults so the resolved form can be written alongside the output for
reproducibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ftw_dataset_tools import __version__
from ftw_dataset_tools.api import splits

# Mask types that ``create-masks`` understands. Kept here so config validation
# gives the same error the CLI does.
VALID_MASK_TYPES = ("instance", "semantic_2_class", "semantic_3_class")

# Current config schema version. Bump when the schema changes incompatibly.
CONFIG_SCHEMA_VERSION = 1

# YAML keys allowed at the top level of a config file. `class_filter` on
# DatasetConfig is resolved from stages.masks.class_filter, not set via YAML.
_ALLOWED_TOP_KEYS = ("fields_file", "output_dir", "name", "year", "skip_reproject", "stages")


class ConfigError(ValueError):
    """Raised when a config file is malformed or contains invalid values."""


class ClassFilterError(ConfigError):
    """Raised when a class filter file is malformed or inconsistent with the data."""


@dataclass
class ClassFilter:
    """A field/background class filter loaded from a self-contained YAML file.

    Attributes:
        column: Column in the fields file holding the class/crop-type value.
        include: Class values that count as field (rasterized as foreground).
        exclude: Class values that count as background.
        source: Path the filter was loaded from (for provenance).

    Class values are compared as strings, so numeric crop codes work too.
    """

    column: str
    include: list[str]
    exclude: list[str]
    source: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> ClassFilter:
        """Load and structurally validate a class filter YAML file."""
        filter_path = Path(path)
        if not filter_path.exists():
            raise ClassFilterError(f"Class filter file not found: {filter_path}")

        try:
            raw = yaml.safe_load(filter_path.read_text())
        except yaml.YAMLError as err:
            raise ClassFilterError(
                f"Could not parse class filter YAML {filter_path}: {err}"
            ) from err

        if not isinstance(raw, dict):
            raise ClassFilterError(
                f"Class filter {filter_path} must be a mapping with 'column', "
                "'include', and 'exclude'."
            )
        unknown = set(raw) - {"column", "include", "exclude"}
        if unknown:
            raise ClassFilterError(
                f"Unknown key(s) in class filter {filter_path.name}: "
                f"{', '.join(sorted(unknown))}. Allowed: column, include, exclude."
            )

        column = raw.get("column")
        if not isinstance(column, str) or not column:
            raise ClassFilterError(f"Class filter {filter_path} must specify a string 'column'.")

        include = _as_str_list(raw.get("include"), "include", filter_path)
        exclude = _as_str_list(raw.get("exclude"), "exclude", filter_path)
        if not include and not exclude:
            raise ClassFilterError(
                f"Class filter {filter_path} must list at least one class in "
                "'include' or 'exclude'."
            )

        overlap = sorted(set(include) & set(exclude))
        if overlap:
            raise ClassFilterError(
                f"Class filter {filter_path}: classes appear in both include and exclude: {overlap}"
            )

        return cls(column=column, include=include, exclude=exclude, source=str(filter_path))

    def handled(self) -> set[str]:
        """Return the set of all classes the filter accounts for."""
        return set(self.include) | set(self.exclude)

    def validate_against(self, distinct: set[str | None], on_progress: Any = None) -> None:
        """Enforce that every class in the data is handled; warn on stale entries.

        Args:
            distinct: Distinct class values found in the data (may contain None).
            on_progress: Optional callback for the listed-but-absent warning.

        Raises:
            ClassFilterError: If any value in the data is in neither include nor
                exclude (NULLs are reported as ``<null>``).
        """
        handled = self.handled()
        offenders = sorted(
            "<null>" if value is None else value
            for value in distinct
            if value is None or value not in handled
        )
        if offenders:
            raise ClassFilterError(
                f"Column '{self.column}' has values not covered by the class filter. "
                f"Add each to include or exclude: {offenders}"
            )

        present = {value for value in distinct if value is not None}
        absent = sorted(handled - present)
        if absent and on_progress is not None:
            on_progress(
                f"Warning: class filter lists {len(absent)} class(es) not present "
                f"in the data: {absent}"
            )


@dataclass
class ChipsConfig:
    """Settings for the chips stage (field coverage statistics)."""

    min_coverage: float = 0.01
    drop_border_chips: bool = False
    # Local FTW grid parquet to use instead of fetching from Source Coop. Path is
    # resolved relative to the config file. Optional.
    grid_file: str | None = None
    # Alternate remote grid source (URL / S3 glob) overriding the Source Coop
    # default. Ignored when grid_file is set. Optional.
    grid_source: str | None = None


@dataclass
class SplitsConfig:
    """Settings for the train/val/test split stage."""

    split_type: str | None = None
    split_percents: tuple[int, int, int] = (80, 10, 10)
    random_seed: int = 42


@dataclass
class MasksConfig:
    """Settings for the raster mask stage."""

    mask_types: list[str] = field(
        default_factory=lambda: ["instance", "semantic_2_class", "semantic_3_class"]
    )
    resolution: float = 10.0
    workers: int | None = None
    presence_only: bool = False
    # Path to a class filter YAML (resolved relative to the config file). Optional.
    class_filter: str | None = None


@dataclass
class SelectImagesConfig:
    """Settings for the imagery selection stage."""

    enabled: bool = True
    cloud_cover_chip: float = 2.0
    nodata_max: float = 0.0
    buffer_days: int = 14
    num_buffer_expansions: int = 3
    buffer_expansion_size: int = 14


@dataclass
class DownloadImagesConfig:
    """Settings for the imagery download stage."""

    enabled: bool = False
    bands: list[str] = field(default_factory=lambda: ["red", "green", "blue", "nir"])
    resolution: float = 10.0


@dataclass
class StagesConfig:
    """Per-stage settings. Stages with no options (boundaries, stac) still run."""

    chips: ChipsConfig = field(default_factory=ChipsConfig)
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    masks: MasksConfig = field(default_factory=MasksConfig)
    select_images: SelectImagesConfig = field(default_factory=SelectImagesConfig)
    download_images: DownloadImagesConfig = field(default_factory=DownloadImagesConfig)


@dataclass
class DatasetConfig:
    """Top-level config for a dataset build.

    Attributes:
        fields_file: Path to the input GeoParquet field boundaries.
        output_dir: Output directory. If None, derived from the input stem.
        name: Dataset name used in filenames. If None, derived from the input stem.
        year: Calendar year for temporal extent (optional if the fields file has
            a determination_datetime column).
        skip_reproject: If True, fail instead of reprojecting non-4326 input.
        stages: Per-stage settings.
        class_filter: Resolved class filter (loaded from stages.masks.class_filter).
            Not set directly from YAML; populated by load_config / create_dataset.
    """

    fields_file: str
    output_dir: str | None = None
    name: str | None = None
    year: int | None = None
    skip_reproject: bool = False
    stages: StagesConfig = field(default_factory=StagesConfig)
    class_filter: ClassFilter | None = None

    # ---- construction ---------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetConfig:
        """Build and validate a config from a plain dict (e.g. parsed YAML)."""
        if not isinstance(data, dict):
            raise ConfigError("Config root must be a mapping of keys to values.")

        # Ignore a documented schema-version key if present.
        data = {k: v for k, v in data.items() if k != "version"}

        # class_filter is resolved from stages.masks.class_filter, never set at the
        # top level via YAML, so it is excluded from the allowed top-level keys.
        _reject_unknown(data, set(_ALLOWED_TOP_KEYS), context="config")

        if "fields_file" not in data or data["fields_file"] is None:
            raise ConfigError("Config must specify 'fields_file'.")

        stages_data = data.get("stages") or {}
        if not isinstance(stages_data, dict):
            raise ConfigError("'stages' must be a mapping.")
        stages = _build_stages(stages_data)

        config = cls(
            fields_file=str(data["fields_file"]),
            output_dir=_opt_str(data.get("output_dir")),
            name=_opt_str(data.get("name")),
            year=data.get("year"),
            skip_reproject=bool(data.get("skip_reproject", False)),
            stages=stages,
        )
        config.validate()
        return config

    @classmethod
    def from_kwargs(
        cls,
        *,
        fields_file: str,
        output_dir: str | None,
        field_dataset: str | None,
        split_type: str | None,
        split_percents: tuple[int, int, int],
        min_coverage: float,
        resolution: float,
        num_workers: int | None,
        skip_reproject: bool,
        year: int | None,
        mask_types: list[str] | None,
        presence_only: bool,
        drop_border_chips: bool,
    ) -> DatasetConfig:
        """Build a config from ``create_dataset`` keyword arguments.

        Lets the flag-driven ``create-dataset`` pipeline share the config-driven
        orchestration. Imagery stages are disabled here because the
        ``create-dataset`` API does not run imagery itself.
        """
        config = cls(
            fields_file=str(fields_file),
            output_dir=str(output_dir) if output_dir is not None else None,
            name=field_dataset,
            year=year,
            skip_reproject=skip_reproject,
            stages=StagesConfig(
                chips=ChipsConfig(
                    min_coverage=min_coverage,
                    drop_border_chips=drop_border_chips,
                ),
                splits=SplitsConfig(split_type=split_type, split_percents=split_percents),
                masks=MasksConfig(
                    mask_types=list(mask_types)
                    if mask_types is not None
                    else list(VALID_MASK_TYPES),
                    resolution=resolution,
                    workers=num_workers,
                    presence_only=presence_only,
                ),
                select_images=SelectImagesConfig(enabled=False),
                download_images=DownloadImagesConfig(enabled=False),
            ),
        )
        config.validate()
        return config

    # ---- validation -----------------------------------------------------

    def validate(self) -> None:
        """Validate values, raising ConfigError on the first problem found."""
        split_type = self.stages.splits.split_type
        if split_type is not None and split_type not in splits.SPLIT_TYPE_CHOICES:
            raise ConfigError(
                f"Invalid split_type '{split_type}'. "
                f"Must be one of: {splits.SPLIT_TYPE_CHOICES_STR}."
            )

        try:
            resolved = splits.validate_split_percents(self.stages.splits.split_percents)
        except ValueError as err:
            raise ConfigError(f"Invalid split_percents: {err}") from err
        self.stages.splits.split_percents = resolved

        for mask_type in self.stages.masks.mask_types:
            if mask_type not in VALID_MASK_TYPES:
                raise ConfigError(
                    f"Invalid mask type '{mask_type}'. "
                    f"Must be one of: {', '.join(VALID_MASK_TYPES)}."
                )
        if not self.stages.masks.mask_types:
            raise ConfigError("masks.mask_types must list at least one mask type.")

    # ---- provenance -----------------------------------------------------

    def config_dict(self) -> dict[str, Any]:
        """Return the config as a plain dict with all defaults resolved."""
        data = asdict(self)
        # asdict turns the split_percents tuple into a list; keep it a list for YAML.
        data["stages"]["splits"]["split_percents"] = list(self.stages.splits.split_percents)
        return data

    def provenance_dict(self, generated_at: datetime | None = None) -> dict[str, Any]:
        """Return a resolved provenance record for output and STAC embedding."""
        stamp = generated_at or datetime.now(UTC)
        return {
            "ftwd_version": __version__,
            "config_schema_version": CONFIG_SCHEMA_VERSION,
            "generated_at": stamp.isoformat(),
            "config": self.config_dict(),
        }


def load_config(path: str | Path) -> DatasetConfig:
    """Load and validate a YAML config file into a DatasetConfig."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as err:
        raise ConfigError(f"Could not parse YAML config {config_path}: {err}") from err

    if raw is None:
        raise ConfigError(f"Config file is empty: {config_path}")

    config = DatasetConfig.from_dict(raw)

    # Resolve the optional class filter relative to the config file's directory.
    filter_ref = config.stages.masks.class_filter
    if filter_ref is not None:
        filter_path = Path(filter_ref)
        if not filter_path.is_absolute():
            filter_path = (config_path.parent / filter_path).resolve()
        config.class_filter = ClassFilter.from_file(filter_path)

    # Resolve an optional local grid file relative to the config file's directory.
    grid_ref = config.stages.chips.grid_file
    if grid_ref is not None:
        grid_path = Path(grid_ref)
        if not grid_path.is_absolute():
            grid_path = (config_path.parent / grid_path).resolve()
        config.stages.chips.grid_file = str(grid_path)

    return config


def write_provenance_file(
    provenance: dict[str, Any],
    output_dir: str | Path,
    filename: str = "ftwd-config.resolved.yaml",
) -> Path:
    """Write a resolved provenance record (from :meth:`provenance_dict`) to disk."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(yaml.safe_dump(provenance, sort_keys=False, default_flow_style=False))
    return out_path


# ---- internal helpers ---------------------------------------------------


def _reject_unknown(data: dict[str, Any], known: set[str], context: str) -> None:
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in {context}: {', '.join(sorted(unknown))}. "
            f"Allowed keys: {', '.join(sorted(known))}."
        )


def _opt_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _as_str_list(value: Any, key: str, source: Path) -> list[str]:
    """Coerce a YAML list into a list of unique strings (order preserved)."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ClassFilterError(f"Class filter {source.name}: '{key}' must be a list.")
    seen: dict[str, None] = {}
    for item in value:
        if isinstance(item, bool) or not isinstance(item, str | int | float):
            raise ClassFilterError(
                f"Class filter {source.name}: '{key}' entries must be strings or "
                f"numbers, got {item!r}."
            )
        seen.setdefault(str(item), None)
    return list(seen)


_STAGE_TYPES: dict[str, type] = {
    "chips": ChipsConfig,
    "splits": SplitsConfig,
    "masks": MasksConfig,
    "select_images": SelectImagesConfig,
    "download_images": DownloadImagesConfig,
}


def _build_stages(stages_data: dict[str, Any]) -> StagesConfig:
    _reject_unknown(stages_data, set(_STAGE_TYPES), context="stages")
    kwargs: dict[str, Any] = {}
    for name, stage_type in _STAGE_TYPES.items():
        section = stages_data.get(name)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ConfigError(f"stages.{name} must be a mapping.")
        kwargs[name] = _build_stage(stage_type, section, name)
    return StagesConfig(**kwargs)


def _build_stage(stage_type: type, section: dict[str, Any], name: str) -> Any:
    known = {f.name for f in fields(stage_type)}
    _reject_unknown(section, known, context=f"stages.{name}")
    value = stage_type(**section)
    # Normalize split_percents (YAML lists) into a tuple.
    if isinstance(value, SplitsConfig) and value.split_percents is not None:
        value.split_percents = tuple(value.split_percents)  # type: ignore[assignment]
    return value
