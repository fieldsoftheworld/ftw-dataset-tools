"""FTW Dataset Tools - CLI tools for creating Fields of the World benchmark dataset."""

from ftw_dataset_tools.api.field_stats import FieldStatsResult, add_field_stats
from ftw_dataset_tools.api.geo import (
    CRSInfo,
    CRSMismatchError,
    ReprojectResult,
    detect_crs,
    reproject,
)

__version__ = "0.1.0"

__all__ = [
    "CRSInfo",
    "CRSMismatchError",
    "FieldStatsResult",
    "ReprojectResult",
    "__version__",
    "add_field_stats",
    "detect_crs",
    "reproject",
]
