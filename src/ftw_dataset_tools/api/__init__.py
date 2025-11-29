"""Core API for FTW Dataset Tools."""

from ftw_dataset_tools.api.field_stats import FieldStatsResult, add_field_stats
from ftw_dataset_tools.api.geo import (
    CRSInfo,
    CRSMismatchError,
    ReprojectResult,
    detect_crs,
    reproject,
    validate_crs_match,
)

__all__ = [
    "CRSInfo",
    "CRSMismatchError",
    "FieldStatsResult",
    "ReprojectResult",
    "add_field_stats",
    "detect_crs",
    "reproject",
    "validate_crs_match",
]
