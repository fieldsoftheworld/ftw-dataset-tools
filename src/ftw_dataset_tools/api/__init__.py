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
from ftw_dataset_tools.api.grid import CRSError, GetGridResult, get_grid

__all__ = [
    "CRSError",
    "CRSInfo",
    "CRSMismatchError",
    "FieldStatsResult",
    "GetGridResult",
    "ReprojectResult",
    "add_field_stats",
    "detect_crs",
    "get_grid",
    "reproject",
    "validate_crs_match",
]
