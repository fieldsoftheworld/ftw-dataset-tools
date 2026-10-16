"""Core API for FTW Dataset Tools."""

from ftw_dataset_tools._lazy import lazy_exports

__getattr__, __all__ = lazy_exports(
    __name__,
    {
        "field_stats": ("FieldStatsResult", "add_field_stats"),
        "ftw_grid": (
            "CreateFTWGridResult",
            "InvalidKmSizeError",
            "MultipleGZDError",
            "create_ftw_grid",
        ),
        "geo": (
            "CRSInfo",
            "CRSMismatchError",
            "ReprojectResult",
            "detect_crs",
            "reproject",
            "validate_crs_match",
        ),
        "grid": ("CRSError", "GetGridResult", "get_grid"),
        "stac": (
            "STACGenerationResult",
            "generate_stac_catalog",
            "get_temporal_extent_from_year",
        ),
    },
)
