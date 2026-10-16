"""FTW Dataset Tools - CLI tools for creating Fields of the World benchmark dataset."""

from ftw_dataset_tools._lazy import lazy_exports

__version__ = "0.1.0"

__getattr__, _exports = lazy_exports(
    __name__,
    {
        "api.field_stats": ("FieldStatsResult", "add_field_stats"),
        "api.geo": ("CRSInfo", "CRSMismatchError", "ReprojectResult", "detect_crs", "reproject"),
    },
)
__all__ = ["__version__", *_exports]
