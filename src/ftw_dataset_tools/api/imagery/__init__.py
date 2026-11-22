"""Imagery pipeline for selecting and downloading satellite imagery."""

from ftw_dataset_tools._lazy import lazy_exports

# The names the commands and the pipeline import from the package itself.
__getattr__, __all__ = lazy_exports(
    __name__,
    {
        "catalog_ops": ("clear_chip_selections", "find_collection_dir", "get_imagery_stats"),
        "download_workflow": ("download_imagery_for_catalog",),
        "selection_workflow": ("find_chip_items",),
    },
)
