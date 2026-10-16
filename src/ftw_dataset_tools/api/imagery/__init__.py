"""Imagery pipeline for selecting and downloading satellite imagery."""

from ftw_dataset_tools._lazy import lazy_exports

__getattr__, __all__ = lazy_exports(
    __name__,
    {
        "catalog_ops": (
            "ClearResult",
            "ImageryStats",
            "chip_dir_for_item",
            "clear_chip_selections",
            "find_collection_dir",
            "get_imagery_stats",
            "has_existing_scenes",
            "iter_chip_dirs",
        ),
        "cloud_analysis": ("calculate_pixel_cloud_cover", "read_scl_chip_stats"),
        "crop_calendar": ("CropCalendarDates", "get_crop_calendar_dates"),
        "download_workflow": (
            "DownloadWorkflowResult",
            "download_imagery_for_catalog",
            "find_s2_child_items",
        ),
        "image_download": (
            "DownloadResult",
            "ProcessedSceneResult",
            "download_and_clip_scene",
            "process_downloaded_scene",
        ),
        "preview_conversion": (
            "ConversionResult",
            "conversion_summary_line",
            "convert_previews_for_catalog",
        ),
        "preview_workflow": ("preview_imagery_for_catalog", "preview_summary_line"),
        "progress": ("ImageryProgressBar", "SelectionStats"),
        "scene_selection": ("SceneSelectionResult", "SelectedScene", "select_scenes_for_chip"),
        "selection_workflow": (
            "SelectionWorkflowResult",
            "find_chip_items",
            "select_imagery_for_catalog",
        ),
        "settings": (
            "BANDS_OF_INTEREST",
            "CROP_CALENDAR_BASE_URL",
            "CROP_CALENDAR_FILES",
            "STAC_URL",
        ),
        "stac_child_items": ("create_child_items_from_selection",),
    },
)
