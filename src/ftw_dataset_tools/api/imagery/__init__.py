"""Imagery pipeline for selecting and downloading satellite imagery."""

from ftw_dataset_tools.api.imagery.catalog_ops import (
    ClearResult,
    ImageryStats,
    clear_chip_selections,
    get_imagery_stats,
    has_existing_scenes,
)
from ftw_dataset_tools.api.imagery.cloud_analysis import calculate_pixel_cloud_cover
from ftw_dataset_tools.api.imagery.crop_calendar import (
    CropCalendarDates,
    get_crop_calendar_dates,
)
from ftw_dataset_tools.api.imagery.download_workflow import (
    DownloadWorkflowResult,
    download_imagery_for_catalog,
    find_s2_child_items,
)
from ftw_dataset_tools.api.imagery.image_download import (
    DownloadResult,
    ProcessedSceneResult,
    download_and_clip_scene,
    process_downloaded_scene,
)
from ftw_dataset_tools.api.imagery.progress import ImageryProgressBar, SelectionStats
from ftw_dataset_tools.api.imagery.scene_selection import (
    SceneSelectionResult,
    SelectedScene,
    select_scenes_for_chip,
)
from ftw_dataset_tools.api.imagery.selection_workflow import (
    SelectionWorkflowResult,
    find_chip_items,
    select_imagery_for_catalog,
)
from ftw_dataset_tools.api.imagery.settings import (
    BANDS_OF_INTEREST,
    CROP_CALENDAR_BASE_URL,
    CROP_CALENDAR_FILES,
    STAC_URL,
)
from ftw_dataset_tools.api.imagery.stac_child_items import create_child_items_from_selection

__all__ = [
    "BANDS_OF_INTEREST",
    "CROP_CALENDAR_BASE_URL",
    "CROP_CALENDAR_FILES",
    "STAC_URL",
    "ClearResult",
    "CropCalendarDates",
    "DownloadResult",
    "DownloadWorkflowResult",
    "ImageryProgressBar",
    "ImageryStats",
    "ProcessedSceneResult",
    "SceneSelectionResult",
    "SelectedScene",
    "SelectionStats",
    "SelectionWorkflowResult",
    "calculate_pixel_cloud_cover",
    "clear_chip_selections",
    "create_child_items_from_selection",
    "download_and_clip_scene",
    "download_imagery_for_catalog",
    "find_chip_items",
    "find_s2_child_items",
    "get_crop_calendar_dates",
    "get_imagery_stats",
    "has_existing_scenes",
    "process_downloaded_scene",
    "select_imagery_for_catalog",
    "select_scenes_for_chip",
]
