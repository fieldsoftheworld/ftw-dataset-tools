"""Imagery pipeline for selecting and downloading satellite imagery."""

from ftw_dataset_tools.api.imagery.cloud_analysis import calculate_pixel_cloud_cover
from ftw_dataset_tools.api.imagery.crop_calendar import (
    CropCalendarDates,
    get_crop_calendar_dates,
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
from ftw_dataset_tools.api.imagery.settings import (
    BANDS_OF_INTEREST,
    CROP_CALENDAR_BASE_URL,
    CROP_CALENDAR_FILES,
    STAC_URL,
)

__all__ = [
    "BANDS_OF_INTEREST",
    "CROP_CALENDAR_BASE_URL",
    "CROP_CALENDAR_FILES",
    "STAC_URL",
    "CropCalendarDates",
    "DownloadResult",
    "ImageryProgressBar",
    "ProcessedSceneResult",
    "SceneSelectionResult",
    "SelectedScene",
    "SelectionStats",
    "calculate_pixel_cloud_cover",
    "download_and_clip_scene",
    "get_crop_calendar_dates",
    "process_downloaded_scene",
    "select_scenes_for_chip",
]
