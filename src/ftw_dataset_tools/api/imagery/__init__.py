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
from ftw_dataset_tools.api.imagery.planet_client import (
    DEFAULT_BUFFER_DAYS as PLANET_DEFAULT_BUFFER_DAYS,
)
from ftw_dataset_tools.api.imagery.planet_client import (
    DEFAULT_NUM_ITERATIONS as PLANET_DEFAULT_NUM_ITERATIONS,
)
from ftw_dataset_tools.api.imagery.planet_client import (
    PLANET_STAC_URL,
    PLANET_TILES_URL,
    PlanetClient,
)
from ftw_dataset_tools.api.imagery.planet_client import (
    VALID_BANDS as PLANET_VALID_BANDS,
)
from ftw_dataset_tools.api.imagery.planet_download import (
    PlanetDownloadResult,
    download_and_clip_planet_scene,
)
from ftw_dataset_tools.api.imagery.planet_download import (
    activate_asset as activate_planet_asset,
)
from ftw_dataset_tools.api.imagery.planet_download import (
    wait_for_activation as wait_for_planet_activation,
)
from ftw_dataset_tools.api.imagery.planet_selection import (
    PlanetScene,
    PlanetSelectionResult,
    get_clear_coverage,
    select_planet_scenes_for_chip,
)
from ftw_dataset_tools.api.imagery.planet_selection import (
    generate_thumbnail as generate_planet_thumbnail,
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
    "PLANET_DEFAULT_BUFFER_DAYS",
    "PLANET_DEFAULT_NUM_ITERATIONS",
    "PLANET_STAC_URL",
    "PLANET_TILES_URL",
    "PLANET_VALID_BANDS",
    "STAC_URL",
    "ClearResult",
    "CropCalendarDates",
    "DownloadResult",
    "DownloadWorkflowResult",
    "ImageryProgressBar",
    "ImageryStats",
    "PlanetClient",
    "PlanetDownloadResult",
    "PlanetScene",
    "PlanetSelectionResult",
    "ProcessedSceneResult",
    "SceneSelectionResult",
    "SelectedScene",
    "SelectionStats",
    "SelectionWorkflowResult",
    "activate_planet_asset",
    "calculate_pixel_cloud_cover",
    "clear_chip_selections",
    "create_child_items_from_selection",
    "download_and_clip_planet_scene",
    "download_and_clip_scene",
    "download_imagery_for_catalog",
    "find_chip_items",
    "find_s2_child_items",
    "generate_planet_thumbnail",
    "get_clear_coverage",
    "get_crop_calendar_dates",
    "get_imagery_stats",
    "has_existing_scenes",
    "process_downloaded_scene",
    "select_imagery_for_catalog",
    "select_planet_scenes_for_chip",
    "select_scenes_for_chip",
    "wait_for_planet_activation",
]
