"""Shared test fixtures for imagery workflow tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pystac
import pytest

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path
    from unittest.mock import MagicMock


# Sentinel value to distinguish "use default" from "explicitly None"
_BBOX_DEFAULT = object()


def create_mock_stac_item(
    item_id: str,
    bbox: tuple[float, float, float, float] | None | object = _BBOX_DEFAULT,
    geometry: dict | None = None,
    dt: datetime | None = None,
    properties: dict | None = None,
    assets: dict[str, pystac.Asset] | None = None,
    links: list[pystac.Link] | None = None,
) -> pystac.Item:
    """Create a mock pystac.Item for testing.

    Args:
        item_id: Item identifier
        bbox: Bounding box (minx, miny, maxx, maxy). Defaults to (0.0, 0.0, 1.0, 1.0).
              Pass None explicitly to create an item without bbox.
        geometry: GeoJSON geometry dict. Defaults to a simple polygon
        dt: Item datetime. Defaults to now
        properties: Additional properties dict
        assets: Assets to add to the item
        links: Links to add to the item

    Returns:
        A pystac.Item instance
    """
    # Handle sentinel default value
    if bbox is _BBOX_DEFAULT:
        bbox = (0.0, 0.0, 1.0, 1.0)

    if geometry is None:
        if bbox is not None:
            minx, miny, maxx, maxy = bbox
            geometry = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [minx, miny],
                        [maxx, miny],
                        [maxx, maxy],
                        [minx, maxy],
                        [minx, miny],
                    ]
                ],
            }
        else:
            # Default geometry for None bbox case
            geometry = {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            }

    if dt is None:
        dt = datetime.now(UTC)

    item = pystac.Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=dt,
        properties=properties or {},
    )

    if assets:
        for asset_key, asset in assets.items():
            item.assets[asset_key] = asset

    if links:
        for link in links:
            item.add_link(link)

    return item


def create_mock_s2_assets() -> dict[str, pystac.Asset]:
    """Create mock Sentinel-2 band assets for testing."""
    bands = ["red", "green", "blue", "nir", "scl", "visual", "cloud"]
    assets = {}
    for band in bands:
        assets[band] = pystac.Asset(
            href=f"https://example.com/data/{band}.tif",
            media_type="image/tiff; application=geotiff",
            title=band.upper(),
        )
    return assets


@pytest.fixture
def mock_bbox() -> tuple[float, float, float, float]:
    """Standard test bounding box."""
    return (10.0, 50.0, 10.01, 50.01)


@pytest.fixture
def mock_parent_item(mock_bbox: tuple[float, float, float, float]) -> pystac.Item:
    """Create a mock parent chip STAC item."""
    return create_mock_stac_item(
        item_id="chip_001",
        bbox=mock_bbox,
        properties={"id": "chip_001"},
    )


@pytest.fixture
def mock_s2_child_item(mock_bbox: tuple[float, float, float, float]) -> pystac.Item:
    """Create a mock S2 child STAC item with assets."""
    return create_mock_stac_item(
        item_id="chip_001_planting_s2",
        bbox=mock_bbox,
        properties={"eo:cloud_cover": 1.5},
        assets=create_mock_s2_assets(),
    )


@pytest.fixture
def mock_selected_scene(
    mock_bbox: tuple[float, float, float, float],
) -> MagicMock:
    """Create a mock SelectedScene for testing."""
    from ftw_dataset_tools.api.imagery.scene_selection import SelectedScene

    mock_item = create_mock_stac_item(
        item_id="S2A_TILE_20240615_test",
        bbox=mock_bbox,
        properties={"eo:cloud_cover": 1.5},
        assets=create_mock_s2_assets(),
    )

    return SelectedScene(
        item=mock_item,
        season="planting",
        cloud_cover=1.5,
        datetime=datetime(2024, 6, 15, 10, 0, 0, tzinfo=UTC),
        stac_url="https://example.com/stac/S2A_TILE_20240615_test",
    )


@pytest.fixture
def mock_crop_calendar() -> MagicMock:
    """Create a mock CropCalendarDates object."""
    from ftw_dataset_tools.api.imagery.crop_calendar import CropCalendarDates

    return CropCalendarDates(planting_day=150, harvest_day=250)


@pytest.fixture
def mock_selection_result(
    mock_selected_scene: MagicMock,
    mock_crop_calendar: MagicMock,
    mock_bbox: tuple[float, float, float, float],
) -> MagicMock:
    """Create a mock SceneSelectionResult with both scenes."""
    from ftw_dataset_tools.api.imagery.scene_selection import (
        SceneSelectionResult,
        SelectedScene,
    )

    # Create harvest scene
    harvest_item = create_mock_stac_item(
        item_id="S2A_TILE_20240915_test",
        bbox=mock_bbox,
        properties={"eo:cloud_cover": 2.0},
        assets=create_mock_s2_assets(),
    )

    harvest_scene = SelectedScene(
        item=harvest_item,
        season="harvest",
        cloud_cover=2.0,
        datetime=datetime(2024, 9, 15, 10, 0, 0, tzinfo=UTC),
        stac_url="https://example.com/stac/S2A_TILE_20240915_test",
    )

    return SceneSelectionResult(
        chip_id="chip_001",
        bbox=mock_bbox,
        year=2024,
        crop_calendar=mock_crop_calendar,
        planting_scene=mock_selected_scene,
        harvest_scene=harvest_scene,
        planting_buffer_used=14,
        harvest_buffer_used=14,
        expansions_performed=0,
    )


@pytest.fixture
def mock_selection_result_planting_only(
    mock_selected_scene: MagicMock,
    mock_crop_calendar: MagicMock,
    mock_bbox: tuple[float, float, float, float],
) -> MagicMock:
    """Create a mock SceneSelectionResult with only planting scene."""
    from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult

    return SceneSelectionResult(
        chip_id="chip_001",
        bbox=mock_bbox,
        year=2024,
        crop_calendar=mock_crop_calendar,
        planting_scene=mock_selected_scene,
        harvest_scene=None,
        skipped_reason="No cloud-free harvest scene found",
        planting_buffer_used=14,
        harvest_buffer_used=42,
        expansions_performed=2,
    )


@pytest.fixture
def mock_selection_result_harvest_only(
    mock_crop_calendar: MagicMock,
    mock_bbox: tuple[float, float, float, float],
) -> MagicMock:
    """Create a mock SceneSelectionResult with only harvest scene."""
    from ftw_dataset_tools.api.imagery.scene_selection import (
        SceneSelectionResult,
        SelectedScene,
    )

    harvest_item = create_mock_stac_item(
        item_id="S2A_TILE_20240915_test",
        bbox=mock_bbox,
        properties={"eo:cloud_cover": 2.0},
        assets=create_mock_s2_assets(),
    )

    harvest_scene = SelectedScene(
        item=harvest_item,
        season="harvest",
        cloud_cover=2.0,
        datetime=datetime(2024, 9, 15, 10, 0, 0, tzinfo=UTC),
        stac_url="https://example.com/stac/S2A_TILE_20240915_test",
    )

    return SceneSelectionResult(
        chip_id="chip_001",
        bbox=mock_bbox,
        year=2024,
        crop_calendar=mock_crop_calendar,
        planting_scene=None,
        harvest_scene=harvest_scene,
        skipped_reason="No cloud-free planting scene found",
        planting_buffer_used=42,
        harvest_buffer_used=14,
        expansions_performed=2,
    )


@pytest.fixture
def mock_catalog_with_chips(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a catalog directory with parent chip items.

    Creates a temporary directory structure with parent chip STAC items.

    Yields:
        Path to the catalog directory
    """
    for chip_id in ["chip_001", "chip_002"]:
        chip_dir = tmp_path / chip_id
        chip_dir.mkdir()

        item = create_mock_stac_item(
            item_id=chip_id,
            bbox=(10.0, 50.0, 10.01, 50.01),
        )
        item.set_self_href(str(chip_dir / f"{chip_id}.json"))
        item.save_object(dest_href=str(chip_dir / f"{chip_id}.json"))

    yield tmp_path


@pytest.fixture
def mock_catalog_with_s2_children(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a catalog directory with S2 child items.

    Creates a temporary directory structure with planting and harvest S2 child items.

    Yields:
        Path to the catalog directory
    """
    for chip_id in ["chip_001", "chip_002"]:
        chip_dir = tmp_path / chip_id
        chip_dir.mkdir()

        # Create parent item
        parent_item = create_mock_stac_item(
            item_id=chip_id,
            bbox=(10.0, 50.0, 10.01, 50.01),
        )
        parent_item.set_self_href(str(chip_dir / f"{chip_id}.json"))
        parent_item.save_object(dest_href=str(chip_dir / f"{chip_id}.json"))

        # Create planting child item
        planting_id = f"{chip_id}_planting_s2"
        planting_item = create_mock_stac_item(
            item_id=planting_id,
            bbox=(10.0, 50.0, 10.01, 50.01),
            properties={"eo:cloud_cover": 1.5, "ftw:season": "planting"},
            assets=create_mock_s2_assets(),
        )
        planting_item.set_self_href(str(chip_dir / f"{planting_id}.json"))
        planting_item.save_object(dest_href=str(chip_dir / f"{planting_id}.json"))

        # Create harvest child item
        harvest_id = f"{chip_id}_harvest_s2"
        harvest_item = create_mock_stac_item(
            item_id=harvest_id,
            bbox=(10.0, 50.0, 10.01, 50.01),
            properties={"eo:cloud_cover": 2.0, "ftw:season": "harvest"},
            assets=create_mock_s2_assets(),
        )
        harvest_item.set_self_href(str(chip_dir / f"{harvest_id}.json"))
        harvest_item.save_object(dest_href=str(chip_dir / f"{harvest_id}.json"))

    yield tmp_path


@pytest.fixture
def mock_catalog_with_existing_scenes(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a catalog directory with parent items that already have scene links.

    Yields:
        Path to the catalog directory
    """
    chip_dir = tmp_path / "chip_001"
    chip_dir.mkdir()

    item = create_mock_stac_item(
        item_id="chip_001",
        bbox=(10.0, 50.0, 10.01, 50.01),
    )
    item.set_self_href(str(chip_dir / "chip_001.json"))

    # Add existing scene links
    item.add_link(
        pystac.Link(
            rel="ftw:planting",
            target="./chip_001_planting_s2.json",
            media_type="application/json",
        )
    )
    item.add_link(
        pystac.Link(
            rel="ftw:harvest",
            target="./chip_001_harvest_s2.json",
            media_type="application/json",
        )
    )

    item.save_object(dest_href=str(chip_dir / "chip_001.json"))

    yield tmp_path


@pytest.fixture
def mock_catalog_empty(tmp_path: Path) -> Generator[Path, None, None]:
    """Create an empty catalog directory.

    Yields:
        Path to the empty catalog directory
    """
    yield tmp_path


@pytest.fixture
def mock_catalog_with_hidden_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a catalog directory with a hidden subdirectory.

    Yields:
        Path to the catalog directory
    """
    # Create hidden directory with an item
    hidden_dir = tmp_path / ".hidden"
    hidden_dir.mkdir()
    hidden_item = create_mock_stac_item(item_id="hidden_chip")
    hidden_item.set_self_href(str(hidden_dir / "hidden_chip.json"))
    hidden_item.save_object(dest_href=str(hidden_dir / "hidden_chip.json"))

    # Create normal directory with an item
    normal_dir = tmp_path / "chip_001"
    normal_dir.mkdir()
    normal_item = create_mock_stac_item(
        item_id="chip_001",
        bbox=(10.0, 50.0, 10.01, 50.01),
    )
    normal_item.set_self_href(str(normal_dir / "chip_001.json"))
    normal_item.save_object(dest_href=str(normal_dir / "chip_001.json"))

    yield tmp_path


@pytest.fixture
def mock_catalog_with_invalid_json(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a catalog directory with an invalid JSON file.

    Yields:
        Path to the catalog directory
    """
    # Create directory with invalid JSON
    chip_dir = tmp_path / "invalid_chip"
    chip_dir.mkdir()
    invalid_json = chip_dir / "invalid_chip.json"
    invalid_json.write_text("{ invalid json content }")

    # Create directory with valid item
    valid_dir = tmp_path / "chip_001"
    valid_dir.mkdir()
    valid_item = create_mock_stac_item(
        item_id="chip_001",
        bbox=(10.0, 50.0, 10.01, 50.01),
    )
    valid_item.set_self_href(str(valid_dir / "chip_001.json"))
    valid_item.save_object(dest_href=str(valid_dir / "chip_001.json"))

    yield tmp_path


@pytest.fixture
def mock_catalog_no_bbox(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a catalog directory with an item that has no bbox.

    Yields:
        Path to the catalog directory
    """
    chip_dir = tmp_path / "chip_001"
    chip_dir.mkdir()

    item = create_mock_stac_item(
        item_id="chip_001",
        bbox=None,
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
    )
    item.set_self_href(str(chip_dir / "chip_001.json"))
    item.save_object(dest_href=str(chip_dir / "chip_001.json"))

    yield tmp_path
