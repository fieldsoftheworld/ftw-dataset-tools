"""Tests for Planet imagery selection module."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pystac
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from ftw_dataset_tools.api.imagery.crop_calendar import CropCalendarDates
from ftw_dataset_tools.api.imagery.planet_client import (
    DEFAULT_BUFFER_DAYS,
    DEFAULT_NUM_ITERATIONS,
    PLANET_STAC_URL,
    PLANET_TILES_URL,
    VALID_BANDS,
    PlanetClient,
)
from ftw_dataset_tools.api.imagery.planet_selection import (
    PlanetScene,
    PlanetSelectionResult,
    generate_thumbnail,
    get_clear_coverage,
    select_planet_scenes_for_chip,
)


class TestPlanetClient:
    """Tests for PlanetClient class."""

    def test_init_with_api_key(self) -> None:
        """Test client initialization with explicit API key."""
        client = PlanetClient(api_key="test_key")
        assert client.api_key == "test_key"

    def test_init_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test client initialization from environment variable."""
        monkeypatch.setenv("PL_API_KEY", "env_test_key")
        client = PlanetClient()
        assert client.api_key == "env_test_key"

    def test_init_no_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that missing API key raises ValueError."""
        monkeypatch.delenv("PL_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Planet API key required"):
            PlanetClient()

    def test_constants(self) -> None:
        """Test that constants are properly defined."""
        assert PLANET_STAC_URL == "https://api.planet.com/x/data/"
        assert PLANET_TILES_URL == "https://tiles.planet.com/data/v1/"
        assert DEFAULT_BUFFER_DAYS == 14
        assert DEFAULT_NUM_ITERATIONS == 3
        assert "red" in VALID_BANDS
        assert "green" in VALID_BANDS
        assert "blue" in VALID_BANDS
        assert "nir" in VALID_BANDS


class TestPlanetScene:
    """Tests for PlanetScene dataclass."""

    def test_id_property(self) -> None:
        """Test that id property returns item id."""
        mock_item = MagicMock(spec=pystac.Item)
        mock_item.id = "test_scene_id"

        scene = PlanetScene(
            item=mock_item,
            season="planting",
            clear_coverage=95.0,
            datetime=datetime(2024, 6, 15, tzinfo=UTC),
            stac_url="https://example.com/scene",
        )

        assert scene.id == "test_scene_id"

    def test_cloud_cover_property(self) -> None:
        """Test that cloud_cover is calculated from clear_coverage."""
        mock_item = MagicMock(spec=pystac.Item)
        mock_item.id = "test_scene"

        scene = PlanetScene(
            item=mock_item,
            season="harvest",
            clear_coverage=97.5,
            datetime=datetime(2024, 9, 15, tzinfo=UTC),
            stac_url="https://example.com/scene",
        )

        assert scene.cloud_cover == 2.5


class TestPlanetSelectionResult:
    """Tests for PlanetSelectionResult dataclass."""

    def test_success_with_both_scenes(self) -> None:
        """Test success is True when both scenes are present."""
        mock_item = MagicMock(spec=pystac.Item)
        mock_item.id = "test"

        planting = PlanetScene(
            item=mock_item,
            season="planting",
            clear_coverage=95.0,
            datetime=datetime(2024, 6, 15, tzinfo=UTC),
            stac_url="",
        )
        harvest = PlanetScene(
            item=mock_item,
            season="harvest",
            clear_coverage=95.0,
            datetime=datetime(2024, 9, 15, tzinfo=UTC),
            stac_url="",
        )

        result = PlanetSelectionResult(
            chip_id="chip_001",
            bbox=(0.0, 0.0, 1.0, 1.0),
            year=2024,
            crop_calendar=CropCalendarDates(150, 250),
            planting_scene=planting,
            harvest_scene=harvest,
        )

        assert result.success is True

    def test_success_false_with_missing_planting(self) -> None:
        """Test success is False when planting scene is missing."""
        mock_item = MagicMock(spec=pystac.Item)
        mock_item.id = "test"

        harvest = PlanetScene(
            item=mock_item,
            season="harvest",
            clear_coverage=95.0,
            datetime=datetime(2024, 9, 15, tzinfo=UTC),
            stac_url="",
        )

        result = PlanetSelectionResult(
            chip_id="chip_001",
            bbox=(0.0, 0.0, 1.0, 1.0),
            year=2024,
            crop_calendar=CropCalendarDates(150, 250),
            planting_scene=None,
            harvest_scene=harvest,
        )

        assert result.success is False

    def test_success_false_with_missing_harvest(self) -> None:
        """Test success is False when harvest scene is missing."""
        mock_item = MagicMock(spec=pystac.Item)
        mock_item.id = "test"

        planting = PlanetScene(
            item=mock_item,
            season="planting",
            clear_coverage=95.0,
            datetime=datetime(2024, 6, 15, tzinfo=UTC),
            stac_url="",
        )

        result = PlanetSelectionResult(
            chip_id="chip_001",
            bbox=(0.0, 0.0, 1.0, 1.0),
            year=2024,
            crop_calendar=CropCalendarDates(150, 250),
            planting_scene=planting,
            harvest_scene=None,
        )

        assert result.success is False


class TestGetClearCoverage:
    """Tests for get_clear_coverage function."""

    def test_returns_fallback_on_error(self) -> None:
        """Test that function returns 0.0 on API error."""
        client = MagicMock(spec=PlanetClient)
        client.api_key = "test_key"

        with patch("httpx.post") as mock_post:
            mock_post.side_effect = Exception("API error")
            result = get_clear_coverage(
                client, "test_scene", {"type": "Polygon", "coordinates": []}
            )

        assert result == 0.0

    def test_parses_api_response(self) -> None:
        """Test that function correctly parses API response."""
        client = MagicMock(spec=PlanetClient)
        client.api_key = "test_key"

        mock_response = MagicMock()
        mock_response.json.return_value = {"clear_percent": 0.95}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            result = get_clear_coverage(
                client, "test_scene", {"type": "Polygon", "coordinates": []}
            )

        assert result == 95.0


class TestGenerateThumbnail:
    """Tests for generate_thumbnail function."""

    def test_returns_none_on_error(self, tmp_path: Path) -> None:
        """Test that function returns None on API error."""
        client = MagicMock(spec=PlanetClient)
        client.api_key = "test_key"

        with patch("httpx.get") as mock_get:
            mock_get.side_effect = Exception("API error")
            result = generate_thumbnail(client, "test_scene", tmp_path / "thumb.png")

        assert result is None

    def test_saves_thumbnail_on_success(self, tmp_path: Path) -> None:
        """Test that function saves thumbnail on success."""
        client = MagicMock(spec=PlanetClient)
        client.api_key = "test_key"

        mock_response = MagicMock()
        mock_response.content = b"fake png data"
        mock_response.raise_for_status = MagicMock()

        output_path = tmp_path / "thumb.png"

        with patch("httpx.get", return_value=mock_response):
            result = generate_thumbnail(client, "test_scene", output_path)

        assert result == output_path
        assert output_path.exists()
        assert output_path.read_bytes() == b"fake png data"


class TestSelectPlanetScenesForChip:
    """Tests for select_planet_scenes_for_chip function."""

    @patch("ftw_dataset_tools.api.imagery.planet_selection.get_crop_calendar_dates")
    @patch("ftw_dataset_tools.api.imagery.planet_selection._query_planet_stac")
    @patch("ftw_dataset_tools.api.imagery.planet_selection.get_clear_coverage")
    def test_successful_selection(
        self,
        mock_coverage: MagicMock,
        mock_query: MagicMock,
        mock_crop_calendar: MagicMock,
    ) -> None:
        """Test successful scene selection for both seasons."""
        client = MagicMock(spec=PlanetClient)

        # Mock crop calendar
        mock_crop_calendar.return_value = CropCalendarDates(150, 250)

        # Mock STAC query results
        planting_item = pystac.Item(
            id="planting_scene",
            geometry={"type": "Point", "coordinates": [0, 0]},
            bbox=[0, 0, 1, 1],
            datetime=datetime(2024, 6, 1, tzinfo=UTC),
            properties={"eo:cloud_cover": 5.0},
        )
        harvest_item = pystac.Item(
            id="harvest_scene",
            geometry={"type": "Point", "coordinates": [0, 0]},
            bbox=[0, 0, 1, 1],
            datetime=datetime(2024, 9, 1, tzinfo=UTC),
            properties={"eo:cloud_cover": 5.0},
        )

        mock_query.side_effect = [[planting_item], [harvest_item]]

        # Mock coverage API - return high clear coverage
        mock_coverage.return_value = 98.0

        result = select_planet_scenes_for_chip(
            client=client,
            chip_id="test_chip",
            bbox=(0.0, 0.0, 1.0, 1.0),
            year=2024,
        )

        assert result.success is True
        assert result.planting_scene is not None
        assert result.harvest_scene is not None
        assert result.planting_scene.id == "planting_scene"
        assert result.harvest_scene.id == "harvest_scene"

    @patch("ftw_dataset_tools.api.imagery.planet_selection.get_crop_calendar_dates")
    def test_crop_calendar_error(self, mock_crop_calendar: MagicMock) -> None:
        """Test handling of crop calendar errors."""
        client = MagicMock(spec=PlanetClient)
        mock_crop_calendar.side_effect = ValueError("No crop calendar data")

        result = select_planet_scenes_for_chip(
            client=client,
            chip_id="test_chip",
            bbox=(0.0, 0.0, 1.0, 1.0),
            year=2024,
        )

        assert result.success is False
        assert "Crop calendar error" in result.skipped_reason

    @patch("ftw_dataset_tools.api.imagery.planet_selection.get_crop_calendar_dates")
    @patch("ftw_dataset_tools.api.imagery.planet_selection._query_planet_stac")
    def test_no_scenes_found(
        self,
        mock_query: MagicMock,
        mock_crop_calendar: MagicMock,
    ) -> None:
        """Test handling when no scenes are found."""
        client = MagicMock(spec=PlanetClient)
        mock_crop_calendar.return_value = CropCalendarDates(150, 250)
        mock_query.return_value = []

        result = select_planet_scenes_for_chip(
            client=client,
            chip_id="test_chip",
            bbox=(0.0, 0.0, 1.0, 1.0),
            year=2024,
        )

        assert result.success is False
        assert result.skipped_reason is not None

    @patch("ftw_dataset_tools.api.imagery.planet_selection.get_crop_calendar_dates")
    @patch("ftw_dataset_tools.api.imagery.planet_selection._query_planet_stac")
    @patch("ftw_dataset_tools.api.imagery.planet_selection.get_clear_coverage")
    def test_iteration_logic(
        self,
        mock_coverage: MagicMock,
        mock_query: MagicMock,
        mock_crop_calendar: MagicMock,
    ) -> None:
        """Test that buffer expansion iterations work correctly."""
        client = MagicMock(spec=PlanetClient)
        mock_crop_calendar.return_value = CropCalendarDates(150, 250)

        # First iteration: no planting scene
        # Second iteration: planting scene found
        planting_item = pystac.Item(
            id="planting_scene_expanded",
            geometry={"type": "Point", "coordinates": [0, 0]},
            bbox=[0, 0, 1, 1],
            datetime=datetime(2024, 6, 1, tzinfo=UTC),
            properties={"eo:cloud_cover": 5.0},
        )
        harvest_item = pystac.Item(
            id="harvest_scene",
            geometry={"type": "Point", "coordinates": [0, 0]},
            bbox=[0, 0, 1, 1],
            datetime=datetime(2024, 9, 1, tzinfo=UTC),
            properties={"eo:cloud_cover": 5.0},
        )

        # First iteration: empty for planting, harvest found
        # Second iteration: planting found
        mock_query.side_effect = [
            [],  # First planting query
            [harvest_item],  # First harvest query
            [planting_item],  # Second planting query (expanded buffer)
        ]

        mock_coverage.return_value = 98.0

        result = select_planet_scenes_for_chip(
            client=client,
            chip_id="test_chip",
            bbox=(0.0, 0.0, 1.0, 1.0),
            year=2024,
            buffer_days=14,
            num_iterations=3,
        )

        assert result.success is True
        assert result.iterations_used >= 1


@pytest.mark.network
class TestPlanetSelectionNetwork:
    """Network-dependent tests for Planet selection.

    These tests require a valid PL_API_KEY environment variable.
    Skip with: pytest -m "not network"
    """

    def test_validate_auth_real(self) -> None:
        """Test real authentication validation."""
        import os

        api_key = os.environ.get("PL_API_KEY")
        if not api_key:
            pytest.skip("PL_API_KEY not set")

        client = PlanetClient()
        assert client.validate_auth() is True
