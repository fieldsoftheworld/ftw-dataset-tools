"""Tests for Planet imagery download module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from ftw_dataset_tools.api.imagery.planet_client import PlanetClient
from ftw_dataset_tools.api.imagery.planet_download import (
    DEFAULT_ASSET_TYPE,
    VALID_ASSET_TYPES,
    PlanetDownloadResult,
    activate_asset,
    download_and_clip_planet_scene,
    wait_for_activation,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestConstants:
    """Tests for module constants."""

    def test_default_asset_type(self) -> None:
        """Test default asset type is defined."""
        assert DEFAULT_ASSET_TYPE == "ortho_analytic_4b"

    def test_valid_asset_types(self) -> None:
        """Test valid asset types list."""
        assert "ortho_analytic_4b" in VALID_ASSET_TYPES
        assert "ortho_analytic_8b" in VALID_ASSET_TYPES
        assert "ortho_visual" in VALID_ASSET_TYPES


class TestPlanetDownloadResult:
    """Tests for PlanetDownloadResult dataclass."""

    def test_success_result(self, tmp_path: Path) -> None:
        """Test creating a successful result."""
        result = PlanetDownloadResult(
            output_path=tmp_path / "output.tif",
            scene_id="test_scene",
            season="planting",
            bands=["red", "green", "blue", "nir"],
            width=256,
            height=256,
            crs="EPSG:32632",
            success=True,
        )

        assert result.success is True
        assert result.error is None
        assert result.scene_id == "test_scene"

    def test_failed_result(self, tmp_path: Path) -> None:
        """Test creating a failed result."""
        result = PlanetDownloadResult(
            output_path=tmp_path / "output.tif",
            scene_id="test_scene",
            season="harvest",
            bands=[],
            width=0,
            height=0,
            crs="",
            success=False,
            error="Activation timeout",
        )

        assert result.success is False
        assert result.error == "Activation timeout"


class TestActivateAsset:
    """Tests for activate_asset function."""

    def test_activates_inactive_asset(self) -> None:
        """Test that inactive assets are activated."""
        client = MagicMock(spec=PlanetClient)
        mock_pl = MagicMock()
        client.get_sdk_client.return_value = mock_pl

        # Asset is not active
        mock_pl.data.get_asset.return_value = {"status": "inactive"}

        activate_asset(client, "test_scene")

        mock_pl.data.activate_asset.assert_called_once()

    def test_skips_active_asset(self) -> None:
        """Test that already-active assets are not re-activated."""
        client = MagicMock(spec=PlanetClient)
        mock_pl = MagicMock()
        client.get_sdk_client.return_value = mock_pl

        # Asset is already active
        mock_pl.data.get_asset.return_value = {"status": "active"}

        activate_asset(client, "test_scene")

        mock_pl.data.activate_asset.assert_not_called()


class TestWaitForActivation:
    """Tests for wait_for_activation function."""

    def test_returns_immediately_if_active(self) -> None:
        """Test that function returns immediately for active assets."""
        client = MagicMock(spec=PlanetClient)
        mock_pl = MagicMock()
        client.get_sdk_client.return_value = mock_pl

        active_asset = {"status": "active", "location": "https://example.com"}
        mock_pl.data.get_asset.return_value = active_asset

        result = wait_for_activation(client, "test_scene")

        assert result["status"] == "active"
        mock_pl.data.wait_asset.assert_not_called()

    def test_waits_for_activation(self) -> None:
        """Test that function waits for non-active assets."""
        client = MagicMock(spec=PlanetClient)
        mock_pl = MagicMock()
        client.get_sdk_client.return_value = mock_pl

        # Asset is activating
        mock_pl.data.get_asset.return_value = {"status": "activating"}
        mock_pl.data.wait_asset.return_value = {
            "status": "active",
            "location": "https://example.com",
        }

        result = wait_for_activation(client, "test_scene")

        assert result["status"] == "active"
        mock_pl.data.wait_asset.assert_called_once()

    def test_activates_inactive_asset(self) -> None:
        """Test that function activates inactive assets before waiting."""
        client = MagicMock(spec=PlanetClient)
        mock_pl = MagicMock()
        client.get_sdk_client.return_value = mock_pl

        # Asset is inactive
        mock_pl.data.get_asset.return_value = {"status": "inactive"}
        mock_pl.data.wait_asset.return_value = {
            "status": "active",
            "location": "https://example.com",
        }

        wait_for_activation(client, "test_scene")

        mock_pl.data.activate_asset.assert_called_once()

    def test_raises_on_timeout(self) -> None:
        """Test that function raises TimeoutError on timeout."""
        client = MagicMock(spec=PlanetClient)
        mock_pl = MagicMock()
        client.get_sdk_client.return_value = mock_pl

        mock_pl.data.get_asset.return_value = {"status": "activating"}
        mock_pl.data.wait_asset.side_effect = Exception("Timeout")

        with pytest.raises(TimeoutError, match="Asset activation timed out"):
            wait_for_activation(client, "test_scene", timeout=1)


class TestDownloadAndClipPlanetScene:
    """Tests for download_and_clip_planet_scene function."""

    @patch("ftw_dataset_tools.api.imagery.planet_download.wait_for_activation")
    def test_returns_error_on_activation_timeout(
        self, mock_wait: MagicMock, tmp_path: Path
    ) -> None:
        """Test that function handles activation timeout gracefully."""
        client = MagicMock(spec=PlanetClient)
        mock_wait.side_effect = TimeoutError("Asset activation timed out")

        result = download_and_clip_planet_scene(
            client=client,
            item_id="test_scene",
            bbox=(0.0, 0.0, 1.0, 1.0),
            output_path=tmp_path / "output.tif",
        )

        assert result.success is False
        assert "timed out" in result.error

    @patch("ftw_dataset_tools.api.imagery.planet_download.wait_for_activation")
    def test_returns_error_on_missing_location(self, mock_wait: MagicMock, tmp_path: Path) -> None:
        """Test that function handles missing download location."""
        client = MagicMock(spec=PlanetClient)
        mock_wait.return_value = {"status": "active"}  # No location

        result = download_and_clip_planet_scene(
            client=client,
            item_id="test_scene",
            bbox=(0.0, 0.0, 1.0, 1.0),
            output_path=tmp_path / "output.tif",
        )

        assert result.success is False
        assert "No download location" in result.error


@pytest.mark.network
class TestPlanetDownloadNetwork:
    """Network-dependent tests for Planet download.

    These tests require a valid PL_API_KEY environment variable.
    Skip with: pytest -m "not network"
    """

    def test_activate_real_asset(self) -> None:
        """Test real asset activation."""
        import os

        api_key = os.environ.get("PL_API_KEY")
        if not api_key:
            pytest.skip("PL_API_KEY not set")

        # This test would need a valid scene ID to test with
        pytest.skip("Requires valid scene ID for testing")
