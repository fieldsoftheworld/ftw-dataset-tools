"""Planet API client configuration and authentication.

This module provides authenticated access to Planet's STAC API and Data API
for imagery selection and download operations.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from planet import Planet
    from pystac_client import Client

__all__ = [
    "DEFAULT_BUFFER_DAYS",
    "DEFAULT_NUM_ITERATIONS",
    "PLANET_STAC_URL",
    "PLANET_TILES_URL",
    "VALID_BANDS",
    "PlanetClient",
]

# Planet API endpoints
PLANET_STAC_URL = "https://api.planet.com/x/data/"
PLANET_TILES_URL = "https://tiles.planet.com/data/v1/"

# Default selection parameters
DEFAULT_BUFFER_DAYS = 14
DEFAULT_NUM_ITERATIONS = 3

# Valid bands for PSScene (8-band SuperDove)
VALID_BANDS = [
    "blue",
    "green",
    "red",
    "nir",
    "coastal_blue",
    "green_i",
    "yellow",
    "red_edge",
]


class PlanetClient:
    """Authenticated client for Planet APIs.

    Provides access to:
    - Planet STAC API for scene discovery
    - Planet Data API for coverage estimation and asset operations
    - Planet Tiles API for thumbnail generation

    Authentication is via API key, either passed directly or from PL_API_KEY env var.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize Planet client with API key.

        Args:
            api_key: Planet API key. If None, reads from PL_API_KEY env var.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self._api_key = api_key or os.environ.get("PL_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Planet API key required. Pass api_key or set PL_API_KEY environment variable."
            )
        self._sdk_client: Planet | None = None
        self._stac_client: Client | None = None

    @property
    def api_key(self) -> str:
        """Return the API key (for authenticated requests)."""
        return self._api_key

    def validate_auth(self) -> bool:
        """Validate API key by making a simple API call.

        Returns:
            True if authentication is valid.

        Raises:
            Exception: If authentication fails.
        """
        import httpx

        # Simple validation: check if we can access the STAC API root
        response = httpx.get(
            PLANET_STAC_URL,
            auth=(self._api_key, ""),
            timeout=30.0,
        )
        response.raise_for_status()
        return True

    def get_stac_client(self) -> Client:
        """Get authenticated PySTAC client for Planet STAC API.

        Returns:
            Configured pystac_client.Client instance.
        """
        if self._stac_client is None:
            import pystac_client

            self._stac_client = pystac_client.Client.open(
                PLANET_STAC_URL,
                headers={"Authorization": f"api-key {self._api_key}"},
            )
        return self._stac_client

    def get_sdk_client(self) -> Planet:
        """Get Planet SDK client for Data API operations.

        Returns:
            Configured planet.Planet client instance.
        """
        if self._sdk_client is None:
            from planet import Planet

            self._sdk_client = Planet(api_key=self._api_key)
        return self._sdk_client
