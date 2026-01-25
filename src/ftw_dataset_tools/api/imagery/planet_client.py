"""Planet API client configuration and authentication.

This module provides authenticated access to Planet's STAC API and Data API
for imagery selection and download operations.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from planet import Planet
    from planet.auth import Auth
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

    Authentication supports:
    - Explicit API key (via parameter or PL_API_KEY env var)
    - OAuth via Planet CLI authentication (planet auth login)
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize Planet client.

        Args:
            api_key: Planet API key. If None, uses PL_API_KEY env var or
                    Planet CLI authentication (OAuth).

        Raises:
            ValueError: If no authentication method is available.
        """
        self._api_key = api_key or os.environ.get("PL_API_KEY")
        self._auth: Auth | None = None
        self._sdk_client: Planet | None = None
        self._stac_client: Client | None = None

        # If no explicit API key, try to get auth from Planet CLI
        if not self._api_key:
            self._auth = self._get_cli_auth()
            if not self._auth:
                raise ValueError(
                    "Planet authentication required. Either:\n"
                    "  - Pass api_key parameter\n"
                    "  - Set PL_API_KEY environment variable\n"
                    "  - Authenticate with: planet auth login"
                )

    @staticmethod
    def _get_cli_auth() -> Auth | None:
        """Try to get auth from Planet CLI.

        Returns:
            Auth object if available, None otherwise.
        """
        try:
            from planet.auth import Auth

            return Auth.from_user_default_session()
        except Exception:
            return None

    @property
    def api_key(self) -> str | None:
        """Return the API key if using API key auth, None for OAuth."""
        return self._api_key

    def _get_auth_header(self) -> dict[str, str]:
        """Get authorization header for HTTP requests.

        Returns:
            Dict with Authorization header.
        """
        if self._api_key:
            return {"Authorization": f"api-key {self._api_key}"}

        # For OAuth, get bearer token from auth flow
        import httpx

        request = httpx.Request("GET", "https://api.planet.com/")
        flow = self._auth.sync_auth_flow(request)
        authed_request = next(flow)
        return {"Authorization": authed_request.headers["authorization"]}

    def validate_auth(self) -> bool:
        """Validate authentication by making a simple API call.

        Returns:
            True if authentication is valid.

        Raises:
            Exception: If authentication fails.
        """
        import httpx

        headers = self._get_auth_header()
        response = httpx.get(
            PLANET_STAC_URL,
            headers=headers,
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
                headers=self._get_auth_header(),
            )
        return self._stac_client

    def get_sdk_client(self) -> Planet:
        """Get Planet SDK client for Data API operations.

        Returns:
            Configured planet.Planet client instance.
        """
        if self._sdk_client is None:
            from planet import Planet

            if self._api_key:
                self._sdk_client = Planet(api_key=self._api_key)
            else:
                # Let SDK auto-detect auth from CLI
                self._sdk_client = Planet()
        return self._sdk_client
