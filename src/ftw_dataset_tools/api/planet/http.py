"""Small Planet REST client with bounded retries and credential-safe errors."""

import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse

import httpx


class PlanetError(RuntimeError):
    """An API failure without request URLs or authentication material."""


class PlanetClient:
    """Thread-safe HTTP client; API credentials are sent only to api.planet.com."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        attempts: int = 6,
        timeout: float = 120,
    ) -> None:
        key = api_key or os.environ.get("PL_API_KEY") or os.environ.get("PLANET_API_KEY")
        if not key:
            raise ValueError("Set PL_API_KEY or PLANET_API_KEY before accessing Planet")
        if attempts < 1:
            raise ValueError("attempts must be positive")
        self.auth = httpx.BasicAuth(key, "")
        self.attempts = attempts
        self.client = httpx.Client(timeout=timeout, follow_redirects=True, transport=transport)

    def __enter__(self) -> "PlanetClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.client.close()

    def authentication(self, url: str) -> httpx.BasicAuth | None:
        parsed = urlparse(url)
        if parsed.scheme != "https":
            raise PlanetError("Planet endpoints must use HTTPS")
        return self.auth if parsed.netloc == "api.planet.com" else None

    def retry(self, operation: Callable) -> object:
        for attempt in range(self.attempts):
            delay = min(30, 0.5 * 2**attempt) + random.random() * 0.25
            try:
                return operation()
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                if status != 429 and status < 500:
                    raise PlanetError(f"Planet returned HTTP {status}") from None
                retry_after = error.response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    delay = min(120, max(delay, float(retry_after)))
            except httpx.TransportError:
                pass
            if attempt + 1 < self.attempts:
                time.sleep(delay)
        raise PlanetError(f"Planet request failed after {self.attempts} attempts")

    def request(self, method: str, url: str, **kwargs: object) -> dict:
        """Request JSON, retrying 429/5xx/transport failures only."""

        def operation() -> dict:
            response = self.client.request(method, url, auth=self.authentication(url), **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}

        return self.retry(operation)

    def download(self, url: str, destination: Path, validate: Callable[[Path], None]) -> None:
        """Stream, validate, then atomically replace; failed attempts preserve old files."""
        destination.parent.mkdir(parents=True, exist_ok=True)

        def operation() -> None:
            with NamedTemporaryFile(dir=destination.parent, suffix=".tif", delete=False) as stream:
                temporary = Path(stream.name)
            try:
                with self.client.stream("GET", url, auth=self.authentication(url)) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_bytes(1024 * 1024):
                            stream.write(chunk)
                validate(temporary)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)

        self.retry(operation)
