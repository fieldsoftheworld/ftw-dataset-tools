"""Reliability checks shared by the Planet download workflows."""

import httpx
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from ftw_dataset_tools.api.planet.http import PlanetClient, PlanetError
from ftw_dataset_tools.api.planet.raster import output_matches, validate_raster, write_raster
from ftw_dataset_tools.api.planet.state import bind_config, in_shard, latest, run_tasks


def test_journal_repairs_only_interrupted_tail(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text('{"id":"old","status":"ok"}\n{"id":')
    assert run_tasks([{"id": "new"}], lambda row: dict(row, status="ok"), path, 1) == 0
    assert set(latest(path)) == {"old", "new"}
    path.write_text('bad\n{"id":"new"}\n')
    with pytest.raises(ValueError, match="Invalid JSON"):
        latest(path)


def test_stable_shards_and_frozen_configuration(tmp_path):
    with pytest.raises(ValueError):
        in_shard("a", 4, 4)
    path = tmp_path / "config.json"
    bind_config(path, {"a": 1})
    bind_config(path, {"a": 1})
    with pytest.raises(ValueError, match="Settings changed"):
        bind_config(path, {"a": 2})


def test_retries_only_transient_errors_and_hides_urls(monkeypatch):
    monkeypatch.setattr("ftw_dataset_tools.api.planet.http.time.sleep", lambda _: None)
    statuses = iter([429, 503, 200, 403])
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(next(statuses), json={"ok": True})

    with PlanetClient("secret", transport=httpx.MockTransport(handle)) as client:
        assert client.request("GET", "https://api.planet.com/test") == {"ok": True}
        with pytest.raises(PlanetError, match="HTTP 403") as error:
            client.request("GET", "https://api.planet.com/test?api_key=secret")
        assert "secret" not in str(error.value)
    assert len(calls) == 4


def test_credentials_not_forwarded_to_download_host(tmp_path):
    seen = []

    def handle(request):
        seen.append(request.headers.get("authorization"))
        if request.url.host == "api.planet.com":
            return httpx.Response(302, headers={"Location": "https://download.test/file"})
        return httpx.Response(200, content=b"imagery")

    with PlanetClient("secret", transport=httpx.MockTransport(handle)) as client:
        path = tmp_path / "output"
        client.download("https://api.planet.com/quad", path, lambda _: None)
    assert seen[0].startswith("Basic ")
    assert seen[1] is None
    assert path.read_bytes() == b"imagery"


def test_invalid_download_preserves_existing_file(tmp_path):
    path = tmp_path / "quad.tif"
    path.write_bytes(b"previous")
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=b"not a tiff"))
    with (
        PlanetClient("secret", transport=transport) as client,
        pytest.raises(rasterio.errors.RasterioIOError),
    ):
        client.download("https://api.planet.com/quad", path, validate_raster)
    assert path.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [path]


def test_output_resume_checks_signature_and_pixels(tmp_path):
    path = tmp_path / "image.tif"
    profile = {"crs": "EPSG:4326", "transform": from_origin(0, 1, 0.1, 0.1)}
    write_raster(path, np.ones((1, 10, 10), dtype="uint8"), profile, "version1")
    assert output_matches(path, "version1")
    assert not output_matches(path, "version2")
    path.write_bytes(path.read_bytes()[:200])
    assert not output_matches(path, "version1")


def test_exception_text_never_enters_journal(tmp_path):
    def fail(_):
        raise RuntimeError("https://secret.test?token=private")

    path = tmp_path / "jobs.jsonl"
    assert run_tasks([{"id": "a"}], fail, path, 1) == 1
    assert "private" not in path.read_text()
    assert latest(path)["a"]["error"] == "RuntimeError"
