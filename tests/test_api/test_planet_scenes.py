"""Offline pipeline tests with real raster crops and mocked Planet HTTP responses."""

import json
from datetime import date

import httpx
import numpy as np
import pytest
import rasterio
from click.testing import CliRunner
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from ftw_dataset_tools.api.planet import scene_workflow as workflow
from ftw_dataset_tools.api.planet.http import PlanetClient, PlanetError
from ftw_dataset_tools.api.planet.raster import write_raster
from ftw_dataset_tools.api.planet.scene_extract import extract_group, output_path
from ftw_dataset_tools.api.planet.scenes import ASSETS, asset_url, search_candidates, utc
from ftw_dataset_tools.api.planet.state import latest
from ftw_dataset_tools.api.planet.udm import quality
from ftw_dataset_tools.cli import cli
from tests.test_api.test_planet_inputs import write_fields


@pytest.fixture
def aoi():
    return {
        "id": "patch-a",
        "geometry": mapping(box(0.2, 0.2, 0.8, 0.8)),
        "start": "2024-01-01",
        "end": "2024-02-01",
        "target_date": "2024-01-15",
    }


@pytest.fixture
def rasters(tmp_path):
    profile = {"crs": "EPSG:4326", "transform": from_origin(0, 1, 0.1, 0.1)}
    sr = tmp_path / "source-sr.tif"
    udm = tmp_path / "source-udm.tif"
    write_raster(sr, np.full((4, 10, 10), 100, dtype="uint16"), profile, "source")
    data = np.zeros((8, 10, 10), dtype="uint8")
    data[0] = 1
    data[6] = 100
    write_raster(udm, data, profile, "source")
    return {ASSETS[0]: str(sr), ASSETS[1]: str(udm)}


def feature(identifier, geometry, acquired="2024-01-15T00:00:00Z"):
    return {
        "id": identifier,
        "geometry": geometry,
        "properties": {"acquired": acquired, "cloud_cover": 0.02},
    }


def test_search_paginates_filters_coverage_and_normalizes_time(aoi):
    calls = []

    def handle(request):
        calls.append(request)
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "features": [feature("partial", mapping(box(0, 0, 0.3, 0.3)))],
                    "_links": {"_next": "https://api.planet.com/page2"},
                },
            )
        return httpx.Response(
            200,
            json={
                "features": [
                    feature("later", aoi["geometry"], "2024-01-16T00:00:00Z"),
                    feature("best", aoi["geometry"]),
                ],
                "_links": {},
            },
        )

    with PlanetClient("secret", transport=httpx.MockTransport(handle)) as client:
        result = search_candidates(client, aoi, 0.1, 0.99, 10)
    assert [row["item_id"] for row in result] == ["best", "later"]
    assert len(calls) == 2
    assert utc("2024-01-01T01:00:00+01:00") == utc("2024-01-01")
    request = json.loads(calls[0].content)
    date_filter = request["filter"]["config"][1]
    assert date_filter["config"]["lt"] == "2024-02-01T00:00:00+00:00"


def test_activation_deadline_and_refresh(monkeypatch):
    calls = []
    monotonic = iter([0, 1, 2, 3])
    monkeypatch.setattr(
        "ftw_dataset_tools.api.planet.scenes.time.monotonic", lambda: next(monotonic)
    )
    monkeypatch.setattr("ftw_dataset_tools.api.planet.scenes.time.sleep", lambda _: None)

    def handle(request):
        calls.append(request.url.path)
        return httpx.Response(
            200,
            json={
                ASSETS[0]: {
                    "status": "activating",
                    "_links": {"activate": "https://api.planet.com/activate"},
                }
            },
        )

    with (
        PlanetClient("secret", transport=httpx.MockTransport(handle)) as client,
        pytest.raises(PlanetError, match="timed out"),
    ):
        asset_url(client, "scene", ASSETS[0], timeout=1)
    assert calls.count("/activate") == 1


def test_active_asset_refreshes_broken_url():
    calls = []

    def handle(request):
        calls.append(request.url.path)
        return httpx.Response(
            200,
            json={
                ASSETS[0]: {
                    "status": "active",
                    "location": "https://download.test/fresh",
                    "_links": {"activate": "https://api.planet.com/activate"},
                }
            },
        )

    with PlanetClient("secret", transport=httpx.MockTransport(handle)) as client:
        assert asset_url(client, "scene", ASSETS[0], refresh=True, poll_interval=0).endswith(
            "fresh"
        )
    assert calls.count("/activate") == 1


def test_cli_pipeline_and_resume(tmp_path, monkeypatch, aoi, rasters):
    calls = []

    def handle(request):
        calls.append(request)
        if request.url.path.endswith("quick-search"):
            return httpx.Response(
                200,
                json={
                    "features": [
                        feature("scene1", aoi["geometry"]),
                        feature("scene2", aoi["geometry"]),
                    ],
                    "_links": {},
                },
            )
        return httpx.Response(
            200,
            json={asset: {"status": "active", "location": path} for asset, path in rasters.items()},
        )

    monkeypatch.setattr(
        workflow,
        "PlanetClient",
        lambda: PlanetClient("secret", transport=httpx.MockTransport(handle)),
    )
    path = tmp_path / "fields.parquet"
    write_fields(
        path,
        [{"field_id": aoi["id"], "geometry": aoi["geometry"], "survey_date": date(2024, 1, 15)}],
        crs="EPSG:3857",
        geometry_column="boundary",
    )
    workdir = tmp_path / "run"
    runner = CliRunner()
    for arguments in (
        ["search", str(path), "--id-column", "field_id", "--date-column", "survey_date"],
        ["activate"],
        ["extract"],
        ["quality"],
    ):
        result = runner.invoke(cli, ["planet-scenes", *arguments, "--work-dir", str(workdir)])
        assert result.exit_code == 0, result.output
    before_search = len(calls)
    assert workflow.search(path, workdir, id_column="field_id", date_column="survey_date") == 0
    assert len(calls) == before_search
    assert latest(workdir / "quality.jsonl")[aoi["id"]]["usable"]
    output = output_path(workdir, aoi["id"], ASSETS[0])
    with rasterio.open(output) as source:
        assert source.count == 4
        assert np.all(source.read() == 100)
        assert source.bounds.left == pytest.approx(0.2)
        assert source.bounds.right == pytest.approx(0.8)
    before = len(calls)
    assert workflow.extract(workdir) == 0
    assert len(calls) == before

    timestamp = output.stat().st_mtime_ns
    output_path(workdir, aoi["id"], ASSETS[1]).unlink()
    assert workflow.extract(workdir) == 0
    assert output.stat().st_mtime_ns == timestamp
    assert len(calls) == before + 1
    workflow.select_alternate(workdir, aoi["id"], 1)
    assert workflow.report_quality(workdir) == 1
    assert workflow.extract(workdir) == 0
    with rasterio.open(output) as source:
        assert source.tags()["planet_item_id"] == "scene2"
    assert "secret" not in "".join(p.read_text() for p in workdir.rglob("*.jsonl"))


def test_scene_group_opens_once_per_asset_and_recovers_companion(
    tmp_path, monkeypatch, aoi, rasters
):
    calls = []

    def resolve(_client, _item, asset, **_kwargs):
        calls.append(asset)
        if asset == ASSETS[1]:
            raise PlanetError("temporarily unavailable")
        return rasters[asset]

    monkeypatch.setattr("ftw_dataset_tools.api.planet.scene_extract.asset_url", resolve)
    task = {"id": "scene", "members": [aoi, dict(aoi, id="patch-b")]}
    result = extract_group(None, task, tmp_path, 1)
    assert result["status"] == "failed"
    assert calls.count(ASSETS[0]) == 1
    assert calls.count(ASSETS[1]) == 2
    assert output_path(tmp_path, aoi["id"], ASSETS[0]).exists()


def test_udm_zero_class_values_blackfill_and_polygon_denominator(tmp_path):
    path = tmp_path / "udm.tif"
    data = np.zeros((8, 2, 2), dtype="uint8")
    data[0] = 1
    data[6] = 100
    data[7, 0, 0] = 1
    data[7, 1, 1] = 4
    write_raster(path, data, {"crs": "EPSG:4326", "transform": from_origin(0, 2, 1, 1)}, "test")
    result = quality(path, mapping(box(0, 0, 2, 2)))
    assert result["clear"] == 0.75
    assert result["blackfill"] == 0.25
    assert result["unusable"] == 0.5
    assert result["confidence"] == 100
    assert result["cloud"] == 0
    assert not result["usable"]
    assert quality(path, mapping(box(1, 1, 2, 2)))["usable"]


def test_scene_edge_padding_is_blackfill(tmp_path, monkeypatch, aoi, rasters):
    monkeypatch.setattr(
        "ftw_dataset_tools.api.planet.scene_extract.asset_url",
        lambda _c, _i, asset, **_kw: rasters[asset],
    )
    row = dict(aoi, geometry=mapping(box(-0.2, 0.2, 0.2, 0.8)))
    assert extract_group(None, {"id": "scene", "members": [row]}, tmp_path, 1)["status"] == "ok"
    result = quality(output_path(tmp_path, row["id"], ASSETS[1]), row["geometry"])
    assert result["blackfill"] == pytest.approx(0.5)
    assert result["clear"] == pytest.approx(0.5)


def test_changed_input_rejected(tmp_path, aoi):
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(aoi) + "\n")
    workflow.prepare(path, tmp_path / "run")
    path.write_text(json.dumps(dict(aoi, target_date="2024-01-20")) + "\n")
    with pytest.raises(ValueError, match="Settings changed"):
        workflow.prepare(path, tmp_path / "run")


def test_cli_rejects_invalid_shard(tmp_path, aoi, monkeypatch):
    monkeypatch.setenv("PL_API_KEY", "test")
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps(aoi) + "\n")
    result = CliRunner().invoke(
        cli,
        [
            "planet-scenes",
            "search",
            str(path),
            "--work-dir",
            str(tmp_path / "run"),
            "--shard-id",
            "2",
        ],
    )
    assert result.exit_code != 0
    assert "shard_id" in result.output
