"""Offline Basemaps pipeline tests with actual adjacent GeoTIFF quads."""

import json

import httpx
import numpy as np
import pytest
import rasterio
from click.testing import CliRunner
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from ftw_dataset_tools.api.planet import basemap_workflow as workflow
from ftw_dataset_tools.api.planet.basemap_raster import assemble_patch
from ftw_dataset_tools.api.planet.basemaps import (
    download_quad,
    list_quads,
    quad_path,
    resolve_mosaic,
)
from ftw_dataset_tools.api.planet.http import PlanetClient, PlanetError
from ftw_dataset_tools.api.planet.raster import write_raster
from ftw_dataset_tools.api.planet.state import collect
from ftw_dataset_tools.cli import cli
from tests.test_api.test_planet_inputs import write_fields


def make_quad(path, left=0, value=10, resolution=0.1):
    data = np.full((4, 10, 10), value, dtype="uint8")
    data[3] = 255
    data[:3, 0, 0] = 0
    write_raster(
        path,
        data,
        {"crs": "EPSG:4326", "transform": from_origin(left, 1, resolution, resolution)},
        "quad",
    )
    with rasterio.open(path, "r+") as target:
        target.colorinterp = (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        )
    return path.read_bytes()


def test_mosaic_exact_name_and_permissions():
    def handle(request):
        name = request.url.params["name__is"]
        return httpx.Response(
            200,
            json={"mosaics": [{"name": name, "id": "m1", "quad_download": name != "restricted"}]},
        )

    with PlanetClient("test", transport=httpx.MockTransport(handle)) as client:
        assert resolve_mosaic(client, "quarter-1")["id"] == "m1"
        with pytest.raises(PlanetError, match="unavailable"):
            resolve_mosaic(client, "restricted")


def test_quad_list_pagination_deduplicates_and_ignores_unavailable():
    def handle(request):
        if request.url.path.endswith("quads"):
            return httpx.Response(
                200,
                json={
                    "items": [
                        {"id": "q1", "_links": {"download": "https://download.test/q1"}},
                        {"id": "q-missing", "_links": {}},
                    ],
                    "_links": {"_next": "https://api.planet.com/page2"},
                },
            )
        return httpx.Response(
            200,
            json={
                "items": [
                    {"id": identifier, "_links": {"download": "https://download.test/quad"}}
                    for identifier in ("q1", "q2")
                ],
                "_links": {},
            },
        )

    with PlanetClient("test", transport=httpx.MockTransport(handle)) as client:
        assert list_quads(client, "m1", mapping(box(0, 0, 2, 1))) == ["q1", "q2"]


def test_cli_full_pipeline_shared_cache_stack_order_and_resume(tmp_path, monkeypatch):
    contents = {
        "left": make_quad(tmp_path / "left.tif"),
        "right": make_quad(tmp_path / "right.tif", left=1, value=20),
    }
    calls = []

    def handle(request):
        calls.append(str(request.url))
        path = request.url.path
        if path.endswith("/mosaics"):
            name = request.url.params["name__is"]
            return httpx.Response(
                200, json={"mosaics": [{"id": name, "name": name, "quad_download": True}]}
            )
        if path.endswith("/quads"):
            return httpx.Response(
                200,
                json={
                    "items": [
                        {"id": qid, "_links": {"download": "https://download.test/quad"}}
                        for qid in ("left", "right")
                    ],
                    "_links": {},
                },
            )
        return httpx.Response(200, content=contents[path.split("/")[-2]])

    monkeypatch.setattr(
        workflow,
        "PlanetClient",
        lambda: PlanetClient("test", transport=httpx.MockTransport(handle)),
    )
    geometry = mapping(box(0, 0, 2, 1))
    path = tmp_path / "fields.parquet"
    write_fields(path, [{"id": name, "geometry": geometry} for name in ("a", "b")], crs="EPSG:3857")
    workdir = tmp_path / "run"
    runner = CliRunner()
    args = [
        "planet-basemaps",
        "plan",
        str(path),
        "--work-dir",
        str(workdir),
        "--mosaic",
        "quarter-2",
        "--mosaic",
        "quarter-1",
    ]
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, result.output
    for stage in ("download", "assemble"):
        result = runner.invoke(cli, ["planet-basemaps", stage, "--work-dir", str(workdir)])
        assert result.exit_code == 0, result.output
    assert len([call for call in calls if call.endswith("/full")]) == 4
    output = workdir / "images" / "a.tif"
    with rasterio.open(output) as source:
        assert source.shape == (10, 20)
        assert source.count == 6
        assert source.descriptions[0] == "quarter-2:band_1"
        assert source.descriptions[3] == "quarter-1:band_1"
        assert source.read(1)[5, 5] == 10
        assert source.read(1)[5, 15] == 20
        assert source.read(1)[0, 0] == 0
        assert source.dataset_mask()[0, 0] == 255
    before = len(calls)
    timestamp = output.stat().st_mtime_ns
    assert workflow.download(workdir) == 0
    assert workflow.assemble(workdir) == 0
    assert len(calls) == before
    assert output.stat().st_mtime_ns == timestamp
    quad_path(workdir, "quarter-1", "left").write_bytes(b"broken")
    assert workflow.download(workdir) == 0
    assert len(calls) == before + 1
    assert "https://" not in "".join(p.read_text() for p in workdir.rglob("*.jsonl"))


def test_partial_period_is_reported_not_silently_filled(tmp_path, monkeypatch):
    def handle(request):
        if request.url.path.endswith("mosaics"):
            name = request.url.params["name__is"]
            return httpx.Response(
                200, json={"mosaics": [{"id": name, "name": name, "quad_download": True}]}
            )
        items = (
            []
            if "/q2/" in request.url.path
            else [{"id": "quad", "_links": {"download": "https://download.test/quad"}}]
        )
        return httpx.Response(200, json={"items": items, "_links": {}})

    monkeypatch.setattr(
        workflow,
        "PlanetClient",
        lambda: PlanetClient("test", transport=httpx.MockTransport(handle)),
    )
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps({"id": "a", "geometry": mapping(box(0, 0, 1, 1))}) + "\n")
    workdir = tmp_path / "run"
    assert workflow.plan(path, workdir, mosaics=("q1", "q2")) == 0
    assert {row["status"] for row in collect(workdir / "plan").values()} == {"ok", "empty"}
    assert workflow.assemble(workdir) == 1
    assert not (workdir / "images" / "a.tif").exists()


def test_missing_quad_never_creates_incomplete_stack(tmp_path):
    path = quad_path(tmp_path, "m1", "left")
    make_quad(path)
    task = {
        "id": "a",
        "geometry": mapping(box(0, 0, 2, 1)),
        "plans": [
            {
                "mosaic_id": "m1",
                "mosaic_name": "period",
                "quad_ids": ["left", "missing"],
                "status": "ok",
            }
        ],
    }
    with pytest.raises(rasterio.errors.RasterioIOError):
        assemble_patch(tmp_path, task, (1, 2, 3))
    assert not (tmp_path / "images" / "a.tif").exists()


def test_mismatched_grids_fail_and_preserve_old_output(tmp_path):
    make_quad(quad_path(tmp_path, "m1", "quad"))
    make_quad(quad_path(tmp_path, "m2", "quad"), resolution=0.2)
    task = {
        "id": "a",
        "geometry": mapping(box(0, 0, 1, 1)),
        "plans": [
            {"mosaic_id": mid, "mosaic_name": mid, "quad_ids": ["quad"], "status": "ok"}
            for mid in ("m1", "m2")
        ],
    }
    output = tmp_path / "images" / "a.tif"
    output.parent.mkdir()
    output.write_bytes(b"previous")
    with pytest.raises(ValueError, match="resolution"):
        assemble_patch(tmp_path, task, (1, 2, 3))
    assert output.read_bytes() == b"previous"


def test_transient_download_retries_and_cleans_temporary_file(tmp_path, monkeypatch):
    content = make_quad(tmp_path / "source.tif")
    statuses = iter([503, 200])
    monkeypatch.setattr("ftw_dataset_tools.api.planet.http.time.sleep", lambda _: None)
    transport = httpx.MockTransport(lambda _: httpx.Response(next(statuses), content=content))
    with PlanetClient("test", transport=transport) as client:
        result = download_quad(client, tmp_path, {"id": "x", "mosaic_id": "m", "quad_id": "q"})
    assert result["status"] == "ok"
    assert list((tmp_path / "quads" / "m").iterdir()) == [quad_path(tmp_path, "m", "q")]


def test_shards_cover_plan_once(tmp_path, monkeypatch):
    def handle(request):
        if request.url.path.endswith("mosaics"):
            return httpx.Response(
                200, json={"mosaics": [{"id": "m", "name": "period", "quad_download": True}]}
            )
        return httpx.Response(
            200, json={"items": [{"id": "q", "_links": {"download": "https://d.test/q"}}]}
        )

    monkeypatch.setattr(
        workflow,
        "PlanetClient",
        lambda: PlanetClient("test", transport=httpx.MockTransport(handle)),
    )
    path = tmp_path / "input.jsonl"
    path.write_text(
        "".join(
            json.dumps({"id": f"a{i}", "geometry": mapping(box(0, 0, 1, 1))}) + "\n"
            for i in range(9)
        )
    )
    workdir = tmp_path / "run"
    for shard in range(3):
        assert workflow.plan(path, workdir, mosaics=("period",), shard_id=shard, num_shards=3) == 0
    assert len(collect(workdir / "plan")) == 9
    assert len(workflow.assembly_tasks(workdir)) == 9
