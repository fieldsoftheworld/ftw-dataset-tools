"""Plan, download and assemble Planet Basemaps with portable stage manifests."""

import json
from pathlib import Path

from ftw_dataset_tools.api.planet.basemap_raster import assemble_patch
from ftw_dataset_tools.api.planet.basemaps import download_quad, list_quads, resolve_mosaic
from ftw_dataset_tools.api.planet.http import PlanetClient
from ftw_dataset_tools.api.planet.state import (
    bind_config,
    collect,
    digest,
    in_shard,
    latest,
    manifest,
    run_tasks,
    safe_id,
)


def plan_key(identifier: str, mosaic_name: str) -> str:
    return digest([identifier, mosaic_name])


def plan(
    path: Path,
    workdir: Path,
    *,
    mosaics: tuple[str, ...] = (),
    workers: int = 4,
    shard_id: int = 0,
    num_shards: int = 1,
) -> int:
    """List quads for each requested AOI/period; retain only IDs, not signed URLs."""
    rows = manifest(path)
    for row in rows:
        row["mosaics"] = list(mosaics) or row.get("mosaics", [])
        if not row["mosaics"] or len(set(row["mosaics"])) != len(row["mosaics"]):
            raise ValueError(f"Specify distinct ordered mosaic names for {row['id']}")
        for name in row["mosaics"]:
            safe_id(name)
    bind_config(workdir / "inputs.json", {"manifest": rows})
    bind_config(workdir / "plan" / "config.json", {"num_shards": num_shards})
    journal = workdir / "plan" / f"shard-{shard_id:05d}.jsonl"
    done = latest(journal)
    tasks = [
        {
            "id": plan_key(row["id"], name),
            "aoi_id": row["id"],
            "geometry": row["geometry"],
            "mosaic_name": name,
        }
        for row in rows
        for name in row["mosaics"]
        if in_shard(plan_key(row["id"], name), shard_id, num_shards)
        and done.get(plan_key(row["id"], name), {}).get("status") != "ok"
    ]
    with PlanetClient() as client:
        resolved = {
            name: resolve_mosaic(client, name) for name in sorted({t["mosaic_name"] for t in tasks})
        }

        def work(task: dict) -> dict:
            mosaic_id = resolved[task["mosaic_name"]]["id"]
            quads = list_quads(client, mosaic_id, task["geometry"])
            return dict(
                task, mosaic_id=mosaic_id, quad_ids=quads, status="ok" if quads else "empty"
            )

        return run_tasks(tasks, work, journal, workers)


def download(workdir: Path, *, workers: int = 4, shard_id: int = 0, num_shards: int = 1) -> int:
    """Download each unique (mosaic, quad) once, across AOIs and array tasks."""
    # Ensure plan has completed for all AOIs before starting shared-cache jobs.
    assembly_tasks(workdir)
    targets = {}
    for row in collect(workdir / "plan").values():
        for quad_id in row.get("quad_ids", []):
            identifier = digest([row["mosaic_id"], quad_id])
            if in_shard(identifier, shard_id, num_shards):
                targets[identifier] = {
                    "id": identifier,
                    "mosaic_id": row["mosaic_id"],
                    "quad_id": quad_id,
                }
    with PlanetClient() as client:

        def work(task: dict) -> dict:
            return download_quad(client, workdir, task)

        return run_tasks(
            list(targets.values()),
            work,
            workdir / "download" / f"shard-{shard_id:05d}.jsonl",
            workers,
        )


def assembly_tasks(workdir: Path) -> list[dict]:
    rows = json.loads((workdir / "inputs.json").read_text())["manifest"]
    planned = collect(workdir / "plan")
    output = []
    for row in rows:
        plans = [planned.get(plan_key(row["id"], name)) for name in row["mosaics"]]
        if any(item is None or item.get("status") == "failed" for item in plans):
            raise ValueError("Plan is incomplete; finish or retry all plan shards first")
        output.append(dict(row, plans=plans))
    return output


def assemble(
    workdir: Path,
    *,
    bands: tuple[int, ...] = (1, 2, 3),
    workers: int = 4,
    shard_id: int = 0,
    num_shards: int = 1,
) -> int:
    """Assemble ordered bands/periods, preserving the native CRS/grid and validity mask."""
    if not bands or any(band < 1 for band in bands) or len(set(bands)) != len(bands):
        raise ValueError("Specify distinct positive band indices")
    tasks = [row for row in assembly_tasks(workdir) if in_shard(row["id"], shard_id, num_shards)]

    def work(task: dict) -> dict:
        return assemble_patch(workdir, task, bands)

    return run_tasks(tasks, work, workdir / "assemble" / f"shard-{shard_id:05d}.jsonl", workers)
