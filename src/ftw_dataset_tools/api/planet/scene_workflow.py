"""Resumable PlanetScope search, preactivation, extraction and quality stages."""

import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from ftw_dataset_tools.api.planet.http import PlanetClient
from ftw_dataset_tools.api.planet.inputs import read_aois
from ftw_dataset_tools.api.planet.scene_extract import extract_group, output_path, signature
from ftw_dataset_tools.api.planet.scenes import (
    ASSETS,
    asset_url,
    search_candidates,
    utc,
    validate_dates,
)
from ftw_dataset_tools.api.planet.state import (
    bind_config,
    collect,
    in_shard,
    latest,
    run_tasks,
    safe_id,
)
from ftw_dataset_tools.api.planet.udm import quality


def prepare(
    path: Path,
    workdir: Path,
    *,
    start: str | None = None,
    end: str | None = None,
    target_date: str | None = None,
    date_column: str = "determination_datetime",
    search_days: int = 30,
    **input_options: Any,
) -> list[dict]:
    """Normalize polygon dates and freeze the effective input for resumable runs."""
    if search_days < 1:
        raise ValueError("search_days must be positive")
    rows = read_aois(path, extra_columns=(date_column,), **input_options)
    for row in rows:
        target = target_date or row.get("target_date") or row.get(date_column)
        first, last = start or row.get("start"), end or row.get("end")
        if target is None:
            if not first or not last:
                raise ValueError(
                    f"Supply --target-date, {date_column}, or both --start/--end for {row['source_id']}"
                )
            target = (utc(first) + (utc(last) - utc(first)) / 2).isoformat()
        row["target_date"] = utc(target).isoformat()
        row["start"] = (
            utc(first).isoformat()
            if first
            else (utc(target) - timedelta(days=search_days)).isoformat()
        )
        row["end"] = (
            utc(last).isoformat()
            if last
            else (utc(target) + timedelta(days=search_days)).isoformat()
        )
        validate_dates(row)
    bind_config(workdir / "inputs.json", {"manifest": rows})
    return rows


def search(
    path: Path,
    workdir: Path,
    *,
    workers: int = 4,
    shard_id: int = 0,
    num_shards: int = 1,
    max_cloud: float = 0.1,
    min_coverage: float = 0.99,
    max_candidates: int = 10,
    **input_options: Any,
) -> int:
    """Persist ranked candidates. Failures and empty searches are retried on rerun."""
    if not 0 <= max_cloud <= 1 or not 0 < min_coverage <= 1 or max_candidates < 1:
        raise ValueError("Invalid search thresholds or candidate count")
    rows = prepare(path, workdir, **input_options)
    bind_config(
        workdir / "search" / "config.json",
        {
            "max_cloud": max_cloud,
            "min_coverage": min_coverage,
            "max_candidates": max_candidates,
            "num_shards": num_shards,
        },
    )
    journal = workdir / "search" / f"shard-{shard_id:05d}.jsonl"
    done = latest(journal)
    tasks = [
        row
        for row in rows
        if in_shard(row["id"], shard_id, num_shards)
        and done.get(row["id"], {}).get("status") != "ok"
    ]
    with PlanetClient() as client:

        def work(row: dict) -> dict:
            candidates = search_candidates(client, row, max_cloud, min_coverage, max_candidates)
            return {
                "id": row["id"],
                "status": "ok" if candidates else "empty",
                "candidates": candidates,
            }

        return run_tasks(tasks, work, journal, workers)


def selected(workdir: Path) -> list[dict]:
    """Join current candidate selections to the frozen input geometry."""
    rows = json.loads((workdir / "inputs.json").read_text())["manifest"]
    candidates = collect(workdir / "search")
    picks = latest(workdir / "selections.jsonl")
    output = []
    for row in rows:
        result = candidates.get(row["id"], {})
        if result.get("status") not in {"ok", "empty"}:
            raise ValueError("Search is incomplete; finish or retry all search shards first")
        choices = result.get("candidates", [])
        if result.get("status") != "ok" or not choices:
            continue
        rank = picks.get(row["id"], {}).get("rank", 0)
        output.append(dict(row, **choices[rank], rank=rank))
    return output


def groups(workdir: Path, shard_id: int, num_shards: int) -> list[dict]:
    grouped = defaultdict(list)
    for row in selected(workdir):
        if in_shard(row["item_id"], shard_id, num_shards):
            grouped[row["item_id"]].append(row)
    return [{"id": item_id, "members": members} for item_id, members in sorted(grouped.items())]


def activate(
    workdir: Path,
    *,
    workers: int = 4,
    shard_id: int = 0,
    num_shards: int = 1,
    activation_timeout: float = 3600,
) -> int:
    """Prewarm unique scene/assets; no signed URLs are persisted or trusted on rerun."""
    tasks = [
        {"id": f"{group['id']}--{asset}", "item_id": group["id"], "asset": asset}
        for group in groups(workdir, shard_id, num_shards)
        for asset in ASSETS
    ]
    with PlanetClient() as client:

        def work(task: dict) -> dict:
            asset_url(client, task["item_id"], task["asset"], timeout=activation_timeout)
            return {"id": task["id"], "status": "ok"}

        return run_tasks(tasks, work, workdir / "activate" / f"shard-{shard_id:05d}.jsonl", workers)


def extract(
    workdir: Path,
    *,
    workers: int = 4,
    shard_id: int = 0,
    num_shards: int = 1,
    activation_timeout: float = 3600,
) -> int:
    """Download selected SR/UDM2 crops; incomplete companions retry independently."""
    tasks = groups(workdir, shard_id, num_shards)
    with PlanetClient() as client:

        def work(task: dict) -> dict:
            return extract_group(client, task, workdir, activation_timeout)

        return run_tasks(tasks, work, workdir / "extract" / f"shard-{shard_id:05d}.jsonl", workers)


def report_quality(workdir: Path, *, workers: int = 4, min_clear: float = 0.95) -> int:
    """Score current selected companions, including missing/stale outputs as failures."""
    from ftw_dataset_tools.api.planet.raster import output_matches

    def work(row: dict) -> dict:
        path = output_path(workdir, row["id"], ASSETS[1])
        expected = signature(row, row["item_id"], ASSETS[1])
        if not output_matches(path, expected):
            raise ValueError("Current UDM2 output missing or invalid")
        return {
            "id": row["id"],
            "item_id": row["item_id"],
            "status": "ok",
            "signature": expected,
            **quality(path, row["geometry"], min_clear),
        }

    return run_tasks(selected(workdir), work, workdir / "quality.jsonl", workers)


def select_alternate(workdir: Path, identifier: str, rank: int) -> None:
    """Choose a cached alternate explicitly after inspecting patch quality."""
    safe_id(identifier)
    choices = collect(workdir / "search").get(identifier, {}).get("candidates", [])
    if not 0 <= rank < len(choices):
        raise ValueError(f"{identifier} has {len(choices)} candidates; rank is zero-based")
    run_tasks(
        [{"id": identifier}],
        lambda row: dict(row, rank=rank, status="ok"),
        workdir / "selections.jsonl",
        1,
    )
