"""Validated AOIs, deterministic shards, and single-writer progress journals."""

import hashlib
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile

from ftw_dataset_tools.api.imagery.parallel import ParallelOutcome, run_in_parallel
from ftw_dataset_tools.api.logging_config import get_logger

logger = get_logger(__name__)


def safe_id(value: str) -> str:
    """Require a portable filename component, also safe in an API path."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(
            "IDs must start with a letter/digit and contain only letters, digits, ._-."
        )
    return value


def read_jsonl(path: Path, *, journal: bool = False) -> list[dict]:
    """Read objects; recover only an interrupted final journal line, never input errors."""
    if journal and not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            if journal and index == len(lines) - 1 and not line.endswith("\n"):
                logger.warning("Ignoring interrupted final record in %s", path)
                break
            raise ValueError(f"Invalid JSON at {path}:{index + 1}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at {path}:{index + 1}")
        rows.append(row)
    return rows


def digest(value: object) -> str:
    """Stable content identity for configuration and output provenance."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def in_shard(key: str, shard_id: int, num_shards: int) -> bool:
    """Assign the same key to the same task across processes and machines."""
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_id < num_shards")
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % num_shards == shard_id


def bind_config(path: Path, config: dict) -> None:
    """Freeze a run's inputs/settings, using atomic create across array tasks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(config, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if digest(json.loads(path.read_text())) != digest(config):
                raise ValueError(f"Settings changed for {path}; use a new work directory") from None
    finally:
        temporary.unlink(missing_ok=True)


def latest(path: Path) -> dict[str, dict]:
    """Last record wins, including failures after an earlier success."""
    return {row["id"]: row for row in read_jsonl(path, journal=True)}


def collect(directory: Path) -> dict[str, dict]:
    """Read disjoint array-task journals."""
    rows = {}
    for path in sorted(directory.glob("shard-*.jsonl")):
        rows.update(latest(path))
    return rows


def run_tasks(tasks: list[dict], work: Callable[[dict], dict], journal: Path, workers: int) -> int:
    """Run independent jobs, serialize status writes, and return the failure count.

    One process owns each journal. Never serialize exception strings: HTTP/GDAL
    errors can contain credentials or signed download URLs.
    """
    if not 1 <= workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    existing = read_jsonl(journal, journal=True)
    journal.parent.mkdir(parents=True, exist_ok=True)

    if journal.exists() and not journal.read_bytes().endswith(b"\n"):
        journal.write_text("".join(json.dumps(row) + "\n" for row in existing))
    failures = 0
    with journal.open("a", encoding="utf-8") as stream:

        def apply(outcome: ParallelOutcome) -> None:
            nonlocal failures
            if outcome.error is not None:
                row = {
                    "id": outcome.task["id"],
                    "status": "failed",
                    "error": type(outcome.error).__name__,
                }
            else:
                row = outcome.value
            if row["status"] not in {"ok", "empty"}:
                failures += 1
                logger.warning("%s: %s (%s)", row["id"], row["status"], row.get("error", ""))
            stream.write(json.dumps(row) + "\n")
            stream.flush()

        run_in_parallel(tasks, work, apply, workers)
    logger.info("Processed %d jobs; %d failed", len(tasks), failures)
    return failures
