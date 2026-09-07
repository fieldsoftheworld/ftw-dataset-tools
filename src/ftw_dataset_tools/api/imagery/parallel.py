"""Run independent per-chip imagery work on a thread pool.

Selecting and downloading imagery is almost entirely network wait: several STAC
searches per chip, then windowed reads over HTTP. The work for one chip touches
nothing another chip touches, so it parallelizes cleanly - but the counters, the
progress bar and any STAC item shared between a chip's seasons must stay on one
thread. This module draws that line: ``work`` runs on the pool, ``apply`` runs on
the caller's thread, one outcome at a time.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["DEFAULT_WORKERS", "ParallelOutcome", "run_in_parallel"]

# Enough threads to hide the network latency without hammering the STAC API.
DEFAULT_WORKERS = 4

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ParallelOutcome(Generic[T, R]):
    """One finished task: either a value or the exception the work raised."""

    task: T
    value: R | None = None
    error: Exception | None = None


def run_in_parallel(
    tasks: Sequence[T],
    work: Callable[[T], R],
    apply: Callable[[ParallelOutcome[T, R]], None],
    workers: int = DEFAULT_WORKERS,
) -> None:
    """Run ``work`` over ``tasks`` concurrently, applying each outcome as it completes.

    Args:
        tasks: Work items, submitted in the order given.
        work: Runs on a worker thread. It must touch only its own task's state;
            anything shared (counters, progress display, a common STAC item)
            belongs in ``apply``.
        apply: Runs on the calling thread, once per task, in completion order.
            Raising from it cancels the tasks that have not started and
            propagates the exception.
        workers: Maximum concurrent tasks. Values below 1 are treated as 1.
    """
    if not tasks:
        return

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(work, task): task for task in tasks}

        try:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    value = future.result()
                except Exception as err:  # surfaced to ``apply`` as an outcome
                    apply(ParallelOutcome(task=task, error=err))
                else:
                    apply(ParallelOutcome(task=task, value=value))
        except BaseException:
            # ``apply`` stopped the run: don't start the chips still queued.
            for pending in futures:
                pending.cancel()
            raise
