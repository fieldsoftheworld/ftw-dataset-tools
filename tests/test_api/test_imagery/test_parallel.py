"""Tests for the shared thread-pool runner used by the imagery workflows."""

from __future__ import annotations

import threading
import time

import pytest

from ftw_dataset_tools.api.imagery.parallel import ParallelOutcome, run_in_parallel


class TestRunInParallel:
    """Work runs on the pool; outcomes are applied on the calling thread."""

    def test_every_task_is_applied(self) -> None:
        applied: list[int] = []

        run_in_parallel(
            [1, 2, 3, 4, 5],
            work=lambda n: n * 2,
            apply=lambda outcome: applied.append(outcome.value),
            workers=3,
        )

        assert sorted(applied) == [2, 4, 6, 8, 10]

    def test_tasks_run_concurrently(self) -> None:
        lock = threading.Lock()
        state = {"active": 0, "max_active": 0}

        def work(_task: int) -> None:
            with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.05)
            with lock:
                state["active"] -= 1

        run_in_parallel(list(range(8)), work=work, apply=lambda _outcome: None, workers=4)

        assert state["max_active"] > 1

    def test_apply_runs_on_the_calling_thread(self) -> None:
        threads: list[str] = []

        run_in_parallel(
            list(range(6)),
            work=lambda n: n,
            apply=lambda _outcome: threads.append(threading.current_thread().name),
            workers=4,
        )

        assert set(threads) == {threading.main_thread().name}

    def test_work_errors_arrive_as_outcomes(self) -> None:
        outcomes: list[ParallelOutcome] = []

        def work(n: int) -> int:
            if n == 2:
                raise RuntimeError("boom")
            return n

        run_in_parallel(list(range(4)), work=work, apply=outcomes.append, workers=4)

        failed = [o for o in outcomes if o.error is not None]
        assert len(failed) == 1
        assert failed[0].task == 2
        assert str(failed[0].error) == "boom"

    def test_raising_from_apply_stops_the_run(self) -> None:
        started: list[int] = []
        lock = threading.Lock()

        def work(n: int) -> int:
            with lock:
                started.append(n)
            time.sleep(0.02)
            return n

        def apply(_outcome: ParallelOutcome) -> None:
            raise ValueError("stop")

        with pytest.raises(ValueError, match="stop"):
            run_in_parallel(list(range(50)), work=work, apply=apply, workers=2)

        assert len(started) < 50

    def test_empty_tasks_do_nothing(self) -> None:
        run_in_parallel([], work=lambda n: n, apply=lambda _o: pytest.fail("applied"), workers=4)
