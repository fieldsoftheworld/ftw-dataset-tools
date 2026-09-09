"""Tests for the temp-file write helpers in api/fs.py.

The point of these helpers is the two things a naive ``tempfile`` +
``Path.replace`` gets wrong: the mode the target ends up with, and the assumption
that the temp file is on the target's filesystem. Both are asserted here.
"""

from __future__ import annotations

import errno
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from ftw_dataset_tools.api.fs import create_temp_file, finalize_temp_file

if TYPE_CHECKING:
    import pytest


@contextmanager
def _umask(value: int):
    """Run the block under ``value``, restoring the process umask afterwards."""
    previous = os.umask(value)
    try:
        yield
    finally:
        os.umask(previous)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class TestCreateTempFile:
    """The temp file has to be an empty sibling of the target, not a /tmp file."""

    def test_creates_an_empty_file_beside_the_target(self, tmp_path: Path) -> None:
        target = tmp_path / "chips.parquet"

        temp = create_temp_file(target, suffix=".parquet")

        assert temp.parent == target.parent
        assert temp.exists()
        assert temp.read_bytes() == b""
        assert temp.name.endswith(".parquet")

    def test_temp_name_is_hidden_and_unique(self, tmp_path: Path) -> None:
        target = tmp_path / "chips.parquet"

        first = create_temp_file(target)
        second = create_temp_file(target)

        assert first != second
        assert first.name.startswith(".") and second.name.startswith(".")

    def test_mode_is_umask_derived_like_a_plain_write(self, tmp_path: Path) -> None:
        with _umask(0o077):
            temp = create_temp_file(tmp_path / "chips.parquet")
            reference = tmp_path / "reference.bin"
            reference.write_bytes(b"x")

        assert _mode(temp) == _mode(reference) == 0o600


class TestFinalizeTempFile:
    """The mode the target ends up with is the whole point of the helper."""

    def test_replaces_the_targets_content(self, tmp_path: Path) -> None:
        target = tmp_path / "chips.parquet"
        target.write_bytes(b"old")
        temp = create_temp_file(target)
        temp.write_bytes(b"new")

        finalize_temp_file(temp, target)

        assert target.read_bytes() == b"new"
        assert not temp.exists()

    def test_carries_an_existing_targets_mode_over(self, tmp_path: Path) -> None:
        target = tmp_path / "chips.parquet"
        target.write_bytes(b"old")
        target.chmod(0o640)
        temp = create_temp_file(target)

        finalize_temp_file(temp, target)

        assert _mode(target) == 0o640

    def test_a_new_target_gets_a_umask_derived_mode_not_0600(self, tmp_path: Path) -> None:
        """A restrictive umask must not be widened to the old hardcoded 0644."""
        target = tmp_path / "chips.parquet"
        with _umask(0o077):
            temp = create_temp_file(target)
            reference = tmp_path / "reference.bin"
            reference.write_bytes(b"x")
            finalize_temp_file(temp, target)

        assert _mode(target) == _mode(reference) == 0o600
        assert _mode(target) != 0o644

    def test_a_new_target_is_group_and_world_readable_under_a_lax_umask(
        self, tmp_path: Path
    ) -> None:
        """The published-dataset case: a 022 umask still yields a readable file."""
        target = tmp_path / "chips.parquet"
        with _umask(0o022):
            temp = create_temp_file(target)
            finalize_temp_file(temp, target)

        assert _mode(target) == 0o644

    def test_a_sibling_temp_file_never_crosses_devices(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guard the regression a single-filesystem CI cannot otherwise see.

        ``finalize_temp_file`` renames, and ``os.rename`` raises ``EXDEV`` when the
        source and target are on different mounts - which is exactly what a temp
        file in the system temp directory is whenever ``/tmp`` is its own
        filesystem. Standing in for a second mount, this rejects any rename whose
        source is not in the target's directory.
        """
        real_replace = Path.replace

        def strict_replace(self: Path, target: str | Path) -> Path:
            if self.parent != Path(target).parent:
                raise OSError(errno.EXDEV, "Cross-device link", str(self), None, str(target))
            return real_replace(self, target)

        monkeypatch.setattr(Path, "replace", strict_replace)

        target = tmp_path / "chips.parquet"
        temp = create_temp_file(target, suffix=".parquet")
        temp.write_bytes(b"new")

        finalize_temp_file(temp, target)

        assert target.read_bytes() == b"new"
