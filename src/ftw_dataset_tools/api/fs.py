"""Filesystem helpers for writing a file through a temporary file.

Writers across ``api/`` build their output on a temporary file and rename it into
place, so a reader never sees a half-written file. The two helpers here keep that
pattern honest about permissions, and about the rename's one precondition: the
temporary file and its target have to live on the same filesystem.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def create_temp_file(target: Path, suffix: str = "") -> Path:
    """Create an empty temporary file beside ``target`` and return its path.

    Beside, because ``finalize_temp_file`` renames rather than copies. A temporary
    file in the system temp directory fails with ``EXDEV`` whenever that directory
    is a different mount - a tmpfs ``/tmp``, a container writing to a bind mount,
    an output dataset on an external or network disk - and even where it works it
    turns a metadata-only rename into a copy of every byte.

    The file is created with mode ``0o666`` so the kernel subtracts the process
    umask, exactly as it does for a plain ``open(path, "wb")``. ``tempfile``
    hardcodes ``0o600``, so a file rewritten through one of its temp files would
    silently become owner-only. Reading the umask to apply it by hand is not an
    option here: this tool runs worker pools (``api/imagery/parallel.py``) and the
    ``os.umask(0)``-then-restore idiom is process-wide and not thread-safe.

    Args:
        target: File the temporary file will eventually replace.
        suffix: Extension for the temporary name, for writers that dispatch on it.

    Returns:
        Path to a new empty file, a hidden sibling of ``target``.
    """
    while True:
        candidate = target.parent / f".{target.name}.{uuid.uuid4().hex}{suffix}"
        try:
            handle = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        except FileExistsError:  # pragma: no cover - a uuid4 collision
            continue
        os.close(handle)
        return candidate


def finalize_temp_file(tmp_path: Path, target: Path) -> None:
    """Rename ``tmp_path`` onto ``target``, keeping the permissions ``target`` had.

    A rename carries the *source's* mode across, so replacing a file through a
    temporary file would otherwise hand the target whatever mode the temporary file
    happened to have. When ``target`` already exists its mode is copied onto the
    temporary file first, which preserves a mode someone set deliberately. When it
    does not exist the temporary file keeps its own umask-derived mode, which is
    what writing ``target`` directly would have produced.

    ``tmp_path`` has to be on the same filesystem as ``target``: the rename is
    ``os.rename``, which fails with ``EXDEV`` across devices. ``create_temp_file``
    returns a path that satisfies this.
    """
    try:
        mode = target.stat().st_mode & 0o777
    except FileNotFoundError:
        pass
    else:
        tmp_path.chmod(mode)
    tmp_path.replace(target)
