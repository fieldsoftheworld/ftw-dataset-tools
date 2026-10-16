"""Lazy package-level re-exports (PEP 562)."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def lazy_exports(
    package: str, exports: dict[str, tuple[str, ...]]
) -> tuple[Callable[[str], Any], list[str]]:
    """Return a module ``__getattr__`` and ``__all__`` for ``{submodule: names}`` exports."""
    origins = {name: module for module, names in exports.items() for name in names}

    def __getattr__(name: str) -> Any:
        module = origins.get(name)
        if module is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        return getattr(importlib.import_module(f"{package}.{module}"), name)

    return __getattr__, sorted(origins)
