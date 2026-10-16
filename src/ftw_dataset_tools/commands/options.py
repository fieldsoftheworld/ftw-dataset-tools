"""Click parameter types shared by several commands."""

from __future__ import annotations

from typing import Any

import click

from ftw_dataset_tools.api.chip_grid import InvalidChipSizeError, chip_size_m


class KmSize(click.ParamType):
    """A chip edge length in kilometres that is a whole number of metres up to 100 km."""

    name = "km"

    def convert(
        self, value: Any, param: click.Parameter | None, ctx: click.Context | None
    ) -> float:
        try:
            km = float(value)
        except (TypeError, ValueError):
            self.fail(f"{value!r} is not a number of kilometres", param, ctx)
        try:
            chip_size_m(km)
        except InvalidChipSizeError as err:
            self.fail(str(err), param, ctx)
        return km


KM_SIZE = KmSize()

KM_SIZE_HELP = "Chip edge length in kilometres, e.g. 0.1 for 100 m chips."
