"""Nested, reproducible chip subsets ("scales") chosen by a stable hash of each 3x3 block."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from ftw_dataset_tools.api.blocks import block_scores, chip_block_ids, mgrs_squares
from ftw_dataset_tools.api.chips_io import write_chips
from ftw_dataset_tools.api.field_stats import CHIP_ID_COLUMN, DEFAULT_CHIP_KM_SIZE
from ftw_dataset_tools.api.geo import ensure_spatial_loaded, sql_path

if TYPE_CHECKING:
    import pandas as pd

SCALE_VERSION = "ftw-scale-v1"
SCORE_COLUMN = "scale_score"
DEFAULT_SCALE_PERCENT = 100.0
DEFAULT_MIN_BLOCKS_PER_SQUARE = 1


@dataclass
class ScaleResult:
    """Result of applying a scale to a chips file."""

    chips_file: Path
    percent: float
    total_chips: int
    kept_chips: int


def validate_scale(percent: float, min_blocks_per_square: int) -> None:
    """Raise ValueError unless percent is in [0, 100] and the floor is a count >= 0."""
    if isinstance(percent, bool) or not isinstance(percent, int | float) or not 0 <= percent <= 100:
        raise ValueError(f"scale percent must be between 0 and 100 (got {percent!r})")
    if (
        isinstance(min_blocks_per_square, bool)
        or not isinstance(min_blocks_per_square, int)
        or min_blocks_per_square < 0
    ):
        raise ValueError(
            f"min blocks per square must be a non-negative integer (got {min_blocks_per_square!r})"
        )


def apply_scale(
    chips_file: str | Path,
    percent: float = DEFAULT_SCALE_PERCENT,
    min_blocks_per_square: int = DEFAULT_MIN_BLOCKS_PER_SQUARE,
    km_size: float = DEFAULT_CHIP_KM_SIZE,
) -> ScaleResult:
    """Subset a chips file in place and record each kept chip's ``scale_score``.

    Keeps every 3x3 block scoring below ``percent / 100``, plus the
    ``min_blocks_per_square`` lowest-scoring blocks of each MGRS 100 km square. For a
    fixed floor, a smaller percent always selects a subset of a larger one.
    """
    validate_scale(percent, min_blocks_per_square)
    chips_path = Path(chips_file).resolve()
    if not chips_path.exists():
        raise FileNotFoundError(f"Chips file not found: {chips_path}")

    con = duckdb.connect(":memory:")
    ensure_spatial_loaded(con)
    try:
        con.execute(
            f"CREATE TABLE chips_table AS SELECT * FROM read_parquet('{sql_path(chips_path)}')"
        )
        columns = [row[0] for row in con.execute("DESCRIBE chips_table").fetchall()]
        if CHIP_ID_COLUMN not in columns:
            raise ValueError(f"Chips file must contain an '{CHIP_ID_COLUMN}' column: {chips_path}")
        ids = con.execute(f'SELECT "{CHIP_ID_COLUMN}" FROM chips_table').df()[CHIP_ID_COLUMN]
        if ids.empty:
            raise ValueError(f"Chips file is empty: {chips_path}")

        kept = _kept_chips(ids.astype(str), percent, min_blocks_per_square, km_size)
        con.register("kept", kept)
        exclude = f' EXCLUDE ("{SCORE_COLUMN}")' if SCORE_COLUMN in columns else ""
        write_chips(
            chips_path,
            con,
            f'SELECT c.*{exclude}, k."{SCORE_COLUMN}" FROM chips_table c '
            f'JOIN kept k ON CAST(c."{CHIP_ID_COLUMN}" AS VARCHAR) = k."{CHIP_ID_COLUMN}"',
        )
    finally:
        con.close()

    return ScaleResult(chips_path, percent, total_chips=len(ids), kept_chips=len(kept))


def _kept_chips(
    ids: pd.Series, percent: float, min_blocks_per_square: int, km_size: float
) -> pd.DataFrame:
    import pandas as pd

    blocks = chip_block_ids(ids, km_size)
    chips = pd.DataFrame(
        {
            CHIP_ID_COLUMN: ids,
            "block": blocks,
            "square": mgrs_squares(ids),
            SCORE_COLUMN: block_scores(blocks, SCALE_VERSION),
        }
    )
    per_block = chips.drop_duplicates("block")
    rank = per_block.groupby("square")[SCORE_COLUMN].rank(method="first")
    keep = (per_block[SCORE_COLUMN] < percent / 100) | (rank <= min_blocks_per_square)
    kept_blocks = per_block.loc[keep, "block"]
    return chips.loc[chips["block"].isin(kept_blocks), [CHIP_ID_COLUMN, SCORE_COLUMN]]
