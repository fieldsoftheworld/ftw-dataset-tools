"""CLI command for quick inspection/summarization of a fields Parquet file."""

from __future__ import annotations

import json

import click

from ftw_dataset_tools.api.field_summary import (
    ColumnSummary,
    FieldSummary,
    GeometrySummary,
    summarize_fields,
)


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def _fmt_num(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:,.4g}"


@click.command("inspect-fields")
@click.argument("fields_file", type=click.Path(exists=True))
@click.option(
    "--top",
    type=int,
    default=20,
    show_default=True,
    help="Max values shown per categorical column (0 = all, capped for safety).",
)
@click.option(
    "-c",
    "--column",
    "columns",
    multiple=True,
    help="Show ALL value counts for this column (repeatable). Useful for class columns.",
)
@click.option(
    "--no-geometry",
    is_flag=True,
    default=False,
    help="Skip the geometry/CRS summary (avoids a full scan of the geometry column).",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON instead of text.")
def inspect_fields_cmd(
    fields_file: str,
    top: int,
    columns: tuple[str, ...],
    no_geometry: bool,
    as_json: bool,
) -> None:
    """Summarize a fields Parquet file: columns, value counts, and geometry.

    Prints row/column counts, per-column value counts (categorical), stats
    (numeric/temporal), and a geometry/CRS summary. Columns that look like good
    class-filter candidates are highlighted.

    \b
    FIELDS_FILE: Path to a (Geo)Parquet file with field boundaries.

    \b
    Examples:
        ftwd inspect-fields fields.parquet
        ftwd inspect-fields fields.parquet -c crop:name --top 0
        ftwd inspect-fields fields.parquet --no-geometry --json
    """
    summary = summarize_fields(
        fields_file,
        top=top,
        focus_columns=list(columns),
        include_geometry=not no_geometry,
    )

    if as_json:
        click.echo(json.dumps(summary.to_dict(), indent=2, default=str))
        return

    _print_text(summary)


def _print_text(summary: FieldSummary) -> None:
    click.echo(click.style(f"Fields file: {summary.path}", fg="cyan", bold=True))
    click.echo(f"  Size: {_human_size(summary.file_size_bytes)}")
    click.echo(f"  Rows: {summary.num_rows:,}")
    click.echo(f"  Columns: {summary.num_columns}")

    if summary.geometry is not None:
        _print_geometry(summary.geometry)

    click.echo("")
    click.echo(click.style("Columns:", fg="cyan", bold=True))
    for col in summary.columns:
        _print_column(col, summary.num_rows)

    if summary.class_filter_candidates:
        click.echo("")
        click.echo(click.style("Class-filter candidates:", fg="cyan", bold=True))
        click.echo(
            "  Low-cardinality categorical columns you could filter on "
            "(see 'ftwd run --class-filter'):"
        )
        by_name = {c.name: c for c in summary.columns}
        for name in summary.class_filter_candidates:
            click.echo(f"    - {name} ({by_name[name].distinct:,} distinct)")
        click.echo("  Tip: `ftwd inspect-fields <file> -c <column> --top 0` lists all its values.")


def _print_geometry(geom: GeometrySummary) -> None:
    click.echo("")
    click.echo(click.style("Geometry:", fg="cyan", bold=True))
    click.echo(f"  Column: {geom.column}")
    crs_bits = []
    if geom.crs_name:
        crs_bits.append(geom.crs_name)
    if geom.epsg:
        crs_bits.append(f"EPSG:{geom.epsg}")
    if geom.crs_kind:
        crs_bits.append(geom.crs_kind)
    click.echo(f"  CRS: {', '.join(crs_bits) if crs_bits else 'unknown'}")
    if geom.geometry_types:
        types = ", ".join(f"{gtype}: {count:,}" for gtype, count in geom.geometry_types)
        click.echo(f"  Geometry types: {types}")
    if geom.bounds:
        xmin, ymin, xmax, ymax = geom.bounds
        click.echo(f"  Bounds: [{xmin:.6g}, {ymin:.6g}, {xmax:.6g}, {ymax:.6g}]")


def _dtype_label(col: ColumnSummary) -> str:
    if col.kind == "geometry":
        return "GEOMETRY"
    if len(col.dtype) > 40:
        return col.dtype[:37] + "..."
    return col.dtype


def _print_column(col: ColumnSummary, num_rows: int) -> None:
    null_pct = (100 * col.nulls / num_rows) if num_rows else 0.0
    header = (
        f"  {col.name}  ({_dtype_label(col)}, {col.kind})  "
        f"distinct={col.distinct:,}  nulls={col.nulls:,} ({null_pct:.1f}%)"
    )
    click.echo(click.style(header, bold=True))

    if col.kind == "identifier":
        click.echo("      ~unique identifier (value counts omitted)")
    elif col.kind == "categorical" and col.value_counts is not None:
        _print_value_counts(col)
    elif col.kind == "numeric" and col.stats:
        s = col.stats
        click.echo(
            f"      min={_fmt_num(s['min'])}  max={_fmt_num(s['max'])}  "
            f"mean={_fmt_num(s['mean'])}  median={_fmt_num(s['median'])}"
        )
        click.echo(
            f"      p25={_fmt_num(s['p25'])}  p75={_fmt_num(s['p75'])}  "
            f"stddev={_fmt_num(s['stddev'])}"
        )
    elif col.kind == "temporal" and col.stats:
        click.echo(f"      range: {col.stats['min']}  →  {col.stats['max']}")


def _looks_like_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _print_value_counts(col: ColumnSummary) -> None:
    if not col.value_counts:
        click.echo("      (no non-null values)")
        return
    # Base the padding on non-URL values; URLs are printed in full (see below).
    width = max((len(v) for v, _ in col.value_counts if not _looks_like_url(v)), default=0)
    width = min(width, 50)
    for value, count in col.value_counts:
        pct = (100 * count / col.non_null) if col.non_null else 0.0
        # Never truncate URLs so they stay complete and clickable in the terminal.
        label = value if (_looks_like_url(value) or len(value) <= 50) else value[:47] + "..."
        click.echo(f"      {label:<{width}}  {count:>12,}  ({pct:5.1f}%)")
    if col.value_counts_truncated:
        remaining = col.distinct - len(col.value_counts)
        click.echo(
            click.style(
                f"      ... {remaining:,} more (use `--top 0` or `-c {col.name}` to see all)",
                fg="yellow",
            )
        )


# Alias for registration
inspect_fields = inspect_fields_cmd
