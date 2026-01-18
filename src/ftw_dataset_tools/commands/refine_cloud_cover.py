"""CLI command for refining cloud cover estimates using pixel-level analysis."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pystac
from tqdm import tqdm

from ftw_dataset_tools.api.imagery import calculate_pixel_cloud_cover


@click.command("refine-cloud-cover")
@click.argument("catalog_path", type=click.Path(exists=True))
@click.option(
    "--cloud-cover-pixel",
    type=float,
    default=0.0,
    show_default=True,
    help="Maximum acceptable pixel-level cloud cover percentage.",
)
@click.option(
    "--update-items",
    is_flag=True,
    default=False,
    help="Update STAC items with calculated eo:cloud_cover values.",
)
@click.option(
    "--cloud-threshold",
    type=int,
    default=30,
    show_default=True,
    help="SCL cloud class threshold (30=cloud shadow, higher catches more cloud types).",
)
@click.option(
    "--output-report",
    type=click.Path(),
    default=None,
    help="Path for JSON report of cloud cover analysis.",
)
def refine_cloud_cover_cmd(
    catalog_path: str,
    cloud_cover_pixel: float,
    update_items: bool,
    cloud_threshold: int,
    output_report: str | None,
) -> None:
    """Analyze pixel-level cloud cover for selected scenes.

    Reads the Scene Classification Layer (SCL) or cloud probability COGs
    for each selected scene and calculates actual cloud cover percentage
    within the chip's bounding box.

    \b
    CATALOG_PATH: Path to the STAC catalog directory (containing collection.json)

    \b
    Examples:
        ftwd refine-cloud-cover ./my-dataset-chips
        ftwd refine-cloud-cover ./chips --cloud-cover-pixel 5 --update-items
        ftwd refine-cloud-cover ./chips --output-report cloud-report.json
    """
    catalog_dir = Path(catalog_path)
    collection_file = catalog_dir / "collection.json"

    if not collection_file.exists():
        raise click.ClickException(f"No collection.json found in {catalog_path}")

    click.echo(f"Catalog: {catalog_path}")
    click.echo(f"Cloud cover threshold: {cloud_cover_pixel}%")
    if update_items:
        click.echo("Will update STAC items with calculated values")

    # Find all child S2 items
    child_items = []

    # Search in subdirectories for child items
    for subdir in catalog_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("."):
            for json_file in subdir.glob("*_s2.json"):
                try:
                    item = pystac.Item.from_file(str(json_file))
                    if item.id.endswith("_planting_s2") or item.id.endswith("_harvest_s2"):
                        child_items.append((item, json_file))
                except Exception:
                    pass

    if not child_items:
        raise click.ClickException(
            "No S2 child items found. Run 'select-images' first to create them."
        )

    click.echo(f"\nFound {len(child_items)} S2 items to analyze")

    # Track results
    results: list[dict] = []
    above_threshold: list[dict] = []
    failed: list[dict] = []

    # Process each child item
    with tqdm(total=len(child_items), desc="Analyzing cloud cover", unit="scene") as pbar:
        for item, item_path in child_items:
            bbox = tuple(item.bbox) if item.bbox else None

            if bbox is None:
                failed.append({"item": item.id, "error": "No bbox in item"})
                pbar.update(1)
                continue

            # Try to find cloud mask asset (SCL or cloud probability)
            cloud_href = None
            cloud_type = None

            if "scl" in item.assets:
                cloud_href = item.assets["scl"].href
                cloud_type = "scl"
            elif "cloud_probability" in item.assets:
                cloud_href = item.assets["cloud_probability"].href
                cloud_type = "probability"
            elif "cloud" in item.assets:
                cloud_href = item.assets["cloud"].href
                cloud_type = "probability"

            if not cloud_href:
                failed.append({"item": item.id, "error": "No cloud mask asset found"})
                pbar.update(1)
                continue

            try:
                cloud_pct = calculate_pixel_cloud_cover(
                    cloud_href=cloud_href,
                    bbox=bbox,
                    cloud_type=cloud_type,
                    cloud_threshold=cloud_threshold,
                )

                result = {
                    "item": item.id,
                    "cloud_cover_pixel": round(cloud_pct, 2),
                    "scene_cloud_cover": item.properties.get("eo:cloud_cover"),
                    "passes_threshold": cloud_pct <= cloud_cover_pixel,
                }
                results.append(result)

                if cloud_pct > cloud_cover_pixel:
                    above_threshold.append(result)

                # Update item if requested
                if update_items:
                    item.properties["eo:cloud_cover"] = round(cloud_pct, 2)
                    item.properties["ftw:cloud_cover_source"] = "pixel"
                    item.save_object(str(item_path))

            except Exception as e:
                failed.append({"item": item.id, "error": str(e)})

            pbar.update(1)

    # Print summary
    click.echo("\n" + "=" * 50)
    click.echo("Summary:")
    click.echo(f"  Analyzed: {len(results)}")
    click.echo(f"  Pass threshold ({cloud_cover_pixel}%): {len(results) - len(above_threshold)}")
    click.echo(f"  Above threshold: {len(above_threshold)}")
    click.echo(f"  Failed: {len(failed)}")

    if above_threshold:
        click.echo(click.style(f"\n{len(above_threshold)} items above threshold:", fg="yellow"))
        for r in above_threshold[:10]:
            click.echo(f"  - {r['item']}: {r['cloud_cover_pixel']}%")
        if len(above_threshold) > 10:
            click.echo(f"  ... and {len(above_threshold) - 10} more")

    if failed:
        click.echo(click.style(f"\n{len(failed)} items failed:", fg="red"))
        for f in failed[:5]:
            click.echo(f"  - {f['item']}: {f['error']}")
        if len(failed) > 5:
            click.echo(f"  ... and {len(failed) - 5} more")

    # Calculate statistics
    if results:
        cloud_values = [r["cloud_cover_pixel"] for r in results]
        click.echo("\nCloud cover statistics:")
        click.echo(f"  Min: {min(cloud_values):.1f}%")
        click.echo(f"  Max: {max(cloud_values):.1f}%")
        click.echo(f"  Mean: {np.mean(cloud_values):.1f}%")
        click.echo(f"  Median: {np.median(cloud_values):.1f}%")

    # Write report if requested
    if output_report:
        report = {
            "total_analyzed": len(results),
            "pass_threshold": len(results) - len(above_threshold),
            "above_threshold": len(above_threshold),
            "failed": len(failed),
            "threshold": cloud_cover_pixel,
            "results": results,
            "failed_items": failed,
        }
        report_path = Path(output_report)
        report_path.write_text(json.dumps(report, indent=2))
        click.echo(f"\nReport written to: {report_path}")

    if update_items:
        click.echo(
            click.style(f"\nUpdated {len(results)} STAC items with pixel cloud cover.", fg="green")
        )

    click.echo(click.style("\nDone!", fg="green"))


# Alias for registration
refine_cloud_cover = refine_cloud_cover_cmd
