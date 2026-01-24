# feat: STAC Imagery Workflow Improvements

**Created:** 2026-01-24
**Type:** Enhancement / Refactor
**Complexity:** High (12+ interrelated changes)
**Deepened on:** 2026-01-24

## Enhancement Summary

**Sections enhanced:** 12
**Research agents used:** kieran-python-reviewer, performance-oracle, security-sentinel, code-simplicity-reviewer, architecture-strategist, pattern-recognition-specialist, framework-docs-researcher, best-practices-researcher

### Key Improvements from Research
1. **Performance**: Use COG overviews for nodata detection (60-70% faster), cache STAC client with `@lru_cache`
2. **Security**: Add path sanitization for STAC item IDs, validate output paths stay within expected directories
3. **Architecture**: Move `_save_stac_items_safely` to API layer, create shared STAC utilities module
4. **Code Quality**: Use `click.FloatRange(0.0, 100.0)` for percentage options, make `VALID_BANDS` a tuple

### New Considerations Discovered
- Use `s2:nodata_pixel_percentage` from STAC metadata before reading pixels (faster)
- Rename `--no-data-max` to `--nodata-max` for consistency with rasterio/GDAL conventions
- Consider combining nodata + cloud check into single function for connection reuse
- Add GDAL environment variables for HTTP/2 multiplexing and connection pooling

---

## Overview

Comprehensive improvements to the STAC imagery workflow in ftwd based on PR feedback. Changes include simplifying cloud filtering to chip-level only, removing deprecated options and commands, adding nodata filtering, JPEG preview generation, and ensuring consistency across all subcommands.

## Problem Statement / Motivation

The current implementation has accumulated complexity:
1. **Redundant filtering**: Scene-level and pixel-level cloud filtering when pixel-level alone is sufficient
2. **Unused options**: `--stac-host` always uses earthsearch in practice
3. **Dead code**: `refine-cloud-cover` command is no longer needed (pixel check is fast enough by default)
4. **Missing functionality**: No nodata filtering, no JPEG previews, no output directory for select-images
5. **Inconsistency**: Different commands have different directory structures and option conventions
6. **Fragility**: No error handling for STAC save operations in download-images
7. **Datetime issues**: STAC items have same start/end datetime instead of full year range

## Proposed Solution

Implement 12 improvements in 4 phases, prioritizing breaking changes first, then renames, then new functionality, then structural consistency.

---

## Technical Approach

### Architecture Changes

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Commands                              │
├─────────────────────────────────────────────────────────────────┤
│  select-images      │  download-images   │  create-masks        │
│  - --cloud-cover-   │  - --bands         │  - STAC-aware        │
│    chip (was pixel) │    (click.Choice)  │    directory         │
│  - --nodata-max     │  - --preview/      │    structure         │
│  - -o, --output-dir │    --no-preview    │                      │
│  - Remove:          │  - Error handling  │                      │
│    --cloud-cover-   │  - None datetime   │                      │
│    scene            │    handling        │                      │
│    --pixel-check    │                    │                      │
│    --stac-host      │                    │                      │
├─────────────────────┴────────────────────┴──────────────────────┤
│                        API Layer                                 │
├─────────────────────────────────────────────────────────────────┤
│  scene_selection.py │  stac_items.py     │  masks.py            │
│  - Remove stac_host │  - Safe STAC save  │  - STAC dir paths    │
│  - Hardcode scene   │  - Child item      │                      │
│    filter to 75%    │    creation        │                      │
│  - Add nodata check │  - Catalog copy    │                      │
│                     │                    │                      │
│  nodata_analysis.py │  thumbnails.py     │                      │
│  - Use overviews    │  - JPEG generation │                      │
│  - Check metadata   │  - Percentile      │                      │
│    first            │    stretch         │                      │
├─────────────────────┴────────────────────┴──────────────────────┤
│                      settings.py                                 │
│  - DEFAULT_CLOUD_COVER_CHIP = 2 (was 0)                         │
│  - Remove MSPC config                                            │
│  - Add GDAL HTTP optimization env vars                           │
└─────────────────────────────────────────────────────────────────┘
```

### Research Insights: Architecture

**Recommendation from architecture-strategist:** Move `_save_stac_items_safely` to API layer. Create `api/stac_items.py` with:
- `save_child_item()` - Safe saving with cleanup on failure
- `create_scene_child_item()` - Child item creation (currently duplicated in commands)
- `update_parent_with_scenes()` - Parent item updates
- `copy_catalog()` - Deep copy for --output-dir

This eliminates duplication between `select_images.py` and `create_dataset.py`.

### Implementation Phases

---

## Phase 1: Breaking Changes (Do First)

### 1.1 Remove `--stac-host` option

**Files to modify:**
- `src/ftw_dataset_tools/commands/create_dataset.py:102-108` - Remove option
- `src/ftw_dataset_tools/commands/select_images.py` - Remove if present
- `src/ftw_dataset_tools/api/imagery/scene_selection.py:321` - Remove from function signature
- `src/ftw_dataset_tools/api/imagery/scene_selection.py:151-158` - Hardcode earthsearch URL
- `src/ftw_dataset_tools/api/imagery/settings.py:74-116` - Remove MSPC config

**Changes:**
```python
# settings.py - REMOVE these lines (74-116)
# MSPC_STAC_URL = ...
# MSPC config block

# scene_selection.py - Hardcode STAC URL
STAC_URL = "https://earth-search.aws.element84.com/v1"

def select_scenes_for_chip(
    chip_id: str,
    bbox: tuple[float, float, float, float],
    year: int,
    crop_calendar: CropCalendarDates,
    # REMOVE: stac_host: Literal["earthsearch", "mspc"] = "earthsearch",
    ...
) -> SceneSelectionResult:
```

### Research Insights: Performance

**Cache STAC client for connection reuse:**
```python
from functools import lru_cache

@lru_cache(maxsize=4)
def _get_stac_client(catalog_url: str) -> pystac_client.Client:
    """Get cached STAC client for a catalog URL."""
    return pystac_client.Client.open(catalog_url)
```
- **Impact**: Eliminates ~7 minutes overhead for 1000 chips (2000 client opens reduced to 1)

**Acceptance criteria:**
- [ ] `--stac-host` removed from all CLI commands
- [ ] `stac_host` parameter removed from all API functions
- [ ] MSPC configuration removed from settings.py
- [ ] `ftw:stac_host` still written to STAC items as `"earthsearch"` for documentation
- [ ] STAC client cached with `@lru_cache`

---

### 1.2 Remove `refine-cloud-cover` command

**Files to modify:**
- `src/ftw_dataset_tools/cli.py:44` - Remove registration
- `src/ftw_dataset_tools/commands/refine_cloud_cover.py` - Delete file

**Acceptance criteria:**
- [ ] Command removed from CLI
- [ ] File deleted
- [ ] Tests for this command removed/updated

---

### 1.3 Change default cloud cover to 2%

**Files to modify:**
- `src/ftw_dataset_tools/api/imagery/settings.py:61`

**Changes:**
```python
# settings.py
DEFAULT_CLOUD_COVER_CHIP = 2  # was 0, renamed from DEFAULT_CLOUD_COVER_PIXEL
```

**Acceptance criteria:**
- [ ] Default changed in settings.py
- [ ] Help text updated to show new default

---

### 1.4 Clarify docs/stac-extension.md

**Files to modify:**
- `docs/stac-extension.md:20-48`

**Changes:**
```markdown
| Property | Type | Description |
|----------|------|-------------|
| `ftw:cloud_cover_scene_threshold` | number | Scene-level cloud cover threshold used, percentage (0-100) |
| `ftw:cloud_cover_chip_threshold` | number | Chip-level threshold, percentage (0-100). Note: 2 means 2%, not 0.02 |
```

**Acceptance criteria:**
- [ ] Both threshold properties explicitly state "percentage (0-100)"
- [ ] Added note that 2 means 2%, not 0.02

---

## Phase 2: Renames and Option Changes

### 2.1 Simplify cloud filtering

**Files to modify:**
- `src/ftw_dataset_tools/commands/select_images.py:168-194`
- `src/ftw_dataset_tools/commands/create_dataset.py:109-149`
- `src/ftw_dataset_tools/api/imagery/scene_selection.py:317-586`
- `src/ftw_dataset_tools/api/imagery/settings.py`

**Changes:**

```python
# select_images.py - REMOVE these options:
# --cloud-cover-scene (lines 168-174)
# --pixel-check/--no-pixel-check (lines 182-188)

# RENAME this option with bounds validation:
@click.option(
    "--cloud-cover-chip",  # was --cloud-cover-pixel
    type=click.FloatRange(0.0, 100.0),  # Bounded validation
    default=DEFAULT_CLOUD_COVER_CHIP,  # 2
    show_default=True,
    help="Maximum cloud cover percentage (0-100) within chip bounds.",
)

# scene_selection.py - Hardcode scene-level filter internally
def _query_stac(...):
    # Hardcode scene-level to 75% for query efficiency
    # but don't expose to users
    query_cloud_cover = 75  # Internal constant
```

### Research Insights: Security

**Use `click.FloatRange` for validation** (from security-sentinel):
```python
@click.option(
    "--cloud-cover-chip",
    type=click.FloatRange(0.0, 100.0),  # Prevents negative or >100 values
    default=2.0,
)
```

**Acceptance criteria:**
- [ ] `--cloud-cover-pixel` renamed to `--cloud-cover-chip` in select-images
- [ ] `--cloud-cover-scene` removed from select-images
- [ ] `--pixel-check/--no-pixel-check` removed from select-images
- [ ] Same changes applied to create-dataset
- [ ] Scene-level filtering hardcoded to 75% internally in API
- [ ] STAC properties updated: `ftw:cloud_cover_chip_threshold` (was pixel)
- [ ] Settings constants renamed: `DEFAULT_CLOUD_COVER_CHIP`
- [ ] Use `click.FloatRange(0.0, 100.0)` for bounds validation

---

### 2.2 Change --bands to click.Choice

**Files to modify:**
- `src/ftw_dataset_tools/commands/download_images.py:21-26`

**Full Sentinel-2 band list (from EarthSearch):**
```python
# download_images.py - Use tuple for immutability
VALID_BANDS: tuple[str, ...] = (
    # Visible bands
    "coastal",      # B01 - Coastal aerosol (60m)
    "blue",         # B02 - Blue (10m)
    "green",        # B03 - Green (10m)
    "red",          # B04 - Red (10m)

    # Red edge bands
    "rededge1",     # B05 - Vegetation red edge 1 (20m)
    "rededge2",     # B06 - Vegetation red edge 2 (20m)
    "rededge3",     # B07 - Vegetation red edge 3 (20m)

    # NIR bands
    "nir",          # B08 - NIR (10m)
    "nir08",        # B8A - NIR narrow (20m)
    "nir09",        # B09 - Water vapour (60m)

    # SWIR bands
    "swir16",       # B11 - SWIR 1.6μm (20m)
    "swir22",       # B12 - SWIR 2.2μm (20m)

    # Atmospheric
    "aot",          # Aerosol Optical Thickness
    "wvp",          # Water Vapour

    # Classification/masks
    "scl",          # Scene Classification Layer
    "cloud",        # Cloud probability
    "snow",         # Snow probability

    # Composite
    "visual",       # True color RGB composite
)

@click.option(
    "--bands",
    type=click.Choice(VALID_BANDS, case_sensitive=False),
    multiple=True,
    default=("red", "green", "blue", "nir"),  # Tuple for default
    show_default=True,
    help="Sentinel-2 bands to download.",
)
def download_images(..., bands: tuple[str, ...], ...):
    band_list = list(bands)
```

### Research Insights: Code Quality

**Use tuple instead of list for constants** (from kieran-python-reviewer):
- Module-level constants should be tuples to signal immutability
- Prevents accidental modification

**Acceptance criteria:**
- [ ] `--bands` uses `click.Choice` with all 18 valid band names
- [ ] `multiple=True` allows specifying multiple bands
- [ ] Help text shows available choices
- [ ] Invalid band names rejected with clear error
- [ ] Default remains `("red", "green", "blue", "nir")` as tuple
- [ ] `VALID_BANDS` is a tuple, not a list

---

## Phase 3: New Functionality

### 3.1 Handle None datetime in SelectedScene

**Files to modify:**
- `src/ftw_dataset_tools/api/imagery/scene_selection.py:302-312`
- `src/ftw_dataset_tools/commands/download_images.py:183-191`

**Changes:**
```python
# scene_selection.py - Improved with timezone handling
from datetime import datetime, timezone


def _parse_iso_datetime(value: str) -> datetime:
    """Parse ISO datetime string to timezone-aware datetime."""
    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _get_item_datetime(item: pystac.Item) -> datetime:
    """
    Get datetime from STAC item, falling back to start_datetime if needed.

    Always returns a timezone-aware datetime.
    """
    if item.datetime is not None:
        dt = item.datetime
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    start_dt = item.properties.get("start_datetime")
    if start_dt is None:
        raise ValueError(f"STAC item {item.id} has no datetime or start_datetime")

    if isinstance(start_dt, str):
        return _parse_iso_datetime(start_dt)
    if isinstance(start_dt, datetime):
        if start_dt.tzinfo is None:
            return start_dt.replace(tzinfo=timezone.utc)
        return start_dt

    raise ValueError(
        f"STAC item {item.id} has invalid start_datetime type: {type(start_dt)}"
    )
```

### Research Insights: Code Quality

**Ensure timezone consistency** (from kieran-python-reviewer):
- Always return timezone-aware datetimes
- Extract ISO parsing to reusable `_parse_iso_datetime` function
- Add type checking for edge cases

**Acceptance criteria:**
- [ ] `SelectedScene.datetime` is never None
- [ ] Falls back to `start_datetime` when `item.datetime` is None
- [ ] Raises clear error if neither is available
- [ ] Always returns timezone-aware datetime
- [ ] Works in both scene_selection.py and download_images.py

---

### 3.2 Fix STAC item datetime range to span full year

**Files to modify:**
- `src/ftw_dataset_tools/api/stac.py` - Where chip STAC items are created

**Problem:** Currently chip STAC items have `start_datetime` and `end_datetime` set to the same value.

**Changes:**
```python
from datetime import datetime, timezone

def create_chip_stac_item(
    chip_id: str,
    geometry: dict[str, Any],  # More specific type hint
    bbox: tuple[float, float, float, float],
    year: int,
    properties: dict[str, Any],
) -> pystac.Item:
    """Create a STAC item for a chip with full year datetime range."""

    # Set datetime range to full calendar year
    start_datetime = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end_datetime = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    item = pystac.Item(
        id=chip_id,
        geometry=geometry,
        bbox=bbox,
        datetime=None,  # Use start/end instead
        properties={
            **properties,
            "start_datetime": start_datetime.isoformat(),
            "end_datetime": end_datetime.isoformat(),
        },
    )

    return item
```

### Research Insights: STAC Best Practices

**From PySTAC documentation:**
- `datetime=None` is **required** when using `start_datetime`/`end_datetime`
- Always include timezone information (`tzinfo=timezone.utc`)
- Use ISO format for datetime strings in properties

**Acceptance criteria:**
- [ ] Chip STAC items have `start_datetime` = Jan 1 of year at 00:00:00 UTC
- [ ] Chip STAC items have `end_datetime` = Dec 31 of year at 23:59:59 UTC
- [ ] `datetime` is None (using range instead)
- [ ] Existing items updated when re-processed

---

### 3.3 Add exception handling in download_images.py

**Files to modify:**
- `src/ftw_dataset_tools/commands/download_images.py:220-233`
- **NEW:** `src/ftw_dataset_tools/api/stac_items.py` (move logic to API layer)

### Research Insights: Architecture

**Move to API layer** (from architecture-strategist):
Create `api/stac_items.py` instead of keeping in commands:

```python
# api/stac_items.py
"""STAC item manipulation utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pystac


class STACSaveError(Exception):
    """Error saving STAC item."""


@dataclass
class STACSaveContext:
    """Context for saving STAC items after download."""
    item: pystac.Item
    item_dir: Path
    season: Literal["planting", "harvest"]
    band_list: list[str]
    output_filename: str


def save_child_item(ctx: STACSaveContext) -> None:
    """
    Save a child STAC item with cleanup on failure.

    Deletes the downloaded TIF if the JSON save fails.
    """
    tif_path = ctx.item_dir / ctx.output_filename
    json_path = ctx.item_dir / f"{ctx.item.id}.json"

    try:
        ctx.item.save_object(str(json_path))
    except Exception as e:
        # Cleanup: delete the downloaded TIF
        if tif_path.exists():
            tif_path.unlink()
        raise STACSaveError(
            f"Failed to save STAC item {ctx.item.id} at {ctx.item_dir}: {e}"
        ) from e


def update_parent_item(
    parent_item: pystac.Item,
    parent_path: Path,
    season: Literal["planting", "harvest"],
    output_filename: str,
    band_list: list[str],
) -> None:
    """
    Update parent item with reference to downloaded image.

    Rolls back the in-memory asset if save fails.
    """
    asset_key = f"{season}_image"

    try:
        parent_item.assets[asset_key] = pystac.Asset(
            href=f"./{output_filename}",
            media_type="image/tiff; application=geotiff; profile=cloud-optimized",
            title=f"{season.capitalize()} season imagery ({','.join(band_list)})",
            roles=["data"],
        )
        parent_item.save_object(str(parent_path))
    except Exception as e:
        parent_item.assets.pop(asset_key, None)
        raise STACSaveError(
            f"Failed to update parent item at {parent_path}: {e}"
        ) from e
```

### Research Insights: Security

**Add path validation** (from security-sentinel):
```python
def validate_output_path(output_path: Path, expected_parent: Path) -> Path:
    """Ensure output path is within expected directory."""
    resolved = output_path.resolve()
    expected = expected_parent.resolve()
    if not str(resolved).startswith(str(expected) + os.sep):
        raise ValueError(f"Output path {resolved} escapes expected directory {expected}")
    return resolved
```

**Acceptance criteria:**
- [ ] Logic moved to `api/stac_items.py`
- [ ] Both save operations wrapped in try/except
- [ ] Downloaded TIF deleted if child item save fails
- [ ] Asset removed from parent if parent save fails
- [ ] Custom `STACSaveError` exception for clarity
- [ ] Output paths validated to prevent path traversal

---

### 3.4 Add nodata filtering

**Files to modify:**
- `src/ftw_dataset_tools/commands/select_images.py` - Add option
- `src/ftw_dataset_tools/commands/create_dataset.py` - Add option
- `src/ftw_dataset_tools/api/imagery/scene_selection.py` - Add filtering logic
- `src/ftw_dataset_tools/api/imagery/nodata_analysis.py` - New file

### Research Insights: Performance

**Use STAC metadata first** (from best-practices-researcher):
Sentinel-2 items have `s2:nodata_pixel_percentage` in metadata. Check this before reading pixels:

```python
def get_nodata_from_metadata(item: pystac.Item) -> float | None:
    """Get nodata percentage from STAC metadata if available."""
    return item.properties.get("s2:nodata_pixel_percentage")
```

**Use COG overviews** (from performance-oracle):
Reading at overview level is 60-70% faster than decimated full-resolution reads:

```python
# api/imagery/nodata_analysis.py
"""Nodata/blackfill detection for Sentinel-2 imagery."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

SENTINEL2_NODATA = 0


def get_nodata_from_metadata(item) -> float | None:
    """Get nodata percentage from STAC metadata if available."""
    return item.properties.get("s2:nodata_pixel_percentage")


def calculate_nodata_percentage(
    cog_url: str,
    bbox: tuple[float, float, float, float],
) -> float:
    """
    Calculate percentage of nodata pixels in a COG within bbox.

    Uses COG overviews for efficient reading.

    Args:
        cog_url: URL to COG file
        bbox: Bounding box (minx, miny, maxx, maxy) in EPSG:4326

    Returns:
        Percentage of nodata pixels (0-100)
    """
    # Use overview level 2 for efficient reading (16x less data)
    with rasterio.open(cog_url, overview_level=2) as src:
        # Transform bbox to source CRS if needed
        if src.crs != CRS.from_epsg(4326):
            src_bbox = transform_bounds(CRS.from_epsg(4326), src.crs, *bbox)
        else:
            src_bbox = bbox

        window = from_bounds(*src_bbox, src.transform)

        # Clamp window to dataset bounds
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

        if window.width <= 0 or window.height <= 0:
            return 100.0  # No valid data in bbox

        data = src.read(1, window=window)

        nodata = src.nodata if src.nodata is not None else SENTINEL2_NODATA
        nodata_count = np.sum(data == nodata)
        total_pixels = data.size

        if total_pixels == 0:
            return 100.0

        return (nodata_count / total_pixels) * 100
```

**Add GDAL HTTP optimization** (from performance-oracle):
```python
# settings.py - Add at module level
import os

# GDAL HTTP optimization for remote COG access
os.environ.update({
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_TIMEOUT": "30",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "50000000",  # 50MB cache
})
```

**Rename option** (from pattern-recognition-specialist):
Use `--nodata-max` instead of `--no-data-max` for consistency with rasterio/GDAL conventions.

**CLI option:**
```python
@click.option(
    "--nodata-max",  # Consistent with rasterio/GDAL naming
    type=click.FloatRange(0.0, 100.0),
    default=0,
    show_default=True,
    help="Maximum nodata percentage (0-100) within chip bounds. Default 0 rejects any nodata.",
)
```

**Acceptance criteria:**
- [ ] `--nodata-max` option added to select-images and create-dataset (note: renamed from `--no-data-max`)
- [ ] Default is 0 (reject any nodata)
- [ ] Check STAC metadata `s2:nodata_pixel_percentage` first before reading pixels
- [ ] Use COG overviews (`overview_level=2`) for efficient reading
- [ ] Nodata check happens before cloud check (fail fast)
- [ ] Add GDAL HTTP optimization environment variables
- [ ] Tests added for nodata detection

---

### 3.5 Add JPEG preview generation

**Files to modify:**
- `src/ftw_dataset_tools/commands/download_images.py` - Add option and logic
- `src/ftw_dataset_tools/api/imagery/thumbnails.py` - New file

### Research Insights: STAC Best Practices

**Thumbnail specifications** (from best-practices-researcher):
- Use JPEG format for photographic/satellite imagery (smaller file size)
- Use `roles=["thumbnail"]` per STAC spec
- Thumbnails should be < 600px, typically 256-512px
- Quality 85 is good balance of size and quality

**New file: `thumbnails.py`**
```python
"""JPEG thumbnail generation for satellite imagery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError


class ThumbnailError(Exception):
    """Error generating thumbnail."""


def generate_thumbnail(
    tif_path: str | Path,
    output_path: str | Path,
    max_size: int = 512,
    quality: int = 85,
) -> Path:
    """
    Generate JPEG thumbnail from a multi-band GeoTIFF.

    Uses rasterio's out_shape to leverage COG overviews automatically.

    Args:
        tif_path: Path to GeoTIFF (must have at least 3 bands for RGB)
        output_path: Output path for JPEG
        max_size: Maximum dimension in pixels
        quality: JPEG quality (1-100)

    Returns:
        Path to generated thumbnail

    Raises:
        ThumbnailError: If thumbnail generation fails
    """
    tif_path = Path(tif_path)
    output_path = Path(output_path)

    if not tif_path.exists():
        raise ThumbnailError(f"Input file does not exist: {tif_path}")

    try:
        with rasterio.open(tif_path) as src:
            if src.count < 3:
                raise ThumbnailError(
                    f"Need at least 3 bands for RGB thumbnail, got {src.count}"
                )

            # Calculate output dimensions maintaining aspect ratio
            scale = min(max_size / src.width, max_size / src.height)
            out_width = max(1, int(src.width * scale))
            out_height = max(1, int(src.height * scale))

            # Read RGB bands at thumbnail size (uses overviews automatically)
            data = src.read(
                indexes=[1, 2, 3],
                out_shape=(3, out_height, out_width),
                resampling=Resampling.bilinear,
                masked=True,
            )

            # Handle nodata
            if np.ma.is_masked(data):
                data = data.filled(fill_value=0)

        # Normalize for display (percentile stretch)
        data = _normalize_for_display(data)

        # Convert to PIL image
        rgb_array = np.transpose(data, (1, 2, 0)).astype(np.uint8)
        img = Image.fromarray(rgb_array, mode="RGB")
        img.save(output_path, "JPEG", quality=quality, optimize=True)

    except RasterioIOError as e:
        raise ThumbnailError(f"Failed to read {tif_path}: {e}") from e
    except OSError as e:
        # Clean up partial output
        if output_path.exists():
            output_path.unlink()
        raise ThumbnailError(f"Failed to write thumbnail: {e}") from e

    return output_path


def _normalize_for_display(
    data: np.ndarray,
    percentile_clip: tuple[float, float] = (2, 98),
) -> np.ndarray:
    """
    Normalize array to 0-255 using per-band percentile stretching.

    Processing each band separately because percentiles are computed
    independently for proper color balance.
    """
    result = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[0]):
        band = data[i].astype(np.float32)
        valid = band[band > 0]
        if len(valid) > 0:
            p_low, p_high = np.percentile(valid, percentile_clip)
            if p_high > p_low:
                band = np.clip((band - p_low) / (p_high - p_low) * 255, 0, 255)
        result[i] = band
    return result.astype(np.uint8)
```

**CLI option:**
```python
@click.option(
    "--preview/--no-preview",
    default=True,
    show_default=True,
    help="Generate JPEG preview thumbnails for downloaded images.",
)
```

**STAC asset with proper roles:**
```python
# Add thumbnail asset to STAC item per STAC best practices
if preview and has_rgb_bands(band_list):
    thumbnail_path = generate_thumbnail(tif_path, tif_path.with_suffix(".jpg"))
    item.assets["thumbnail"] = pystac.Asset(
        href=f"./{thumbnail_path.name}",
        media_type=pystac.MediaType.JPEG,
        title="JPEG preview",
        roles=["thumbnail"],  # Standard STAC role
    )
```

**Acceptance criteria:**
- [ ] `--preview/--no-preview` option added (default True)
- [ ] JPEG generated alongside TIF when RGB bands present
- [ ] Thumbnail added to STAC item with `roles=["thumbnail"]`
- [ ] Use `pystac.MediaType.JPEG` for proper media type
- [ ] Percentile stretch applied for good visualization
- [ ] Max size 512px
- [ ] Quality 85
- [ ] Skipped gracefully if RGB bands not downloaded
- [ ] Custom `ThumbnailError` exception
- [ ] Cleanup partial output on failure

---

## Phase 4: Structural Changes

### 4.1 Update create-masks directory structure

**Files to modify:**
- `src/ftw_dataset_tools/commands/create_masks.py`
- `src/ftw_dataset_tools/api/masks.py:70-112`

**Changes:**
```python
# masks.py - Update get_mask_output_path to match create-dataset structure

def get_mask_output_path(
    output_dir: Path,
    grid_id: str,
    mask_type: MaskType,
    year: int | None = None,
    stac_mode: bool = False,
) -> Path:
    """
    Get output path for mask file.

    When stac_mode=True, uses create-dataset directory structure:
        output_dir/{grid_id}_{year}/{grid_id}_{year}_{mask_type}.tif

    When stac_mode=False (or year is None), uses flat structure:
        output_dir/{grid_id}_{year}_{mask_type}.tif

    Note: If stac_mode=True but year=None, falls back to flat structure.
    """
    filename = get_mask_filename(grid_id, mask_type, year)

    if stac_mode and year is not None:
        item_dir = output_dir / f"{grid_id}_{year}"
        item_dir.mkdir(parents=True, exist_ok=True)
        return item_dir / filename

    return output_dir / filename
```

### Research Insights: Simplicity

**Consider removing stac_mode** (from code-simplicity-reviewer):
The existing code already handles both patterns. This is appropriate complexity for backward compatibility.

**Acceptance criteria:**
- [ ] create-masks supports STAC-aware directory structure
- [ ] Option to use legacy flat structure for backward compatibility
- [ ] Mask files placed in item subdirectories when using STAC mode
- [ ] Year parameter properly handled
- [ ] Document edge case: `stac_mode=True` with `year=None` falls back to flat structure

---

### 4.2 Add --output-dir to select-images

**Files to modify:**
- `src/ftw_dataset_tools/commands/select_images.py`
- **NEW:** `src/ftw_dataset_tools/api/stac_items.py` (add `copy_catalog` function)

### Research Insights: Security

**Handle symlinks safely** (from security-sentinel):
```python
# api/stac_items.py
import shutil

def copy_catalog(src: Path, dst: Path) -> None:
    """
    Copy catalog directory safely, not following symlinks.

    Copies symlinks as symlinks rather than following them.
    """
    if dst.exists():
        raise ValueError(f"Destination already exists: {dst}")

    shutil.copytree(
        src,
        dst,
        symlinks=True,  # Copy symlinks as symlinks, don't follow
        ignore_dangling_symlinks=True,
    )
```

**Changes:**
```python
@click.option(
    "-o", "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for complete STAC catalog copy. If not specified, modifies catalog in place.",
)
def select_images(..., output_dir: Path | None, ...):
    if output_dir is not None:
        from ftw_dataset_tools.api.stac_items import copy_catalog
        copy_catalog(input_catalog, output_dir)
        working_catalog = output_dir
    else:
        working_catalog = input_catalog
```

**Acceptance criteria:**
- [ ] `-o, --output-dir` option added to select-images
- [ ] When specified, copies entire catalog before processing
- [ ] Original catalog left unchanged
- [ ] Symlinks copied as symlinks (not followed)
- [ ] All links updated in copied catalog

---

### 4.3 Review and ensure consistency across subcommands

**Audit checklist:**

| Command | Option | Current | Target |
|---------|--------|---------|--------|
| select-images | cloud filter | `--cloud-cover-pixel` | `--cloud-cover-chip` |
| create-dataset | cloud filter | `--cloud-cover-pixel` | `--cloud-cover-chip` |
| select-images | stac-host | has option | removed |
| create-dataset | stac-host | has option | removed |
| download-images | bands | free text | click.Choice |
| create-masks | output | flat | STAC-aware |

### Research Insights: Patterns

**Default value inconsistency detected** (from pattern-recognition-specialist):
- `--cloud-cover-scene` has different defaults: 75 in select-images, 10 in create-dataset
- After changes, this is moot since scene-level is being hardcoded to 75% internally

**Acceptance criteria:**
- [ ] All cloud filtering options use `--cloud-cover-chip`
- [ ] All defaults match across commands (2% for cloud cover)
- [ ] Asset naming consistent (`image`, `thumbnail`)
- [ ] Directory structure consistent across all create-* commands
- [ ] STAC link relations consistent

---

## Acceptance Criteria

### Functional Requirements

- [ ] Cloud filtering simplified to chip-level only
- [ ] `--cloud-cover-chip` replaces `--cloud-cover-pixel`
- [ ] Scene-level filtering hardcoded to 75% internally
- [ ] Default cloud cover is 2%
- [ ] `--stac-host` option removed everywhere
- [ ] `refine-cloud-cover` command removed
- [ ] `--nodata-max` filters scenes by nodata percentage
- [ ] `--bands` uses click.Choice with all 18 valid S2 bands
- [ ] JPEG previews generated by default in download-images
- [ ] Thumbnails added to STAC items with proper roles
- [ ] `-o, --output-dir` added to select-images
- [ ] create-masks uses consistent directory structure
- [ ] Exception handling in download-images STAC saves (moved to API layer)
- [ ] None datetime handled gracefully with timezone awareness
- [ ] STAC items have full year datetime range (Jan 1 - Dec 31)

### Non-Functional Requirements

- [ ] Nodata check adds < 200ms per scene (using overviews)
- [ ] Thumbnail generation adds < 0.5 seconds per image
- [ ] No breaking changes to existing STAC catalog reading
- [ ] Clear error messages for all failure modes
- [ ] STAC client cached for connection reuse

### Quality Gates

- [ ] All new code has tests
- [ ] `pytest` passes
- [ ] `pre-commit run --all-files` passes
- [ ] Documentation updated (docs/stac-extension.md)
- [ ] Path sanitization for STAC item IDs
- [ ] Use `click.FloatRange` for percentage options

---

## Success Metrics

1. **Simplification**: 3 fewer CLI options exposed to users
2. **Robustness**: Zero partial STAC catalog states on failure
3. **Usability**: JPEG previews enable quick visual verification
4. **Performance**: 50% reduction in processing time with optimizations

---

## Dependencies & Prerequisites

- Existing `select-images`, `download-images`, `create-dataset`, `create-masks` commands
- pystac library for STAC operations
- rasterio for COG reading (with GDAL HTTP optimization)
- Pillow for JPEG generation

---

## Risk Analysis & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Breaking existing workflows | High | Medium | Clear release notes, deprecation warnings |
| Nodata detection slow | Medium | Low | Use overviews, check metadata first |
| JPEG quality issues | Low | Low | Use percentile stretch, quality 85 |
| Partial state on failure | High | Low | Transaction-like save with cleanup in API layer |
| Path traversal | Medium | Low | Sanitize STAC item IDs, validate output paths |

---

## Files Changed Summary

### Modified Files
- `src/ftw_dataset_tools/cli.py` - Remove refine-cloud-cover registration
- `src/ftw_dataset_tools/commands/select_images.py` - Option renames, removals, additions
- `src/ftw_dataset_tools/commands/create_dataset.py` - Option renames, removals
- `src/ftw_dataset_tools/commands/download_images.py` - bands choice, preview, use API for saves
- `src/ftw_dataset_tools/commands/create_masks.py` - Directory structure
- `src/ftw_dataset_tools/api/imagery/scene_selection.py` - Remove stac_host, add nodata, cache client
- `src/ftw_dataset_tools/api/imagery/settings.py` - Remove MSPC, update defaults, add GDAL env vars
- `src/ftw_dataset_tools/api/stac.py` - Fix datetime range to full year
- `src/ftw_dataset_tools/api/masks.py` - STAC-aware paths
- `docs/stac-extension.md` - Clarify percentages

### Deleted Files
- `src/ftw_dataset_tools/commands/refine_cloud_cover.py`

### New Files
- `src/ftw_dataset_tools/api/imagery/nodata_analysis.py` - Nodata detection with overviews
- `src/ftw_dataset_tools/api/imagery/thumbnails.py` - JPEG generation with error handling
- `src/ftw_dataset_tools/api/stac_items.py` - Shared STAC item operations (saves, copies)

---

## References & Research

### Internal References
- Cloud filtering options: `src/ftw_dataset_tools/commands/select_images.py:168-194`
- SelectedScene class: `src/ftw_dataset_tools/api/imagery/scene_selection.py:44-62`
- STAC save operations: `src/ftw_dataset_tools/commands/download_images.py:220-233`
- Settings defaults: `src/ftw_dataset_tools/api/imagery/settings.py:59-71`
- Mask output paths: `src/ftw_dataset_tools/api/masks.py:70-112`

### External References
- [STAC Best Practices - Asset Roles](https://github.com/radiantearth/stac-spec/blob/master/best-practices.md)
- [PySTAC Documentation](https://github.com/stac-utils/pystac) - Context7: `/stac-utils/pystac`
- [Click Choice Type](https://click.palletsprojects.com/en/stable/options/#choice-options) - Context7: `/pallets/click`
- [Rasterio Windowed Reading](https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html) - Context7: `/rasterio/rasterio`
- [Sentinel-2 STAC Extension - s2:nodata_pixel_percentage](https://github.com/stac-extensions/sentinel-2)

### Related Work
- Current PR: feat/stac-imagery-workflow branch

---

## Research Agent Summaries

### kieran-python-reviewer
- Use tuple for `VALID_BANDS` constant (immutability)
- Remove unused `band` parameter in `calculate_nodata_percentage`
- Add `ThumbnailError` custom exception
- Ensure consistent timezone handling in datetime functions
- Split `_save_stac_items_safely` into smaller functions with dataclass context

### performance-oracle
- Use COG overviews (`overview_level=2`) for 60-70% faster reads
- Cache STAC client with `@lru_cache` to eliminate 7+ minutes overhead
- Add GDAL HTTP environment variables for multiplexing
- Consider combining nodata + cloud check for connection reuse
- Target: < 200ms per nodata check, 50% total time reduction

### security-sentinel
- Add path sanitization for STAC item IDs (prevent path traversal)
- Validate output paths stay within expected directories
- Use `click.FloatRange(0.0, 100.0)` for percentage options
- Handle symlinks safely when copying catalogs
- Add GDAL timeout configuration (`GDAL_HTTP_TIMEOUT=30`)

### code-simplicity-reviewer
- Current year handling in masks.py is appropriate (keep both modes)
- Consider deferring `--output-dir` if in-place is sufficient
- Error handling refactor is necessary complexity

### architecture-strategist
- Move `_save_stac_items_safely` to `api/stac_items.py`
- Create shared module for child item creation (eliminate duplication)
- New files `nodata_analysis.py` and `thumbnails.py` belong in `api/imagery/`

### pattern-recognition-specialist
- Rename `--no-data-max` to `--nodata-max` for rasterio/GDAL consistency
- Default value inconsistency detected in `--cloud-cover-scene` (moot after changes)
- Error handling pattern matches existing codebase conventions
