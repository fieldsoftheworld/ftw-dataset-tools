# FTW Dataset Tools

CLI tools for creating the [Fields of the World](https://fieldsofthe.world/) benchmark dataset.

## Installation

```bash
uv pip install ftw-dataset-tools
```

Or for development:

```bash
git clone https://github.com/cholmes/ftw-dataset-tools
cd ftw-dataset-tools
uv sync --dev
```

## Usage

```bash
# Show available commands
ftwd --help
```

### create-dataset

Create a complete training dataset from a single fields file. This is the main command that orchestrates the entire pipeline.

```bash
# Basic usage with required split-type
ftwd create-dataset austria_fields.parquet --split-type random-uniform

# Specify output directory and dataset name
ftwd create-dataset fields.parquet --field-dataset austria --split-type block3x3 -o ./austria_dataset

# Custom split percentages
ftwd create-dataset fields.parquet --split-type random-uniform --split-percents 70 20 10

# Generate only specific mask types
ftwd create-dataset fields.parquet --split-type block3x3 --mask-types semantic_2_class,semantic_3_class

# For presence-only labels (background class = 3 instead of 0)
ftwd create-dataset fields.parquet --split-type block3x3 --presence-only

# If fields lack determination_datetime column, specify year
ftwd create-dataset fields.parquet --split-type block3x3 --year 2023

# Custom options
ftwd create-dataset fields.parquet --split-type block3x3 --min-coverage 1.0 --resolution 5.0 --workers 8
```

**Options:**
- `--split-type` - **Required.** Split strategy: `random-uniform` (random assignment of chips) or `block3x3` (3x3 blocks of chips assigned together for spatial coherence)
- `--split-percents` - Train/val/test split percentages as three integers that sum to 100 (default: 80 10 10)
- `--mask-types` - Comma-separated list of mask types to generate: `instance`, `semantic_2_class`, `semantic_3_class` (default: all three)
- `--presence-only` - Flag indicating labels are presence-only; background class value will be 3 instead of 0
- `-o, --output-dir` - Output directory (defaults to `{input_stem}-dataset/`)
- `--field-dataset` - Dataset name for output filenames (defaults to input filename stem)
- `--year` - Year for temporal extent (only required if fields lack `determination_datetime` column)
- `--min-coverage` - Minimum coverage percentage to include grids (default: 0.01)
- `--resolution` - Pixel resolution in meters for masks (default: 10.0)
- `--workers` - Number of parallel workers (default: half of CPUs)
- `--skip-reproject` - Fail if input is not EPSG:4326 instead of auto-reprojecting

**Output structure:**
```
{name}-dataset/
├── {dataset}_fields.parquet          # Field boundaries in EPSG:4326
├── {dataset}_chips.parquet           # Grid cells with coverage stats and split assignments
├── {dataset}_boundary_lines.parquet  # Polygon boundaries as lines
├── catalog.json                      # STAC catalog
├── source/
│   └── collection.json              # STAC collection for source data
├── chips/
│   ├── collection.json              # STAC collection for chips
│   ├── items.parquet                # STAC items as parquet
│   └── {grid_id}/                   # Individual STAC item JSON files
└── label_masks/
    ├── instance/                    # Instance segmentation masks
    ├── semantic_2class/            # 2-class semantic masks (field/boundary)
    └── semantic_3class/            # 3-class semantic masks (field/boundary/background)
```

### create-chips

Create chip definitions with field coverage statistics. Calculates what percentage of each grid cell is covered by field boundary polygons.

```bash
# Fetch grid from FTW grid on Source Coop
ftwd create-chips fields.parquet

# Use local grid file
ftwd create-chips fields.parquet --grid-file grid.parquet

# Filter by minimum coverage
ftwd create-chips fields.parquet --min-coverage 0.01

# Reproject if CRS don't match
ftwd create-chips fields.parquet --reproject
```

**Options:**
- `--grid-file` - Local grid file (if not specified, fetches from FTW grid on Source Coop)
- `-o, --output` - Output file path (defaults to `chips_<fields_basename>.parquet`)
- `--coverage-col` - Name for coverage column (default: `field_coverage_pct`)
- `--min-coverage` - Exclude grid cells below this coverage percentage
- `--reproject` - Reproject both inputs to EPSG:4326 if CRS don't match
- `--grid-geom-col`, `--fields-geom-col` - Geometry column names (auto-detected)
- `--grid-bbox-col`, `--fields-bbox-col` - Bbox column names (auto-detected)

### create-masks

Create raster masks from vector boundaries for each grid cell. Outputs Cloud Optimized GeoTIFFs (COGs).

```bash
# Create semantic 2-class masks
ftwd create-masks chips.parquet fields.parquet boundary_lines.parquet --field-dataset austria

# Create instance masks
ftwd create-masks chips.parquet fields.parquet lines.parquet --field-dataset france --mask-type instance

# Custom settings
ftwd create-masks chips.parquet fields.parquet lines.parquet --field-dataset spain --min-coverage 1.0 --resolution 5.0
```

**Options:**
- `-o, --output-dir` - Output directory (default: `./masks`)
- `--field-dataset` - Dataset name for output filenames (required)
- `--mask-type` - Type of mask: `instance`, `semantic_2_class`, or `semantic_3_class` (default: `semantic_2_class`)
- `--grid-id-col` - Column name for grid cell ID (default: `id`)
- `--coverage-col` - Column name for coverage percentage (default: `field_coverage_pct`)
- `--min-coverage` - Minimum coverage to process (default: 0.01)
- `--resolution` - Pixel resolution in CRS units (default: 10.0)
- `--workers` - Number of parallel workers (default: half of CPUs)

### create-boundaries

Convert polygon geometries to boundary lines using ST_Boundary.

```bash
# Single file
ftwd create-boundaries fields.parquet

# Process entire directory
ftwd create-boundaries ./data/

# Custom output
ftwd create-boundaries fields.parquet -o ./output/ --prefix lines_
```

**Options:**
- `-o, --output-dir` - Output directory (defaults to same directory as input)
- `--prefix` - Prefix for output filenames (default: `boundary_lines_`)

### create-ftw-grid

Create a hierarchical FTW grid from 1km MGRS cells.

```bash
# Single file
ftwd create-ftw-grid mgrs_1km.parquet

# Custom grid size
ftwd create-ftw-grid mgrs_1km.parquet --km-size 4

# Process partitioned folder
ftwd create-ftw-grid ./mgrs_partitioned/ -o ./ftw_output/
```

**Options:**
- `-o, --output` - Output path (required for folder input)
- `--km-size` - Grid cell size in km (default: 2). Must divide 100 evenly (1, 2, 4, 5, 10, 20, 25, 50, 100)

**Output columns:**
- `gzd` - Grid Zone Designator
- `mgrs_10km` - 10km MGRS code from source
- `id` - Unique FTW grid cell ID (e.g., `ftw-33UXPA0410`)
- `geometry` - Unioned polygon of child cells

### get-grid

Fetch FTW grid cells from cloud source that cover the input file's extent.

```bash
# Basic usage
ftwd get-grid fields.parquet

# Precise geometry matching (slower)
ftwd get-grid fields.parquet --precise

# Custom output
ftwd get-grid fields.parquet -o custom_grid.parquet
```

**Options:**
- `-o, --output` - Output file path (defaults to `<input>_grid.parquet`)
- `--precise` - Use geometry union for precise matching (excludes grids in bbox gaps)
- `--grid-source` - URL/path to the grid source

### Reprojection

For reprojecting GeoParquet files to a different CRS, use [geoparquet-io](https://geoparquet.io/):

```bash
gpio reproject input.parquet -o output.parquet --target-crs EPSG:4326
```

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run tests
uv run pytest

# Run linting
uv run ruff check .

# Format code
uv run ruff format .
```

## License

Apache-2.0
