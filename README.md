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

# Add field coverage statistics to a grid
ftwd add-field-stats grid.parquet fields.parquet

# Specify output file
ftwd add-field-stats grid.parquet fields.parquet -o output.parquet
```

## Python API

```python
from ftw_dataset_tools import add_field_stats

result = add_field_stats(
    grid_file="grid.parquet",
    fields_file="fields.parquet",
    output_file="output.parquet",
)

print(f"Total cells: {result.total_cells}")
print(f"Cells with coverage: {result.cells_with_coverage}")
print(f"Average coverage: {result.average_coverage}%")
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
