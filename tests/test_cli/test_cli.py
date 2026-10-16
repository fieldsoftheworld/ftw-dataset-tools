"""Tests for the ftwd command group and lazy imports."""

import json
import subprocess
import sys
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

import ftw_dataset_tools
import ftw_dataset_tools.api
import ftw_dataset_tools.api.imagery
from ftw_dataset_tools.cli import COMMAND_MODULES, cli

HEAVY_MODULES = ("geopandas", "pandas", "matplotlib", "geoparquet_io")
SRC_DIR = str(Path(ftw_dataset_tools.__file__).parents[1])


def _loaded_after(code: str) -> list[str]:
    """Heavy modules present in ``sys.modules`` after running ``code`` in a fresh interpreter."""
    script = (
        f"import sys\nsys.path.insert(0, {SRC_DIR!r})\n{code}\n"
        f"print(json.dumps([m for m in {HEAVY_MODULES!r} if m in sys.modules]))"
    )
    out = subprocess.run(
        [sys.executable, "-c", f"import json\n{script}"], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


class TestCommandGroup:
    """Tests for lazy command registration."""

    @pytest.mark.parametrize("module", COMMAND_MODULES)
    def test_command_resolves(self, module: str) -> None:
        name = module.replace("_", "-")
        command = cli.get_command(click.Context(cli), name)
        assert isinstance(command, click.Command)
        assert command.name == name

    def test_help_lists_all_commands(self) -> None:
        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        for module in COMMAND_MODULES:
            assert module.replace("_", "-") in result.output

    def test_unknown_command(self) -> None:
        result = CliRunner().invoke(cli, ["no-such-command"])
        assert result.exit_code != 0
        assert "No such command" in result.output


class TestLazyExports:
    """Tests for package-level re-exports."""

    @pytest.mark.parametrize(
        "package", [ftw_dataset_tools, ftw_dataset_tools.api, ftw_dataset_tools.api.imagery]
    )
    def test_all_names_resolve(self, package) -> None:
        for name in package.__all__:
            assert getattr(package, name) is not None

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(AttributeError, match="no_such_name"):
            _ = ftw_dataset_tools.api.no_such_name

    def test_submodule_import_still_works(self) -> None:
        from ftw_dataset_tools.api import field_stats

        assert field_stats.add_field_stats is ftw_dataset_tools.api.add_field_stats


class TestStartupImports:
    """Heavy libraries should only load when a command needs them."""

    def test_version_skips_heavy_imports(self) -> None:
        code = (
            "from click.testing import CliRunner\n"
            "from ftw_dataset_tools.cli import cli\n"
            "assert CliRunner().invoke(cli, ['--version']).exit_code == 0"
        )
        assert _loaded_after(code) == []

    def test_help_skips_heavy_imports(self) -> None:
        code = (
            "from click.testing import CliRunner\n"
            "from ftw_dataset_tools.cli import cli\n"
            "assert CliRunner().invoke(cli, ['--help']).exit_code == 0"
        )
        assert _loaded_after(code) == []
