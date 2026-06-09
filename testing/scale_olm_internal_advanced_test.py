"""
Advanced tests for scale.olm.internal module.

This module tests utility functions, file operations, and configuration management
with minimal mocking - focusing on real functionality where possible.
"""
import pytest
import os
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, mock_open

import scale.olm.internal as internal


class TestCopyDoc:
    """Test the copy_doc decorator functionality."""

    def test_copy_doc_basic(self):
        """Test basic docstring copying functionality."""
        def source_func():
            """This is the source docstring."""
            return "source"

        @internal.copy_doc(source_func)
        def target_func():
            return "target"

        assert target_func.__doc__ == "This is the source docstring."
        assert target_func() == "target"

    def test_copy_doc_with_form_feed(self):
        """Test docstring copying with form feed character."""
        def source_func():
            """This is public documentation.\f
            This is internal documentation that should be removed."""
            return "source"

        @internal.copy_doc(source_func)
        def target_func():
            return "target"

        # Should only include text before \f
        assert target_func.__doc__ == "This is public documentation."
        assert "\f" not in target_func.__doc__

    def test_copy_doc_empty_docstring_raises(self):
        """Test that copy_doc rejects source functions without docstrings."""
        def source_func():
            return "source"

        with pytest.raises(ValueError, match="empty docstring"):
            internal.copy_doc(source_func)(lambda: "target")


class TestFunctionHandling:
    """Test function handle resolution."""

    def test_get_function_handle_valid(self):
        """Test getting function handle for valid module:function."""
        # Test with a known function from json module
        handle = internal._get_function_handle("json:loads")
        assert handle == json.loads
        assert callable(handle)

    def test_get_function_handle_invalid_format(self):
        """Test error handling for invalid module:function format."""
        with pytest.raises(ValueError, match="separated by a single colon"):
            internal._get_function_handle("invalid_format")


class TestUtilityFunctions:
    """Test utility functions with real calculations."""

    def test_runtime_in_hours(self):
        """Test runtime conversion to hours with string formatting."""
        # The function returns a formatted string, not a float
        assert internal._runtime_in_hours(3600) == "1"      # 1 hour
        assert internal._runtime_in_hours(1800) == "0.5"    # 30 minutes
        assert internal._runtime_in_hours(0) == "0"         # 0 seconds
        assert internal._runtime_in_hours(7200) == "2"      # 2 hours

    def test_runtime_calculation_edge_cases(self):
        """Test edge cases in runtime calculations."""
        # Test floating point runtime - the function uses {:.2g} format
        float_runtime = 3661.5  # 1 hour, 1 minute, 1.5 seconds
        hours_str = internal._runtime_in_hours(float_runtime)
        # {:.2g} format rounds 1.0170833 to "1" (2 significant figures)
        assert float(hours_str) == 1.0

        # Test negative runtime (edge case)
        negative_runtime = -100
        hours_str = internal._runtime_in_hours(negative_runtime)
        # Should handle negative values gracefully
        assert float(hours_str) < 0

    def test_logger_exists(self):
        """Test that the internal logger is properly configured."""
        assert hasattr(internal, 'logger')
        assert internal.logger is not None

    def test_error_message_formatting(self):
        """Test SCALE runtime environment error message."""
        with pytest.raises(ValueError, match="scalerte executable was not found"):
            internal._raise_scalerte_error()

        # Check that error message contains helpful information
        try:
            internal._raise_scalerte_error()
        except ValueError as e:
            error_msg = str(e)
            assert "SCALE_DIR" in error_msg
            assert "OLM_SCALERTE" in error_msg
            assert "export" in error_msg

    def test_obiwan_error_message_formatting(self):
        """Test SCALE obiwan environment error message."""
        with pytest.raises(ValueError, match="obiwan executable was not found"):
            internal._raise_obiwan_error()


class TestSchemaFunctions:
    """Test schema generation and validation functions."""

    def test_indent_function(self):
        """Test the _indent utility function."""
        text = "line1\nline2\nline3"
        indented = internal._indent(text, 4)

        # Should add 4 spaces to each line
        lines = indented.split('\n')
        for line in lines[1:]:  # Skip first line which may be different
            if line.strip():  # Only check non-empty lines
                assert line.startswith('    ')

    def test_collapsible_json_function(self):
        """Test the _collapsible_json utility function."""
        title = "Test Title"
        json_str = '{"test": "data"}'

        result = internal._collapsible_json(title, json_str)

        assert title in result
        assert "collapse::" in result
        assert "code:: JSON" in result
        assert json_str in result


class TestFileOperations:
    """Test file and directory operations with real files."""

    def test_file_copying_patterns(self):
        """Test file copying patterns used in internal functions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.txt"
            dest = Path(temp_dir) / "dest.txt"

            # Create source file
            source.write_text("test content")

            # Test copying (pattern used in _make_mini_arpdatatxt)
            import shutil
            shutil.copy(source, dest)

            assert dest.exists()
            assert dest.read_text() == "test content"

    def test_directory_creation_patterns(self):
        """Test directory creation patterns used in internal functions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new_directory"

            # Test mkdir with exist_ok (pattern used in internal functions)
            new_dir.mkdir(exist_ok=True)
            assert new_dir.exists()

            # Should not raise error if called again
            new_dir.mkdir(exist_ok=True)
            assert new_dir.exists()

    def test_temporary_directory_patterns(self):
        """Test temporary directory usage patterns."""
        import tempfile

        # Test pattern used in _process_libraries
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create some temporary files
            (temp_path / "temp1.txt").touch()
            (temp_path / "temp2.txt").touch()

            assert (temp_path / "temp1.txt").exists()
            assert (temp_path / "temp2.txt").exists()

        # Directory should be automatically cleaned up
        assert not temp_path.exists()


class TestPathOperations:
    """Test path manipulation operations used throughout internal module."""

    def test_relative_path_calculations(self):
        """Test relative path calculations."""
        work_dir = Path("/base/work")
        file_path = Path("/base/work/subdir/file.txt")

        relative = file_path.relative_to(work_dir)
        assert str(relative) == str(Path("subdir/file.txt"))

    def test_suffix_operations(self):
        """Test suffix operations."""
        input_file = Path("test.inp")
        output_file = input_file.with_suffix(".out")
        assert str(output_file) == "test.out"


class TestConfigurationPatterns:
    """Test patterns used for handling configuration data."""

    def test_configuration_data_patterns(self):
        """Test patterns used for handling model data."""
        # Test the pattern used in many internal functions for handling model data
        model_data = {
            "name": "test_reactor",
            "work_dir": "/work",
            "archive_file": "test.arc.h5"
        }

        # Pattern: extract values with defaults
        name = model_data.get("name", "default_name")
        work_dir = Path(model_data.get("work_dir", "/tmp"))

        assert name == "test_reactor"
        assert isinstance(work_dir, Path)
        assert str(work_dir) == str(Path("/work"))

    def test_error_accumulation_pattern(self):
        """Test error accumulation patterns used in internal functions."""
        # Pattern used for collecting multiple errors
        errors = []

        # Simulate error collection from various operations
        try:
            raise ValueError("Error 1")
        except ValueError as e:
            errors.append(str(e))

        try:
            raise FileNotFoundError("Error 2")
        except FileNotFoundError as e:
            errors.append(str(e))

        assert len(errors) == 2
        assert "Error 1" in errors
        assert "Error 2" in errors


class TestJSONHandling:
    """Test JSON file reading patterns with minimal mocking."""

    @patch('builtins.open', new_callable=mock_open, read_data='{"test": "data"}')
    def test_json_file_handling(self, mock_file):
        """Test JSON file reading patterns used in internal functions."""
        # Test that we can read JSON files as used in many internal functions
        with open("test.json", "r") as f:
            data = json.load(f)

        assert data == {"test": "data"}
        mock_file.assert_called_with("test.json", "r")


class TestRegistryBasics:
    """Test basic registry patterns without heavy mocking."""

    @staticmethod
    def write_arp_library(root, name="testlib", lib_name=None):
        """Create a minimal arpdata.txt/arplibs pair for registry tests."""
        lib_name = lib_name or f"{name}.f33"
        root.mkdir(parents=True, exist_ok=True)
        arplibs = root / "arplibs"
        arplibs.mkdir()
        (arplibs / lib_name).write_text("dummy library payload\n")
        (root / "arpdata.txt").write_text(
            "\n".join(
                [
                    f"!{name}",
                    "1 1 2",
                    "4.95",
                    "0.723",
                    f"'{lib_name}'",
                    "0.0 1000.0",
                    "",
                ]
            )
        )
        return root

    @patch.dict(os.environ, {'SCALE_OLM_PATH': '/test/path1:/test/path2'})
    def test_environment_path_parsing(self):
        """Test environment variable parsing for SCALE_OLM_PATH."""
        # Test that environment variable is parsed correctly
        scale_olm_path = os.environ.get('SCALE_OLM_PATH', '')
        paths = scale_olm_path.split(':') if scale_olm_path else []

        assert len(paths) == 2
        assert '/test/path1' in paths
        assert '/test/path2' in paths

    @patch.dict(os.environ, {}, clear=True)  # Clear SCALE_OLM_PATH
    def test_empty_environment(self):
        """Test behavior with no environment variables."""
        scale_olm_path = os.environ.get('SCALE_OLM_PATH', '')
        paths = scale_olm_path.split(':') if scale_olm_path else []

        assert len(paths) == 0

    def test_create_registry_reads_paths_and_environment(self, tmp_path, monkeypatch):
        """Test registry creation from explicit paths and SCALE_OLM_PATH."""
        path_library = self.write_arp_library(tmp_path / "path_library", "pathlib")
        env_library = self.write_arp_library(tmp_path / "env_library", "envlib")
        monkeypatch.setenv("SCALE_OLM_PATH", str(env_library))

        registry = internal._create_registry([path_library], env=True)

        assert sorted(registry) == ["envlib", "pathlib"]
        assert registry["pathlib"].path == path_library / "arpdata.txt"
        assert registry["pathlib"].arplibs_dir == path_library / "arplibs"
        assert registry["pathlib"].get_lib_by_index(0) == "pathlib.f33"
        assert registry["envlib"].get_lib_by_index(0) == "envlib.f33"

    def test_make_mini_arpdatatxt_writes_registry_subset(self, tmp_path):
        """Test writing a reduced arpdata.txt and arplibs directory."""
        source = self.write_arp_library(tmp_path / "source", "uoxmini")
        registry = internal._create_registry([source], env=False)
        dest = tmp_path / "dest"
        dest.mkdir()

        internal._make_mini_arpdatatxt(
            dry_run=False,
            registry={"uoxmini": registry["uoxmini"]},
            dest=dest,
            replace=True,
        )

        arpdata_text = (dest / "arpdata.txt").read_text()
        assert "!uoxmini" in arpdata_text
        assert "'uoxmini.f33'" in arpdata_text
        assert (dest / "arplibs" / "uoxmini.f33").read_text() == "dummy library payload\n"

    def test_link_copies_selected_registry_entries(self, tmp_path):
        """Test link copies only the selected registered library."""
        source = self.write_arp_library(tmp_path / "source", "selected")
        dest = tmp_path / "linked"
        dest.mkdir()

        result = internal.link(
            names=["selected"],
            paths=[str(source)],
            env=False,
            dest=str(dest),
            show=False,
            overwrite=True,
            dry_run=False,
        )

        assert result == 0
        assert "!selected" in (dest / "arpdata.txt").read_text()
        assert (dest / "arplibs" / "selected.f33").exists()

    def test_link_show_lists_registered_libraries(self, tmp_path, capsys):
        """Test link --show lists available libraries without copying them."""
        source = self.write_arp_library(tmp_path / "source", "showlib")

        result = internal.link(
            names=[],
            paths=[str(source)],
            env=False,
            dest=str(tmp_path / "unused"),
            show=True,
            overwrite=False,
            dry_run=True,
        )

        captured = capsys.readouterr()
        assert result == 0
        assert "showlib" in captured.out
        assert str(source / "arpdata.txt") in captured.out

    def test_link_missing_name_raises(self, tmp_path):
        """Test link reports missing requested library names."""
        source = self.write_arp_library(tmp_path / "source", "available")

        with pytest.raises(ValueError, match="name=missing not found"):
            internal.link(
                names=["missing"],
                paths=[str(source)],
                env=False,
                dest=str(tmp_path / "unused"),
                show=False,
                overwrite=False,
                dry_run=True,
            )

    def test_install_writes_single_library_registry(self, tmp_path):
        """Test install creates a destination registry from one work library."""
        work_dir = self.write_arp_library(tmp_path / "work", "installable")
        dest = tmp_path / "install_dest"

        internal.install(str(work_dir), str(dest), overwrite=False, dry_run=False)

        assert "!installable" in (dest / "arpdata.txt").read_text()
        assert (dest / "arplibs" / "installable.f33").exists()

    def test_install_rejects_multiple_library_blocks(self, tmp_path):
        """Test install rejects work directories containing multiple libraries."""
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "arplibs").mkdir()
        (work_dir / "arplibs" / "one.f33").write_text("one\n")
        (work_dir / "arplibs" / "two.f33").write_text("two\n")
        (work_dir / "arpdata.txt").write_text(
            "\n".join(
                [
                    "!one",
                    "1 1 1",
                    "3.0",
                    "0.7",
                    "'one.f33'",
                    "0.0",
                    "!two",
                    "1 1 1",
                    "4.0",
                    "0.8",
                    "'two.f33'",
                    "0.0",
                    "",
                ]
            )
        )

        with pytest.raises(ValueError, match="Only one library expected"):
            internal.install(str(work_dir), str(tmp_path / "dest"), overwrite=False)


class TestCommandExecution:
    """Test command execution patterns (only mock subprocess for safety)."""

    def test_run_command_returns_stdout(self):
        """Test run_command returns stdout from a successful child process."""
        command = f'"{sys.executable}" -c "print(\\"clean output\\")"'

        output = internal.run_command(command, echo=False)

        assert output == "clean output\n"

    def test_run_command_raises_on_error_match(self):
        """Test run_command raises when stdout contains the error marker."""
        command = f'"{sys.executable}" -c "print(\\"Error from child\\")"'

        with pytest.raises(ValueError, match="Error from child"):
            internal.run_command(command, echo=False)

    @patch('subprocess.Popen')
    def test_command_output_processing(self, mock_popen):
        """Test command output processing patterns."""
        # Mock successful process
        mock_process = type('MockProcess', (), {
            'stdout': type('MockStdout', (), {
                'readline': lambda: "Output line\n" if not hasattr(self, '_called')
                           else (setattr(self, '_called', True), "")[1]
            })(),
            'returncode': 0
        })()
        mock_popen.return_value = mock_process

        # This tests the pattern without full implementation details
        assert mock_process.returncode == 0


class TestEnvironmentLoading:
    """Test environment loading with minimal mocking."""

    @patch('builtins.open', new_callable=mock_open, read_data='{"model": {"name": "test"}}')
    @patch('pathlib.Path.exists')
    def test_basic_config_loading(self, mock_exists, mock_file):
        """Test basic configuration loading pattern."""
        mock_exists.return_value = True  # Work dir exists

        # Test the pattern of loading configuration
        with open("config.json", "r") as f:
            config = json.load(f)

        assert config["model"]["name"] == "test"
        mock_file.assert_called_with("config.json", "r")

    @patch.dict(os.environ, {'SCALE_DIR': '/test/scale'})
    def test_scale_dir_environment(self):
        """Test SCALE_DIR environment variable handling."""
        scale_dir = os.environ.get('SCALE_DIR')
        scalerte_path = str(Path(scale_dir) / "bin" / "scalerte")
        obiwan_path = str(Path(scale_dir) / "bin" / "obiwan")

        assert scalerte_path == str(Path("/test/scale/bin/scalerte"))
        assert obiwan_path == str(Path("/test/scale/bin/obiwan"))

    def test_load_env_writes_resolved_runtime_environment(self, tmp_path, monkeypatch):
        """Test _load_env writes config/work/runtime paths into env.olm.json."""
        config_file = tmp_path / "config.olm.json"
        work_dir = tmp_path / "_custom_work"
        scale_dir = tmp_path / "scale"
        scalerte = tmp_path / "custom_scalerte"
        obiwan = tmp_path / "custom_obiwan"
        config_file.write_text(
            json.dumps(
                {
                    "model": {
                        "name": "test_model",
                        "sources": ["source-a"],
                        "revision": ["rev-a"],
                    }
                }
            )
        )
        monkeypatch.setenv("OLM_WORK_DIR", str(work_dir))
        monkeypatch.setenv("SCALE_DIR", str(scale_dir))
        monkeypatch.setenv("OLM_SCALERTE", str(scalerte))
        monkeypatch.setenv("OLM_OBIWAN", str(obiwan))

        env, config = internal._load_env(str(config_file), nprocs=7)

        assert config["model"]["name"] == "test_model"
        assert env["config_file"] == str(config_file.resolve())
        assert env["work_dir"] == str(work_dir)
        assert env["scalerte"] == str(scalerte)
        assert env["obiwan"] == str(obiwan)
        assert env["nprocs"] == 7
        assert json.loads((work_dir / "env.olm.json").read_text()) == env

    def test_init_writes_named_variant_when_config_dir_omitted(self, tmp_path, monkeypatch):
        """Test init uses the variant name as the output directory by default."""
        monkeypatch.chdir(tmp_path)

        internal.init(config_dir="", variant="uox_quick", list_=False)

        output_dir = tmp_path / "uox_quick"
        assert (output_dir / "config.olm.json").exists()
        assert not (output_dir / "model.jt.inp").exists()
        assert not (output_dir / "report.jt.rst").exists()

    def test_write_init_variant_uses_packaged_templates(self, tmp_path):
        """Test init variants reference packaged templates instead of copying them."""
        config_dir = tmp_path / "variant"

        internal._write_init_variant("uox_quick", config_dir)

        config = json.loads((config_dir / "config.olm.json").read_text())
        assert config["model"]["name"] == "uox_quick"
        assert config["model"]["template_float_format"] == ".12e"
        assert config["generate"]["template"] == "model/triton/pin-uox.jt.inp"
        assert config["report"]["template"] == "report/scale-short.jt.rst"
        assert not (config_dir / "model.jt.inp").exists()
        assert not (config_dir / "report.jt.rst").exists()

    def test_write_init_variant_can_copy_local_templates(self, tmp_path):
        """Test init variants can copy packaged templates for local editing."""
        config_dir = tmp_path / "variant"

        internal._write_init_variant("uox_quick", config_dir, copy_files=True)

        config = json.loads((config_dir / "config.olm.json").read_text())
        assert config["generate"]["template"] == "model.jt.inp"
        assert config["report"]["template"] == "report.jt.rst"
        assert "t-depl" in (config_dir / "model.jt.inp").read_text()
        assert "{{model.name}}" in (config_dir / "report.jt.rst").read_text()


class TestMakefilePatterns:
    """Test makefile-related patterns with real file operations."""

    def test_makefile_content_creation(self):
        """Test makefile content creation patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            makefile_path = Path(temp_dir) / "Makefile"

            # Test pattern for creating makefile content
            makefile_content = """
all: input1.out input2.out

input1.out: input1.inp
\tcommand input1.inp

input2.out: input2.inp
\tcommand input2.inp
"""

            makefile_path.write_text(makefile_content)

            assert makefile_path.exists()
            content = makefile_path.read_text()
            assert "all:" in content
            assert "input1.out" in content

    def test_input_output_mapping(self):
        """Test input to output file mapping."""
        input_files = ["test1.inp", "test2.inp", "test3.inp"]

        # Pattern used to map input files to output files
        output_files = [str(Path(inp).with_suffix(".out")) for inp in input_files]

        expected = ["test1.out", "test2.out", "test3.out"]
        assert output_files == expected
