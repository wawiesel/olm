"""
Tests for the OLM run module.

This module tests the run functionality in scale.olm.run.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import scale.olm.run as run
import scale.olm.internal as internal


class TestRunModule:
    """Test suite for run module functions."""

    def test_schema_makefile(self):
        """Test schema generation for makefile."""
        schema = run._schema_makefile()
        assert isinstance(schema, dict)
        # Should contain schema information
        assert 'properties' in schema or 'type' in schema

    def test_schema_makefile_with_state(self):
        """Test schema generation with state for makefile."""
        schema = run._schema_makefile(with_state=True)
        assert isinstance(schema, dict)

    def test_test_args_makefile(self):
        """Test default test arguments for makefile."""
        args = run._test_args_makefile()
        assert isinstance(args, dict)
        assert args["_type"] == run._TYPE_MAKEFILE
        assert "dry_run" in args
        assert args["dry_run"] == False

    def test_test_args_makefile_with_state(self):
        """Test test arguments with state for makefile."""
        args = run._test_args_makefile(with_state=True)
        assert isinstance(args, dict)
        assert args["_type"] == run._TYPE_MAKEFILE

    def test_module_exports_run(self):
        """Test that module exports the expected functions."""
        assert "makefile" in run.__all__
        assert hasattr(run, "makefile")

    def test_type_constant_run(self):
        """Test that the type constant is properly defined."""
        assert run._TYPE_MAKEFILE == "scale.olm.run:makefile"


class TestMakefileFunction:
    """Test suite for the makefile function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.work_dir = Path(self.temp_dir) / "work"
        self.work_dir.mkdir()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def create_mock_generate_json(self, input_files=None):
        """Create mock generate.olm.json file."""
        if input_files is None:
            input_files = ["perms/input1.inp", "perms/input2.inp"]
        
        generate_data = {
            "perms": [
                {"input_file": inp} for inp in input_files
            ]
        }
        
        generate_json = self.work_dir / "generate.olm.json"
        with open(generate_json, "w") as f:
            json.dump(generate_data, f)
        
        return generate_data

    def test_makefile_with_work_dir_in_env(self):
        """Test makefile function with work_dir specified in environment."""
        # Create mock generate.olm.json
        mock_data = self.create_mock_generate_json()
        
        _env = {"work_dir": str(self.work_dir)}
        _model = {"name": "test_model"}

        with patch('scale.olm.internal._execute_makefile') as mock_execute:
            mock_execute.return_value = {"status": "success"}
            
            result = run.makefile(
                dry_run=False,
                _model=_model,
                _env=_env
            )

            # Verify _execute_makefile was called
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args
            
            # Check arguments passed to _execute_makefile
            assert call_args[0][0] == False  # dry_run
            assert call_args[0][1] == _env  # _env
            assert "base_path" in call_args[1]
            assert "input_list" in call_args[1]
            
            # Check that base_path points to perms directory
            base_path = call_args[1]["base_path"]
            assert "perms" in str(base_path)
            
            # Check input_list contains relative paths
            input_list = call_args[1]["input_list"]
            assert isinstance(input_list, list)
            for inp in input_list:
                assert not inp.startswith("perms/")  # Should be relative

    def test_makefile_without_work_dir_in_env(self):
        """Test makefile function without work_dir in environment."""
        _env = {}  # No work_dir specified
        _model = {"name": "test_model"}

        with patch('scale.olm.core.TempDir') as mock_tempdir, \
             patch('scale.olm.internal._execute_makefile') as mock_execute:
            
            # Mock TempDir
            mock_temp_instance = MagicMock()
            mock_temp_instance.path = Path(self.temp_dir)
            mock_tempdir.return_value = mock_temp_instance
            
            # Create the expected work directory structure
            work_path = Path(self.temp_dir) / "_work"
            work_path.mkdir()
            
            # Create mock generate.olm.json in the temp work directory
            generate_data = {
                "perms": [
                    {"input_file": "perms/test.inp"}
                ]
            }
            with open(work_path / "generate.olm.json", "w") as f:
                json.dump(generate_data, f)
            
            mock_execute.return_value = {"status": "success"}
            
            result = run.makefile(
                dry_run=True,
                _model=_model,
                _env=_env
            )

            # Verify TempDir was created
            mock_tempdir.assert_called_once()
            
            # Verify _execute_makefile was called
            mock_execute.assert_called_once()

    def test_makefile_dry_run(self):
        """Test makefile function with dry_run=True."""
        self.create_mock_generate_json()
        
        _env = {"work_dir": str(self.work_dir)}
        _model = {}

        with patch('scale.olm.internal._execute_makefile') as mock_execute:
            mock_execute.return_value = {"dry_run": True}
            
            result = run.makefile(
                dry_run=True,
                _model=_model,
                _env=_env
            )

            # Verify dry_run was passed correctly
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args
            assert call_args[0][0] == True  # dry_run should be True

    def test_makefile_input_file_processing(self):
        """Test that input files are processed correctly."""
        # Create generate.olm.json with various input file paths
        input_files = [
            "perms/subdir/input1.inp",
            "perms/input2.inp",
            "perms/another/deep/input3.inp"
        ]
        self.create_mock_generate_json(input_files)
        
        _env = {"work_dir": str(self.work_dir)}

        with patch('scale.olm.internal._execute_makefile') as mock_execute:
            mock_execute.return_value = {}
            
            run.makefile(_env=_env)

            # Check that input_list contains correctly processed paths
            call_args = mock_execute.call_args
            input_list = call_args[1]["input_list"]
            
            expected_relative_paths = [
                "subdir/input1.inp",
                "input2.inp", 
                "another/deep/input3.inp"
            ]
            
            assert len(input_list) == len(expected_relative_paths)
            for expected in expected_relative_paths:
                assert expected in input_list

    def test_makefile_missing_generate_json(self):
        """Test makefile behavior when generate.olm.json is missing."""
        _env = {"work_dir": str(self.work_dir)}
        
        # Don't create generate.olm.json file
        
        with pytest.raises(FileNotFoundError):
            run.makefile(_env=_env)

    def test_makefile_invalid_generate_json(self):
        """Test makefile behavior with invalid generate.olm.json."""
        # Create invalid JSON file
        invalid_json = self.work_dir / "generate.olm.json"
        with open(invalid_json, "w") as f:
            f.write("invalid json content")
        
        _env = {"work_dir": str(self.work_dir)}
        
        with pytest.raises(json.JSONDecodeError):
            run.makefile(_env=_env)

    def test_makefile_empty_perms_list(self):
        """Test makefile with empty perms list."""
        generate_data = {"perms": []}
        generate_json = self.work_dir / "generate.olm.json"
        with open(generate_json, "w") as f:
            json.dump(generate_data, f)
        
        _env = {"work_dir": str(self.work_dir)}

        with patch('scale.olm.internal._execute_makefile') as mock_execute:
            mock_execute.return_value = {}
            
            run.makefile(_env=_env)

            # Should still call _execute_makefile with empty input_list
            call_args = mock_execute.call_args
            input_list = call_args[1]["input_list"]
            assert input_list == []


class TestRunIntegration:
    """Integration tests for run functionality."""

    def test_makefile_function_signature(self):
        """Test that makefile has the expected function signature."""
        import inspect
        
        sig = inspect.signature(run.makefile)
        params = list(sig.parameters.keys())
        
        expected_params = ['dry_run', '_model', '_env', '_type']
        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"

    def test_makefile_default_parameters(self):
        """Test makefile default parameter values."""
        import inspect
        
        sig = inspect.signature(run.makefile)
        
        # Check default values
        assert sig.parameters['dry_run'].default == False
        assert sig.parameters['_model'].default == {}
        assert sig.parameters['_env'].default == {}
        assert sig.parameters['_type'].default == None

    def test_makefile_docstring(self):
        """Test that makefile has proper documentation."""
        assert run.makefile.__doc__ is not None
        assert "Makefile" in run.makefile.__doc__
        assert "_env" in run.makefile.__doc__
        assert "_model" in run.makefile.__doc__ 