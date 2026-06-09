"""
Tests for the OLM report module.

This module tests the report generation functionality in scale.olm.report.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import scale.olm.report as report
import scale.olm.internal as internal
import scale.olm.core as core


class TestReportModule:
    """Test suite for report module functions."""

    def test_schema_rst2pdf(self):
        """Test schema generation for rst2pdf."""
        schema = report._schema_rst2pdf()
        assert isinstance(schema, dict)
        # Should contain schema information
        assert 'properties' in schema or 'type' in schema

    def test_schema_rst2pdf_with_state(self):
        """Test schema generation with state for rst2pdf."""
        schema = report._schema_rst2pdf(with_state=True)
        assert isinstance(schema, dict)

    def test_test_args_rst2pdf(self):
        """Test default test arguments for rst2pdf."""
        args = report._test_args_rst2pdf()
        assert isinstance(args, dict)
        assert args["_type"] == report._TYPE_RST2PDF
        assert "template" in args
        assert args["template"] == "report.jt.rst"

    def test_test_args_rst2pdf_with_state(self):
        """Test test arguments with state for rst2pdf."""
        args = report._test_args_rst2pdf(with_state=True)
        assert isinstance(args, dict)
        assert args["_type"] == report._TYPE_RST2PDF

    def test_rst2pdf_dry_run(self):
        """Test rst2pdf function with dry_run=True."""
        result = report.rst2pdf(dry_run=True)
        assert result == {}

    def test_module_exports_report(self):
        """Test that module exports the expected functions."""
        assert "rst2pdf" in report.__all__
        assert hasattr(report, "rst2pdf")

    def test_type_constant_report(self):
        """Test that the type constant is properly defined."""
        assert report._TYPE_RST2PDF == "scale.olm.report:rst2pdf"


class TestRst2pdfFunction:
    """Test suite for the rst2pdf function with mocked file operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.work_dir = Path(self.temp_dir) / "work"
        self.work_dir.mkdir()
        self.config_dir = Path(self.temp_dir) / "config"
        self.config_dir.mkdir()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def create_mock_data_files(self):
        """Create mock data files for testing."""
        # Create mock JSON data files
        mock_data = {"test": "data"}
        
        for stage in ["generate", "run", "assemble", "check"]:
            json_file = self.work_dir / f"{stage}.olm.json"
            with open(json_file, "w") as f:
                json.dump({stage: mock_data}, f)

    def test_rst2pdf_template_loading(self):
        """Test that rst2pdf properly loads template files."""
        # Create template file
        template_content = "Test template: {{ model.name }}"
        template_file = self.config_dir / "test_template.rst"
        with open(template_file, "w") as f:
            f.write(template_content)

        # Create mock data files
        self.create_mock_data_files()

        # Mock environment and model
        _env = {
            "config_file": str(self.config_dir / "config.json"),
            "work_dir": str(self.work_dir)
        }
        _model = {"name": "test_model"}

        # Mock the external dependencies
        with patch('scale.olm.core.TemplateManager.expand_text') as mock_expand, \
             patch('scale.olm.internal.run_command') as mock_run:
            
            mock_expand.return_value = "Expanded template content"
            
            result = report.rst2pdf(
                template="test_template.rst",
                _model=_model,
                _env=_env
            )

            # Verify template was loaded and processed
            mock_expand.assert_called_once()
            mock_run.assert_called_once()
            
            # Check result structure
            assert isinstance(result, dict)
            assert "model" in result
            assert "generate" in result
            assert "run" in result
            assert "assemble" in result
            assert "check" in result
            assert "_" in result

    def test_rst2pdf_uses_packaged_template(self):
        """Test rst2pdf resolves packaged templates when no local file exists."""
        self.create_mock_data_files()

        _env = {
            "config_file": str(self.config_dir / "config.json"),
            "work_dir": str(self.work_dir),
        }
        _model = {"name": "test_model"}

        with patch('scale.olm.core.TemplateManager.expand_text') as mock_expand, \
             patch('scale.olm.internal.run_command') as mock_run:

            mock_expand.return_value = "Expanded template content"

            result = report.rst2pdf(
                template="report/scale-short.jt.rst",
                _model=_model,
                _env=_env,
            )

            template_path = Path(result["_"]["template"])
            assert template_path.name == "scale-short.jt.rst"
            assert template_path.parent.name == "report"
            mock_expand.assert_called_once()
            mock_run.assert_called_once()

    def test_rst2pdf_uses_model_template_float_format(self):
        """Test report template expansion uses the model float format."""
        self.create_mock_data_files()

        template_file = self.config_dir / "test.rst"
        template_file.write_text("model value={{ model.value }}")

        _env = {
            "config_file": str(self.config_dir / "config.json"),
            "work_dir": str(self.work_dir),
        }
        _model = {
            "name": "test_model",
            "template_float_format": ".4e",
            "value": 1.23456789,
        }

        with patch('scale.olm.internal.run_command') as mock_run:
            report.rst2pdf(
                template="test.rst",
                _model=_model,
                _env=_env,
            )

            assert (self.work_dir / "test_model.rst").read_text() == (
                "model value=1.2346e+00"
            )
            mock_run.assert_called_once()

    def test_rst2pdf_data_file_creation(self):
        """Test that rst2pdf creates proper data files."""
        # Create template file
        template_file = self.config_dir / "test.rst"
        with open(template_file, "w") as f:
            f.write("Test template")

        # Create mock data files
        self.create_mock_data_files()

        _env = {
            "config_file": str(self.config_dir / "config.json"),
            "work_dir": str(self.work_dir)
        }
        _model = {"name": "test_model"}

        with patch('scale.olm.core.TemplateManager.expand_text') as mock_expand, \
             patch('scale.olm.internal.run_command') as mock_run:
            
            mock_expand.return_value = "Expanded content"
            
            result = report.rst2pdf(
                template="test.rst",
                _model=_model,
                _env=_env
            )

            # Check that report.olm.json was created
            report_json = self.work_dir / "report.olm.json"
            assert report_json.exists()
            
            # Check RST file was created
            rst_file = self.work_dir / "test_model.rst"
            assert rst_file.exists()
            
            # Verify rst2pdf command was called
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "rst2pdf" in call_args
            assert "-s twocolumn" in call_args
            assert "test_model.rst" in call_args

    def test_rst2pdf_result_metadata(self):
        """Test that rst2pdf returns proper metadata."""
        # Setup minimal test environment
        template_file = self.config_dir / "test.rst"
        with open(template_file, "w") as f:
            f.write("Template")

        self.create_mock_data_files()

        _env = {
            "config_file": str(self.config_dir / "config.json"),
            "work_dir": str(self.work_dir)
        }
        _model = {"name": "model"}

        with patch('scale.olm.core.TemplateManager.expand_text') as mock_expand, \
             patch('scale.olm.internal.run_command') as mock_run:
            
            mock_expand.return_value = "Expanded template content"
            
            result = report.rst2pdf(
                template="test.rst",
                _model=_model,
                _env=_env
            )

            # Check metadata in result
            assert "_" in result
            metadata = result["_"]
            assert "work_dir" in metadata
            assert "template" in metadata
            assert "pdf_file" in metadata
            assert "rst_file" in metadata
            
            # Verify paths are correct
            assert str(self.work_dir) in metadata["work_dir"]
            assert "test.rst" in metadata["template"]
            assert "model.pdf" in metadata["pdf_file"]
            assert "model.rst" in metadata["rst_file"]

    def test_rst2pdf_missing_template(self):
        """Test rst2pdf behavior with missing template file."""
        _env = {
            "config_file": str(self.config_dir / "config.json"),
            "work_dir": str(self.work_dir)
        }
        _model = {"name": "test"}

        # Should raise FileNotFoundError when template doesn't exist
        with pytest.raises(FileNotFoundError):
            report.rst2pdf(
                template="nonexistent.rst",
                _model=_model,
                _env=_env
            )

    def test_rst2pdf_missing_data_files(self):
        """Test rst2pdf behavior with missing data files."""
        # Create template but not data files
        template_file = self.config_dir / "test.rst"
        with open(template_file, "w") as f:
            f.write("Template")

        _env = {
            "config_file": str(self.config_dir / "config.json"),
            "work_dir": str(self.work_dir)
        }
        _model = {"name": "test"}

        # Should raise FileNotFoundError when data files don't exist
        with pytest.raises(FileNotFoundError):
            report.rst2pdf(
                template="test.rst",
                _model=_model,
                _env=_env
            )


class TestReportIntegration:
    """Integration tests for report functionality."""

    def test_rst2pdf_function_signature(self):
        """Test that rst2pdf has the expected function signature."""
        import inspect
        
        sig = inspect.signature(report.rst2pdf)
        params = list(sig.parameters.keys())
        
        expected_params = ['template', '_model', '_env', 'dry_run', '_type']
        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"

    def test_rst2pdf_default_parameters(self):
        """Test rst2pdf default parameter values."""
        import inspect
        
        sig = inspect.signature(report.rst2pdf)
        
        # Check default values
        assert sig.parameters['template'].default == ""
        assert sig.parameters['_model'].default == {}
        assert sig.parameters['_env'].default == {}
        assert sig.parameters['dry_run'].default == False
        assert sig.parameters['_type'].default == None

    def test_logging_behavior(self):
        """Test that rst2pdf logs appropriate messages."""
        with patch('scale.olm.internal.logger') as mock_logger:
            # Test dry run logging
            report.rst2pdf(dry_run=True)

            # Dry run should not log anything
            mock_logger.info.assert_not_called()

    def test_scale_short_report_formats_quality_tables(self):
        """Test report source uses compact q-score formatting and path display."""
        template_path = (
            Path(__file__).parents[1]
            / "scale"
            / "olm"
            / "templates"
            / "report"
            / "scale-short.jt.rst"
        )
        template_text = template_path.read_text()
        data = {
            "model": {
                "name": "uox_quick",
                "description": "test report",
                "sources": {},
                "revision": [],
                "notes": [],
            },
            "assemble": {"date": "2026-06-09", "space": {}},
            "run": {
                "version": "7.0.b12",
                "total_runtime_hrs": 1.23456,
                "runs": [
                    {
                        "input_file": "/tmp/example/work/perms/hash/model_abcdef.inp",
                        "output_file": "/tmp/example/work/perms/hash/model_abcdef.out",
                        "success": True,
                        "runtime_hrs": 0.004321,
                    }
                ],
            },
            "generate": {"static": {}},
            "check": {
                "test_pass": False,
                "sequence": [
                    {
                        "test_pass": False,
                        "q1": 0.6812437810945273,
                        "target_q1": 0.7,
                        "q2": 0.9658407960199005,
                        "target_q2": 0.95,
                        "test_pass_time": False,
                        "metric": "grams_per_initial_hm",
                        "units": "g/gIHM",
                        "eps0": 1.0e-12,
                        "epsa": 1.0e-6,
                        "epsr": 1.0e-3,
                        "m": 20100,
                        "wr": 6407,
                        "wa": 51,
                        "fr": 0.3187562189054726,
                        "fa": 0.002537313432835821,
                        "time_quality_image": "/tmp/example/work/q1-q2-by-time.png",
                        "time_quality": [
                            {
                                "time_days": 25.0,
                                "burnup_gwd_per_mtihm": 0.987483,
                                "q1": 0.6812437810945273,
                                "q2": 0.9658407960199005,
                                "test_pass": False,
                            }
                        ],
                        "first_failed_time_quality": {
                            "time_days": 25.0,
                            "burnup_gwd_per_mtihm": 0.987483,
                            "q1": 0.6812437810945273,
                            "q2": 0.9658407960199005,
                            "limiting_score": "q1",
                        },
                        "worst_time_quality": {
                            "time_days": 25.0,
                            "burnup_gwd_per_mtihm": 0.987483,
                            "q1": 0.6812437810945273,
                            "q2": 0.9658407960199005,
                            "limiting_score": "q1",
                        },
                    }
                ],
            },
        }

        rendered = core.TemplateManager.expand_text(
            template_text,
            data,
            src_path=str(template_path),
        )

        assert "time_quality_image" not in rendered
        assert "/tmp/example/work/perms/hash/model_abcdef.out" not in rendered
        assert "model_abcdef.out" in rendered
        assert ":code:`model_abcdef.inp`" in rendered
        assert "0.681243" not in rendered
        assert "0.681" in rendered
        assert "0.966" in rendered
        assert "first fail" in rendered
        assert "Q-score by high-order time::" in rendered
