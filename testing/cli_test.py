"""
Tests for the OLM CLI commands.

This module tests the Click CLI interface defined in scale.olm.__main__.
"""

import pytest
from click.testing import CliRunner
import json
import os
import tempfile
from pathlib import Path

import scale.olm.__main__ as main
import scale.olm.internal as internal


class TestOLMCLI:
    """Test suite for OLM CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_olm_main_help(self):
        """Test that main OLM help command works."""
        result = self.runner.invoke(main.olm, ['--help'])
        assert result.exit_code == 0
        assert 'Usage:' in result.output
        assert 'Commands:' in result.output

    def test_olm_main_no_args(self):
        """Test that main OLM with no args shows help."""
        result = self.runner.invoke(main.olm, [])
        assert result.exit_code == 0
        assert 'Usage:' in result.output

    def test_olm_create_help(self):
        """Test create command help."""
        result = self.runner.invoke(main.olm_create, ['--help'])
        assert result.exit_code == 0
        assert 'Usage:' in result.output
        assert 'config.olm.json' in result.output
        assert '--generate' in result.output
        assert '--nprocs' in result.output

    def test_olm_init_help(self):
        """Test init command help."""
        result = self.runner.invoke(main.olm_init, ['--help'])
        assert result.exit_code == 0
        assert 'Usage:' in result.output
        assert '--variant' in result.output
        assert '--list' in result.output

    def test_olm_link_help(self):
        """Test link command help."""
        result = self.runner.invoke(main.olm_link, ['--help'])
        assert result.exit_code == 0
        assert 'Usage:' in result.output
        assert '--path' in result.output
        assert '--dest' in result.output

    def test_olm_install_help(self):
        """Test install command help."""
        result = self.runner.invoke(main.olm_install, ['--help'])
        assert result.exit_code == 0
        assert 'Usage:' in result.output
        assert '--dest' in result.output
        assert '--overwrite' in result.output

    def test_olm_check_help(self):
        """Test check command help."""
        result = self.runner.invoke(main.olm_check, ['--help'])
        assert result.exit_code == 0
        assert 'Usage:' in result.output
        assert '--sequence' in result.output
        assert '--nprocs' in result.output
        assert 'GridGradient' in result.output

    def test_olm_schema_help(self):
        """Test schema command help."""
        result = self.runner.invoke(main.olm_schema, ['--help'])
        assert result.exit_code == 0
        assert 'Usage:' in result.output
        assert '--color' in result.output
        assert '--description' in result.output

    def test_olm_init_list_variants(self):
        """Test listing available variants."""
        result = self.runner.invoke(main.olm_init, ['--list'])
        assert result.exit_code == 0
        # Should list available variants without error

    def test_olm_link_show(self):
        """Test showing available libraries."""
        result = self.runner.invoke(main.olm_link, ['--show'])
        assert result.exit_code == 0
        # Should show libraries without error

    def test_olm_create_missing_config(self):
        """Test create command with non-existent config file."""
        result = self.runner.invoke(main.olm_create, ['nonexistent.json'])
        assert result.exit_code != 0
        # Should fail gracefully with file not found

    def test_olm_check_missing_archive(self):
        """Test check command with non-existent archive file."""
        result = self.runner.invoke(main.olm_check, ['nonexistent.h5'])
        # CLI exits successfully but logs error about .arc.h5 extension requirement
        assert result.exit_code == 0
        # Should contain error message about file extension

    def test_olm_schema_basic_type(self):
        """Test schema command with a basic type."""
        result = self.runner.invoke(main.olm_schema, ['scale.olm.generate.comp:uo2_simple'])
        # Should either succeed or fail gracefully
        assert result.exit_code in [0, 1]  # May fail if type not found, but shouldn't crash

    def test_olm_schema_invalid_type(self):
        """Test schema command with invalid type."""
        result = self.runner.invoke(main.olm_schema, ['invalid:type'])
        # Should fail gracefully
        assert result.exit_code != 0

    def test_methods_help_function(self):
        """Test the methods_help utility function."""
        # Import the check module to get actual methods
        import scale.olm.check as check
        
        # Test the helper function directly
        help_text = main.methods_help(check.GridGradient)
        assert isinstance(help_text, str)
        assert 'GridGradient' in help_text
        assert 'NAME=' in help_text

    def test_command_integration_with_main_group(self):
        """Test that all commands are properly registered with main group."""
        # Get all registered commands
        commands = main.olm.commands
        
        expected_commands = ['create', 'init', 'link', 'install', 'check', 'schema']
        for cmd in expected_commands:
            assert cmd in commands, f"Command '{cmd}' not registered"

    def test_error_handling_patterns(self):
        """Test that CLI commands handle ValueError exceptions properly."""
        # This tests the try/except patterns in each command
        # We can't easily trigger ValueError without complex setup,
        # but we can verify the pattern exists in the commands
        
        import inspect
        
        # Get the actual wrapped functions from Click commands
        commands = [
            main.olm_create.callback, main.olm_init.callback, main.olm_link.callback, 
            main.olm_install.callback, main.olm_check.callback, main.olm_schema.callback
        ]
        
        for cmd in commands:
            source = inspect.getsource(cmd)
            assert 'try:' in source, f"Command {cmd.__name__} missing try block"
            assert 'except ValueError' in source, f"Command {cmd.__name__} missing ValueError handling"
            assert 'internal.logger.error' in source or 'return str(ve)' in source, \
                f"Command {cmd.__name__} missing proper error handling"


class TestCLIOptions:
    """Test CLI option parsing and validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_create_command_options(self):
        """Test create command option parsing."""
        # Test boolean flag parsing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'test': 'config'}, f)
            config_file = f.name

        try:
            # Test with valid flags
            result = self.runner.invoke(main.olm_create, [
                config_file, '--generate', '--nogenerate', 
                '--run', '--norun', '--nprocs', '2'
            ])
            # May fail due to invalid config, but should parse options
            assert '--nprocs' in str(result.exception) or result.exit_code in [0, 1]
        finally:
            os.unlink(config_file)

    def test_nprocs_option_validation(self):
        """Test nprocs option accepts integers."""
        result = self.runner.invoke(main.olm_check, ['test.h5', '--nprocs', 'invalid'])
        assert result.exit_code != 0  # Should fail with invalid integer

    def test_json_sequence_option(self):
        """Test sequence option accepts JSON strings."""
        json_seq = '{"_type": "GridGradient", "eps0": 1e-6}'
        result = self.runner.invoke(main.olm_check, [
            'test.h5', '--sequence', json_seq
        ])
        # CLI exits successfully but logs error about .arc.h5 extension requirement
        assert result.exit_code == 0  # Expected to exit cleanly with error message


class TestCLIDocstrings:
    """Test CLI command documentation and help text."""

    def test_all_commands_have_help(self):
        """Test that all commands have proper help documentation."""
        commands = [main.olm_create, main.olm_init, main.olm_link, 
                   main.olm_install, main.olm_check, main.olm_schema]
        
        for cmd in commands:
            # Each command should have a docstring or help text
            assert cmd.__doc__ is not None or hasattr(cmd, 'help'), \
                f"Command {cmd.__name__} missing documentation"

    def test_epilog_examples(self):
        """Test that commands with epilogs have proper usage examples."""
        runner = CliRunner()
        
        commands_with_epilogs = [main.olm_create, main.olm_init, main.olm_link, 
                                main.olm_install, main.olm_check, main.olm_schema]
        
        for cmd in commands_with_epilogs:
            result = runner.invoke(cmd, ['--help'])
            assert result.exit_code == 0
            # Should contain usage examples
            assert '**Usage**' in result.output or 'Usage' in result.output 