"""
Meta-tests for the OLM test suite itself.

This module contains tests that verify the integrity and completeness
of the test suite, including manifest verification and test inventory tracking.
"""

import ast
import glob
import os
import sys
from pathlib import Path
import pytest


def get_current_test_functions():
    """Extract all current test function names from test files."""
    test_functions = set()
    
    # Get all test files in this directory
    test_files = glob.glob('testing/scale_olm_*.py')
    
    for file_path in test_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    test_functions.add(node.name)
        
        except Exception as e:
            pytest.fail(f"Error parsing {file_path}: {e}")
    
    return sorted(test_functions)


def load_manifest():
    """Load the test function manifest file."""
    manifest_path = Path('testing/test_functions_manifest.txt')
    
    if not manifest_path.exists():
        pytest.fail(
            f"Test function manifest not found at {manifest_path}. "
            f"Run: python scripts/regenerate_manifest.py"
        )
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return sorted(line.strip() for line in f if line.strip())
    except Exception as e:
        pytest.fail(f"Error reading manifest {manifest_path}: {e}")


def generate_regeneration_script():
    """Generate script content for regenerating the manifest."""
    return '''#!/usr/bin/env python3
"""
Regenerate the test function manifest file.

This script updates testing/test_functions_manifest.txt with the current
set of test function names found in the test suite.
"""

import ast
import glob
from pathlib import Path

def get_current_test_functions():
    """Extract all current test function names from test files."""
    test_functions = set()
    
    # Get all test files
    test_files = glob.glob('testing/scale_olm_*.py')
    
    for file_path in test_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    test_functions.add(node.name)
        
        except Exception as e:
            print(f"Warning: Error parsing {file_path}: {e}")
    
    return sorted(test_functions)

def main():
    """Regenerate the manifest file."""
    current_functions = get_current_test_functions()
    manifest_path = Path('testing/test_functions_manifest.txt')
    
    print(f"Found {len(current_functions)} unique test functions")
    
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for func_name in current_functions:
            f.write(f"{func_name}\\n")
    
    print(f"Updated manifest: {manifest_path}")
    print(f"Total functions: {len(current_functions)}")

if __name__ == "__main__":
    main()
'''


class TestSuiteIntegrity:
    """Tests that verify the integrity of the test suite itself."""
    
    def test_manifest_is_current(self):
        """Verify that the test function manifest matches current test functions."""
        # Get current test functions
        current_functions = get_current_test_functions()
        
        # Load manifest
        manifest_functions = load_manifest()
        
        # Compare
        missing_from_manifest = set(current_functions) - set(manifest_functions)
        extra_in_manifest = set(manifest_functions) - set(current_functions)
        
        if missing_from_manifest or extra_in_manifest:
            error_msg = [
                "Test function manifest is out of date!",
                f"Current functions: {len(current_functions)}",
                f"Manifest functions: {len(manifest_functions)}",
            ]
            
            if missing_from_manifest:
                error_msg.extend([
                    f"",
                    f"Missing from manifest ({len(missing_from_manifest)}):"
                ] + [f"  + {func}" for func in sorted(missing_from_manifest)])
            
            if extra_in_manifest:
                error_msg.extend([
                    f"",
                    f"Extra in manifest ({len(extra_in_manifest)}):"
                ] + [f"  - {func}" for func in sorted(extra_in_manifest)])
            
            error_msg.extend([
                f"",
                f"To update the manifest, run:",
                f"  python scripts/regenerate_manifest.py",
                f"",
                f"Or create the regeneration script with:",
                f"  python -c \"import testing.scale_olm_test; testing.scale_olm_test.create_regeneration_script()\"",
            ])
            
            pytest.fail("\n".join(error_msg))
        
        # If we get here, everything matches
        assert len(current_functions) == len(manifest_functions)
        assert current_functions == manifest_functions
    
    def test_no_duplicate_function_names_across_files(self):
        """Verify that function names are unique across all test files."""
        function_locations = {}  # func_name -> [file1, file2, ...]
        
        test_files = glob.glob('testing/scale_olm_*.py')
        
        for file_path in test_files:
            filename = os.path.basename(file_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                        func_name = node.name
                        if func_name not in function_locations:
                            function_locations[func_name] = []
                        function_locations[func_name].append(filename)
            
            except Exception as e:
                pytest.fail(f"Error parsing {file_path}: {e}")
        
        # Find duplicates
        duplicates = {func: files for func, files in function_locations.items() if len(files) > 1}
        
        if duplicates:
            error_msg = [
                f"Found {len(duplicates)} duplicate test function names:",
                ""
            ]
            
            for func_name, files in sorted(duplicates.items()):
                error_msg.append(f"  {func_name}:")
                for file in sorted(files):
                    error_msg.append(f"    - {file}")
                error_msg.append("")
            
            error_msg.extend([
                "Duplicate function names can cause test discovery issues.",
                "Consider renaming functions or consolidating test files."
            ])
            
            pytest.fail("\n".join(error_msg))
    
    def test_manifest_baseline_count(self):
        """Verify that we have the expected baseline number of test functions."""
        manifest_functions = load_manifest()
        
        # This test ensures we don't accidentally lose large numbers of tests
        EXPECTED_BASELINE = 200  # Conservative baseline
        
        if len(manifest_functions) < EXPECTED_BASELINE:
            pytest.fail(
                f"Test function count ({len(manifest_functions)}) is below expected baseline ({EXPECTED_BASELINE}). "
                f"This suggests tests may have been lost. Please verify the manifest is correct."
            )


def create_regeneration_script():
    """Create the regeneration script for updating the manifest."""
    script_path = Path('scripts/regenerate_manifest.py')
    script_path.parent.mkdir(exist_ok=True)
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(generate_regeneration_script())
    
    # Make executable on Unix systems
    if hasattr(os, 'chmod'):
        os.chmod(script_path, 0o755)
    
    print(f"Created regeneration script: {script_path}")
    return str(script_path)


if __name__ == "__main__":
    # When run directly, create the regeneration script
    create_regeneration_script()
    print("You can now run: python scripts/regenerate_manifest.py") 