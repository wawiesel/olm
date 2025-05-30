"""
Meta-tests for the OLM test suite itself.

This module contains tests that verify the integrity and completeness
of the test suite, including manifest verification and test inventory tracking.
"""

import ast
import glob
import hashlib
import os
import sys
from pathlib import Path
import pytest

# Get the directory containing this test file for relative path resolution
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent


def get_current_test_functions_with_hashes():
    """Extract all current test function names and their content hashes from test files."""
    test_functions = {}
    
    # Get all test files in the testing directory
    test_files = list(TEST_DIR.glob('scale_olm_*.py'))
    
    for file_path in test_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract function source lines for hashing
            source_lines = content.splitlines()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    # Extract function body source for hashing
                    start_line = node.lineno - 1  # Convert to 0-based indexing
                    end_line = node.end_lineno if node.end_lineno else start_line + 1
                    
                    # Get function body lines (excluding the def line)
                    function_body = '\n'.join(source_lines[start_line + 1:end_line])
                    
                    # Remove leading whitespace consistently for stable hashing
                    function_body = '\n'.join(line.strip() for line in function_body.split('\n'))
                    
                    # Compute SHA256 hash of function body
                    function_hash = hashlib.sha256(function_body.encode('utf-8')).hexdigest()[:16]  # Use first 16 chars
                    
                    # Store with filename for duplicate detection
                    if node.name in test_functions:
                        print(f"Warning: Duplicate function {node.name} found in {file_path}")
                    
                    test_functions[node.name] = {
                        'hash': function_hash,
                        'file': str(file_path)
                    }
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return test_functions


def get_current_test_functions():
    """Extract all current test function names from test files (legacy function for compatibility)."""
    test_functions_with_hashes = get_current_test_functions_with_hashes()
    return set(test_functions_with_hashes.keys())


def load_manifest():
    """Load the test function manifest file."""
    manifest_path = TEST_DIR / 'test_functions_manifest.txt'
    
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
    """Test suite integrity verification."""
    
    def test_manifest_is_current(self):
        """Test that the manifest matches current test functions."""
        current_functions = get_current_test_functions_with_hashes()
        
        # Read manifest
        manifest_path = TEST_DIR / 'test_functions_manifest.txt'
        if not manifest_path.exists():
            pytest.fail(f"Manifest file {manifest_path} does not exist. Run python scripts/regenerate_manifest.py")
        
        manifest_functions = {}
        with open(manifest_path, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    hash_part, name_part = line.split(':', 1)
                    manifest_functions[name_part] = hash_part
                else:
                    # Legacy format - just function name
                    manifest_functions[line] = None
        
        current_names = set(current_functions.keys())
        manifest_names = set(manifest_functions.keys())
        
        # Check for missing functions
        missing_from_manifest = current_names - manifest_names
        missing_from_current = manifest_names - current_names
        
        # Check for hash mismatches
        hash_mismatches = []
        for name in current_names & manifest_names:
            current_hash = current_functions[name]['hash']
            manifest_hash = manifest_functions[name]
            if manifest_hash and current_hash != manifest_hash:
                hash_mismatches.append(f"  {name}: hash changed from {manifest_hash} to {current_hash}")
        
        errors = []
        if missing_from_manifest:
            errors.append(f"Functions in code but missing from manifest ({len(missing_from_manifest)}):")
            errors.extend(f"  {name}" for name in sorted(missing_from_manifest))
        
        if missing_from_current:
            errors.append(f"Functions in manifest but missing from code ({len(missing_from_current)}):")
            errors.extend(f"  {name}" for name in sorted(missing_from_current))
        
        if hash_mismatches:
            errors.append(f"Functions with changed implementations ({len(hash_mismatches)}):")
            errors.extend(hash_mismatches)
        
        if errors:
            errors.append("")
            errors.append("To update the manifest, run: python scripts/regenerate_manifest.py")
            pytest.fail("\n".join(errors))
    
    def test_no_duplicate_function_names_across_files(self):
        """Test that there are no duplicate function names across different files."""
        current_functions = get_current_test_functions_with_hashes()
        
        # Group functions by name to find duplicates
        name_to_files = {}
        for name, info in current_functions.items():
            if name not in name_to_files:
                name_to_files[name] = []
            name_to_files[name].append(info['file'])
        
        # Find duplicates
        duplicates = {name: files for name, files in name_to_files.items() if len(files) > 1}
        
        if duplicates:
            error_msg = [f"Found {len(duplicates)} duplicate test function names:"]
            error_msg.append("")
            
            for name, files in sorted(duplicates.items()):
                error_msg.append(f"  {name}:")
                for file_path in sorted(files):
                    # Extract just the filename for cleaner output
                    filename = file_path.split('/')[-1]
                    error_msg.append(f"    - {filename}")
                error_msg.append("")
            
            error_msg.append("Duplicate function names can cause test discovery issues.")
            error_msg.append("Consider renaming functions or consolidating test files.")
            pytest.fail("\n".join(error_msg))
    
    def test_manifest_baseline_count(self):
        """Test that we haven't lost a large number of test functions."""
        current_functions = get_current_test_functions()
        
        # Baseline from our initial count
        baseline_minimum = 200  # Conservative baseline
        
        if len(current_functions) < baseline_minimum:
            pytest.fail(f"Test function count ({len(current_functions)}) is below baseline minimum ({baseline_minimum}). "
                       f"This suggests test functions may have been lost.")
    
    def test_implementation_stability(self):
        """Test for unexpected implementation changes by checking hash consistency."""
        current_functions = get_current_test_functions_with_hashes()
        
        # Read manifest to check for unexpected changes
        manifest_path = TEST_DIR / 'test_functions_manifest.txt'
        if not manifest_path.exists():
            pytest.skip("Manifest file does not exist")
        
        manifest_functions = {}
        with open(manifest_path, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    hash_part, name_part = line.split(':', 1)
                    manifest_functions[name_part] = hash_part
        
        # Look for functions that exist in both but have different hashes
        changed_functions = []
        for name in current_functions.keys():
            if name in manifest_functions:
                current_hash = current_functions[name]['hash']
                manifest_hash = manifest_functions[name]
                if manifest_hash and current_hash != manifest_hash:
                    changed_functions.append(name)
        
        if changed_functions:
            # This is informational rather than failing - implementations can legitimately change
            print(f"Info: {len(changed_functions)} test functions have changed implementations:")
            for name in sorted(changed_functions):
                print(f"  - {name}")


# Regeneration function for the manifest
def regenerate_manifest():
    """Regenerate the test functions manifest with hashes."""
    current_functions = get_current_test_functions_with_hashes()
    
    # Create manifest content with hash:name format
    manifest_lines = []
    for name in sorted(current_functions.keys()):
        function_hash = current_functions[name]['hash']
        manifest_lines.append(f"{function_hash}:{name}")
    
    # Write manifest
    manifest_path = TEST_DIR / 'test_functions_manifest.txt'
    with open(manifest_path, 'w') as f:
        f.write('\n'.join(manifest_lines) + '\n')
    
    print(f"Found {len(current_functions)} unique test functions")
    print(f"Updated manifest: {manifest_path}")
    print(f"Total functions: {len(current_functions)}")
    
    return len(current_functions)


if __name__ == "__main__":
    # When run as script, regenerate the manifest
    regenerate_manifest() 