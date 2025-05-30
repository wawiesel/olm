#!/usr/bin/env python3
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
            f.write(f"{func_name}\n")
    
    print(f"Updated manifest: {manifest_path}")
    print(f"Total functions: {len(current_functions)}")

if __name__ == "__main__":
    main()
