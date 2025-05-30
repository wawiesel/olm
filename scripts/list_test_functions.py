#!/usr/bin/env python3
"""
List all test functions in the testing directory.

This script provides an inventory of all test functions and classes
for verification and tracking purposes. Useful for ensuring no tests
are lost during refactoring or reorganization.

Usage:
    python scripts/list_test_functions.py              # Default: summary view
    python scripts/list_test_functions.py --detailed   # Show all function names
    python scripts/list_test_functions.py --count-only # Just show counts
"""

import ast
import argparse
import glob
import os
from pathlib import Path

def extract_test_items(file_path):
    """Extract test functions and classes from a Python test file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('test_'):
                    functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                if node.name.startswith('Test'):
                    classes.append(node.name)
        
        return sorted(functions), sorted(classes)
    
    except SyntaxError as e:
        print(f"Warning: Syntax error in {file_path}: {e}")
        return [], []
    except Exception as e:
        print(f"Warning: Error reading {file_path}: {e}")
        return [], []

def get_test_files(directory='testing'):
    """Get all test files in the specified directory."""
    pattern = f"{directory}/scale_olm_*.py"
    return sorted(glob.glob(pattern))

def main():
    parser = argparse.ArgumentParser(description='List all test functions in the testing directory')
    parser.add_argument('--detailed', action='store_true', 
                       help='Show all function and class names (default: summary only)')
    parser.add_argument('--count-only', action='store_true',
                       help='Show only total counts')
    parser.add_argument('--directory', default='testing',
                       help='Directory to scan for test files (default: testing)')
    
    args = parser.parse_args()
    
    # Find all test files
    test_files = get_test_files(args.directory)
    
    if not test_files:
        print(f"No test files found in {args.directory}/")
        return 1
    
    # Extract test items from all files
    total_functions = 0
    total_classes = 0
    file_details = []
    all_function_names = set()
    all_class_names = set()
    
    for file_path in test_files:
        functions, classes = extract_test_items(file_path)
        filename = os.path.basename(file_path)
        
        file_details.append({
            'file': filename,
            'functions': functions,
            'classes': classes,
            'function_count': len(functions),
            'class_count': len(classes)
        })
        
        total_functions += len(functions)
        total_classes += len(classes)
        all_function_names.update(functions)
        all_class_names.update(classes)
    
    # Output based on requested format
    if args.count_only:
        print(f"Total test functions: {total_functions}")
        print(f"Total test classes: {total_classes}")
        print(f"Total test files: {len(test_files)}")
        print(f"Unique function names: {len(all_function_names)}")
        print(f"Unique class names: {len(all_class_names)}")
    
    elif args.detailed:
        print("=== DETAILED TEST FUNCTION INVENTORY ===\n")
        
        for detail in file_details:
            print(f"{detail['file']}:")
            print(f"  Functions ({detail['function_count']}):")
            for func in detail['functions']:
                print(f"    - {func}")
            if detail['classes']:
                print(f"  Classes ({detail['class_count']}):")
                for cls in detail['classes']:
                    print(f"    - {cls}")
            print()
        
        print(f"SUMMARY:")
        print(f"  Total files: {len(test_files)}")
        print(f"  Total functions: {total_functions}")
        print(f"  Total classes: {total_classes}")
        print(f"  Unique function names: {len(all_function_names)}")
        
        if len(all_function_names) != total_functions:
            print(f"  ⚠️  Warning: {total_functions - len(all_function_names)} duplicate function names detected")
    
    else:
        # Default summary view
        print("=== TEST FUNCTION SUMMARY ===\n")
        
        for detail in file_details:
            print(f"{detail['file']:40} {detail['function_count']:3} functions, {detail['class_count']:2} classes")
        
        print(f"\n{'='*50}")
        print(f"Total: {len(test_files)} files, {total_functions} functions, {total_classes} classes")
        
        # Check for duplicates
        if len(all_function_names) != total_functions:
            duplicate_count = total_functions - len(all_function_names)
            print(f"⚠️  Warning: {duplicate_count} duplicate function names detected")
            print("   Run with --detailed to see all function names")
    
    return 0

if __name__ == "__main__":
    exit(main()) 