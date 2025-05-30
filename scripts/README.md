# Utility Scripts

This directory contains utility scripts for development and testing.

## `list_test_functions.py`

Inventories all test functions and classes in the testing directory.

### Usage

```bash
# Summary view (default)
python scripts/list_test_functions.py

# Show all function and class names
python scripts/list_test_functions.py --detailed

# Show only counts
python scripts/list_test_functions.py --count-only

# Scan different directory
python scripts/list_test_functions.py --directory testing_backup
```

### Purpose

- **Track test coverage**: Monitor total number of test functions
- **Verify refactoring**: Ensure no tests are lost during reorganization
- **Detect duplicates**: Identify duplicate test function names across files
- **Generate inventory**: Create comprehensive lists for documentation

### Output

- **Summary**: Files with function/class counts
- **Detailed**: All function and class names organized by file  
- **Count-only**: Just the totals and unique counts
- **Duplicate detection**: Warns if function names are duplicated

This script is particularly useful for verifying that test consolidation or reorganization preserves all test functionality. 