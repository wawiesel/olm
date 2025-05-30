#!/usr/bin/env python3
"""
Regenerate the test functions manifest with content hashes.

This script extracts all test function names and their implementation hashes
from the testing directory and updates the manifest file.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path so we can import the test module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the regeneration function from our test module
from testing.scale_olm_test import regenerate_manifest

if __name__ == "__main__":
    # Change to project root directory
    os.chdir(project_root)
    
    # Generate the manifest
    count = regenerate_manifest()
    
    print(f"✅ Successfully updated manifest with {count} test functions including content hashes")
    print("📝 Manifest format: hash:function_name")
    print("🔍 This will detect both function changes AND implementation changes")
