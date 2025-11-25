#!/usr/bin/env python3
"""
Fix editable install to use local files.

This script patches the editable finder to insert itself at the beginning
of sys.meta_path so it takes precedence over regular packages.

USAGE:
    After running `pip install -e .`, run this script once:
    
    python fix_editable_import.py
    
    Or make it executable and run:
    ./fix_editable_import.py
    
    You only need to run this once after installation. The __init__.py
    will handle moving the finder in memory for subsequent imports.
"""
import sys
import site
from pathlib import Path
import glob

def find_finder_file():
    """Find the editable finder file for flash-attn-cute."""
    # Check common site-packages locations
    for site_packages in site.getsitepackages():
        site_path = Path(site_packages)
        # Look for the finder file
        pattern = str(site_path / "__editable___flash_attn_cute_*_finder.py")
        matches = glob.glob(pattern)
        if matches:
            return Path(matches[0])
    
    # Fallback: try the specific path
    fallback = Path("/usr/local/lib/python3.12/dist-packages/__editable___flash_attn_cute_0_1_0_finder.py")
    if fallback.exists():
        return fallback
    
    return None

def fix_finder(finder_file):
    """Patch the finder file to use insert(0) instead of append."""
    content = finder_file.read_text()
    
    # Check if already fixed
    if "sys.meta_path.insert(0, _EditableFinder)" in content:
        return False, "already fixed"
    
    # Replace append with insert(0)
    if "sys.meta_path.append(_EditableFinder)" in content:
        content = content.replace(
            "sys.meta_path.append(_EditableFinder)",
            "sys.meta_path.insert(0, _EditableFinder)"
        )
        finder_file.write_text(content)
        return True, "fixed"
    
    return False, "pattern not found"

def main():
    finder_file = find_finder_file()
    
    if not finder_file:
        print("Error: Could not find editable finder file for flash-attn-cute")
        print("Make sure you've run: pip install -e .")
        sys.exit(1)
    
    print(f"Found finder file: {finder_file}")
    fixed, status = fix_finder(finder_file)
    
    if fixed:
        print("✓ Fixed editable finder to use local files")
        print("  The finder now takes precedence over regular packages")
        print("\nYou may need to restart Python for changes to take effect.")
    elif status == "already fixed":
        print("✓ Finder already configured correctly")
    else:
        print(f"⚠ Could not find the expected pattern in finder file")
        print("  The file may have a different structure")

if __name__ == "__main__":
    main()

