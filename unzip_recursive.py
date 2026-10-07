#!/usr/bin/env python3
"""
Recursively unzip all zip files in a directory and its subdirectories,
then remove the zip files after successful extraction.
"""

import os
import zipfile
import sys
from pathlib import Path


def unzip_recursive(directory: Path, processed_zips: set = None, remove_zips: bool = True):
    """
    Recursively unzip all zip files in the given directory.
    
    Args:
        directory: The directory to search for zip files
        processed_zips: Set of already processed zip file paths (to avoid infinite loops)
        remove_zips: Whether to delete zip files after successful extraction
    """
    if processed_zips is None:
        processed_zips = set()
    
    directory = directory.resolve()
    
    # Find all zip files in the directory (including subdirectories)
    zip_files = list(directory.rglob("*.zip"))
    
    if not zip_files:
        print(f"No zip files found in {directory}")
        return
    
    for zip_path in zip_files:
        zip_path_resolved = zip_path.resolve()
        
        # Skip if already processed
        if zip_path_resolved in processed_zips:
            continue
        
        processed_zips.add(zip_path_resolved)
        
        # Create extraction directory (same name as zip file without extension)
        extract_dir = zip_path.parent / zip_path.stem
        
        print(f"Extracting: {zip_path} -> {extract_dir}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Check for zip slip vulnerability
                for member in zip_ref.namelist():
                    member_path = (extract_dir / member).resolve()
                    if not member_path.is_relative_to(extract_dir.resolve()):
                        print(f"  WARNING: Skipping potentially malicious path: {member}")
                        continue
                
                zip_ref.extractall(extract_dir)
                print(f"  Successfully extracted to {extract_dir}")
                
            # Remove the zip file after successful extraction
            if remove_zips:
                try:
                    zip_path.unlink()
                    print(f"  Removed zip file: {zip_path}")
                except Exception as e:
                    print(f"  WARNING: Failed to remove zip file {zip_path}: {e}")
                
        except zipfile.BadZipFile:
            print(f"  ERROR: {zip_path} is not a valid zip file")
            continue
        except Exception as e:
            print(f"  ERROR: Failed to extract {zip_path}: {e}")
            continue
        
        # After extraction, check for nested zip files in the extracted directory
        # and process them recursively
        unzip_recursive(extract_dir, processed_zips, remove_zips)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Recursively unzip all zip files and optionally remove them")
    parser.add_argument("directory", nargs="?", default=".", help="Directory to process (default: current)")
    parser.add_argument("--keep-zips", action="store_true", help="Keep zip files after extraction (default: remove them)")
    args = parser.parse_args()
    
    target_dir = Path(args.directory)
    
    if not target_dir.exists():
        print(f"Error: Directory {target_dir} does not exist")
        sys.exit(1)
    
    if not target_dir.is_dir():
        print(f"Error: {target_dir} is not a directory")
        sys.exit(1)
    
    remove_zips = not args.keep_zips
    
    print(f"Starting recursive unzip in: {target_dir}")
    print(f"Remove zip files after extraction: {remove_zips}")
    print("-" * 50)
    
    unzip_recursive(target_dir, remove_zips=remove_zips)
    
    print("-" * 50)
    print("Done!")


if __name__ == "__main__":
    main()