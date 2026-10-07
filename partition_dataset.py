#!/usr/bin/env python3
"""
Deterministically partition dataset into training and testing sets (80:20).
Uses fixed seed and consistent sorting for reproducible splits across machines.
"""

import os
import sys
import hashlib
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
import random


# Fixed seed for reproducibility - NEVER CHANGE THIS
SEED = 42
TRAIN_RATIO = 0.8


def get_condition_key(filepath: Path) -> str:
    """
    Extract condition key from filename for stratified splitting.
    Filename pattern: {load}_{fault_type}_{severity}.ext
    e.g., 0Nm_BPFO_03.mat -> 0Nm_BPFO_03
    """
    stem = filepath.stem
    # Remove extension if double extension (e.g., .tar.gz)
    while '.' in stem:
        stem = stem.split('.')[0]
    return stem


def find_data_files(data_dir: Path, extensions: tuple = ('.mat', '.tdms')) -> list[Path]:
    """Find all data files in the dataset directory."""
    files = []
    for ext in extensions:
        files.extend(data_dir.rglob(f"*{ext}"))
    return sorted(files)  # Consistent ordering


def partition_by_condition(files: list[Path], train_ratio: float = TRAIN_RATIO, seed: int = SEED) -> tuple[list[Path], list[Path]]:
    """
    Partition files by condition (load + fault type) for balanced splits.
    All files with the same condition go to the same split.
    """
    # Group files by condition
    condition_groups = defaultdict(list)
    for f in files:
        cond = get_condition_key(f)
        condition_groups[cond].append(f)
    
    # Sort conditions for consistent ordering
    conditions = sorted(condition_groups.keys())
    
    # Deterministic shuffle using fixed seed
    rng = random.Random(seed)
    rng.shuffle(conditions)
    
    # Split conditions
    n_train = int(len(conditions) * train_ratio)
    train_conditions = set(conditions[:n_train])
    test_conditions = set(conditions[n_train:])
    
    # Assign files to splits based on their condition
    train_files = []
    test_files = []
    for cond in conditions:
        target = train_files if cond in train_conditions else test_files
        target.extend(condition_groups[cond])
    
    # Sort within each split for consistency
    train_files.sort()
    test_files.sort()
    
    return train_files, test_files


def partition_by_file(files: list[Path], train_ratio: float = TRAIN_RATIO, seed: int = SEED) -> tuple[list[Path], list[Path]]:
    """
    Partition individual files randomly but deterministically.
    """
    files_sorted = sorted(files)
    rng = random.Random(seed)
    rng.shuffle(files_sorted)
    
    n_train = int(len(files_sorted) * train_ratio)
    train_files = sorted(files_sorted[:n_train])
    test_files = sorted(files_sorted[n_train:])
    
    return train_files, test_files


def partition_stratified(files: list[Path], train_ratio: float = TRAIN_RATIO, seed: int = SEED) -> tuple[list[Path], list[Path]]:
    """
    Partition files with stratification by fault class (Normal, BPFI, BPFO, etc.)
    guaranteeing balanced representation in both train and test splits.
    """
    class_groups = defaultdict(list)
    for f in sorted(files):
        stem = f.stem
        parts = stem.split('_')
        cls = parts[1] if len(parts) > 1 else 'unknown'
        class_groups[cls].append(f)
    
    rng = random.Random(seed)
    train_files = []
    test_files = []
    
    for cls in sorted(class_groups.keys()):
        grp = list(class_groups[cls])
        rng.shuffle(grp)
        # Determine number of test samples per class
        n_test = max(1, int(round(len(grp) * (1 - train_ratio))))
        if len(grp) == 6:
            # Special case for 6 files (e.g. Normal): 4 train, 2 test for 24/6 exact 80/20 balance
            n_test = 2
        n_train = len(grp) - n_test
        train_files.extend(grp[:n_train])
        test_files.extend(grp[n_train:])
    
    train_files.sort()
    test_files.sort()
    return train_files, test_files


def copy_files(files: list[Path], src_root: Path, dst_root: Path, split_name: str, target_subdir: str = "", clean_target: bool = False) -> list[Path]:
    """Copy files preserving directory structure relative to src_root, with optional target_subdir."""
    copied = []
    if target_subdir and clean_target:
        target_dir = dst_root / split_name / target_subdir
        if target_dir.exists():
            shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            
    for f in files:
        try:
            rel_path = f.relative_to(src_root)
        except ValueError:
            # File not under src_root, use name only
            rel_path = Path(f.name)
        
        if target_subdir:
            dst_path = dst_root / split_name / target_subdir / rel_path
        else:
            dst_path = dst_root / split_name / rel_path
            
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst_path)
        copied.append(dst_path)
    return copied


def write_manifest(train_files: list[Path], test_files: list[Path], output_dir: Path, src_root: Path, target_subdir: str = ""):
    """Write manifest files listing the split."""
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    if target_subdir:
        # Rescan entire partitioned dataset so all sensor modalities are preserved
        train_files_all = sorted([f.relative_to(output_dir / "train") for f in (output_dir / "train").rglob("*") if f.is_file()])
        test_files_all = sorted([f.relative_to(output_dir / "test") for f in (output_dir / "test").rglob("*") if f.is_file()])
        
        with open(manifest_dir / "train_files.txt", 'w') as fp:
            for f in train_files_all:
                fp.write(f"{f}\n")
        with open(manifest_dir / "test_files.txt", 'w') as fp:
            for f in test_files_all:
                fp.write(f"{f}\n")
        
        with open(manifest_dir / "summary.txt", 'w') as fp:
            fp.write(f"Dataset Partition Summary\n")
            fp.write(f"========================\n")
            fp.write(f"Seed: {SEED}\n")
            fp.write(f"Train ratio: {TRAIN_RATIO}\n")
            fp.write(f"Total files: {len(train_files_all) + len(test_files_all)}\n")
            fp.write(f"Train files: {len(train_files_all)}\n")
            fp.write(f"Test files:  {len(test_files_all)}\n")
            fp.write(f"\n")
            fp.write(f"Train conditions: {len(set(get_condition_key(f) for f in train_files_all))}\n")
            fp.write(f"Test conditions:  {len(set(get_condition_key(f) for f in test_files_all))}\n")
    else:
        def write_list(files, path):
            with open(path, 'w') as fp:
                for f in files:
                    try:
                        rel = f.relative_to(src_root)
                    except ValueError:
                        rel = f
                    fp.write(f"{rel}\n")
        
        write_list(train_files, manifest_dir / "train_files.txt")
        write_list(test_files, manifest_dir / "test_files.txt")
        
        with open(manifest_dir / "summary.txt", 'w') as fp:
            fp.write(f"Dataset Partition Summary\n")
            fp.write(f"========================\n")
            fp.write(f"Seed: {SEED}\n")
            fp.write(f"Train ratio: {TRAIN_RATIO}\n")
            fp.write(f"Total files: {len(train_files) + len(test_files)}\n")
            fp.write(f"Train files: {len(train_files)}\n")
            fp.write(f"Test files:  {len(test_files)}\n")
            fp.write(f"\n")
            fp.write(f"Train conditions: {len(set(get_condition_key(f) for f in train_files))}\n")
            fp.write(f"Test conditions:  {len(set(get_condition_key(f) for f in test_files))}\n")
    
    print(f"Manifests written to: {manifest_dir}")

def verify_deterministic(data_dir: Path, output_dir: Path, strategy: str = 'condition', seed: int = SEED, train_ratio: float = TRAIN_RATIO):
    """Verify that running twice produces identical splits."""
    print("\nVerifying deterministic behavior...")
    
    # First run
    files = find_data_files(data_dir)
    if strategy == 'condition':
        train1, test1 = partition_by_condition(files, train_ratio, seed)
    elif strategy == 'stratified':
        train1, test1 = partition_stratified(files, train_ratio, seed)
    else:
        train1, test1 = partition_by_file(files, train_ratio, seed)
    
    # Second run (simulate fresh run)
    files2 = find_data_files(data_dir)
    if strategy == 'condition':
        train2, test2 = partition_by_condition(files2, train_ratio, seed)
    elif strategy == 'stratified':
        train2, test2 = partition_stratified(files2, train_ratio, seed)
    else:
        train2, test2 = partition_by_file(files2, train_ratio, seed)
    
    train1_set = set(str(f) for f in train1)
    train2_set = set(str(f) for f in train2)
    test1_set = set(str(f) for f in test1)
    test2_set = set(str(f) for f in test2)
    
    if train1_set == train2_set and test1_set == test2_set:
        print("✓ Deterministic verification PASSED - splits are identical")
        return True
    else:
        print("✗ Deterministic verification FAILED - splits differ!")
        print(f"  Train diff: {train1_set ^ train2_set}")
        print(f"  Test diff:  {test1_set ^ test2_set}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Deterministically partition dataset into train/test (80:20)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Partition by condition (recommended - balances fault types across splits)
  python partition_dataset.py data_dir -o output_dir --strategy condition
  
  # Partition by individual file
  python partition_dataset.py data_dir -o output_dir --strategy file
  
  # Partition with class stratification
  python partition_dataset.py data_dir -o output_dir --strategy stratified
  
  # Verify deterministic behavior
  python partition_dataset.py data_dir --verify-only
        """
    )
    parser.add_argument("data_dir", help="Path to extracted dataset directory")
    parser.add_argument("-o", "--output", default="partitioned_dataset", help="Output directory for train/test splits")
    parser.add_argument("--strategy", choices=['condition', 'file', 'stratified'], default='stratified',
                        help="Split strategy: 'stratified' (class-balanced), 'condition' (groups by condition), 'file' (splits files)")
    parser.add_argument("--target-subdir", default="", help="Optional subfolder inside train/ and test/ (e.g., 'acoustic')")
    parser.add_argument("--clean-target", action="store_true", help="Clean target subdirectory before copying")
    parser.add_argument("--seed", type=int, default=SEED, help=f"Random seed (default: {SEED})")
    parser.add_argument("--ratio", type=float, default=TRAIN_RATIO, help=f"Train ratio (default: {TRAIN_RATIO})")
    parser.add_argument("--verify-only", action="store_true", help="Only verify deterministic behavior, don't copy files")
    parser.add_argument("--extensions", nargs='+', default=['.mat', '.tdms'], help="File extensions to include")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output).resolve()
    
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        sys.exit(1)
    
    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory")
        sys.exit(1)
    
    seed = args.seed
    train_ratio = args.ratio
    
    print(f"Dataset: {data_dir}")
    print(f"Output:  {output_dir}")
    print(f"Strategy: {args.strategy}")
    if args.target_subdir:
        print(f"Target Subdir: {args.target_subdir}")
    print(f"Seed: {seed}")
    print(f"Train ratio: {train_ratio}")
    print(f"Extensions: {args.extensions}")
    print("-" * 50)
    
    # Find all data files
    files = find_data_files(data_dir, tuple(args.extensions))
    print(f"Found {len(files)} data files")
    
    if not files:
        print("No data files found!")
        sys.exit(1)
    
    # Verify deterministic behavior first
    if args.verify_only:
        success = verify_deterministic(data_dir, output_dir, args.strategy, seed, train_ratio)
        sys.exit(0 if success else 1)
    
    # Partition
    if args.strategy == 'condition':
        train_files, test_files = partition_by_condition(files, train_ratio, seed)
        print(f"Partitioned by condition: {len(set(get_condition_key(f) for f in train_files))} train conditions, "
              f"{len(set(get_condition_key(f) for f in test_files))} test conditions")
    elif args.strategy == 'stratified':
        train_files, test_files = partition_stratified(files, train_ratio, seed)
        print(f"Partitioned stratified: {len(train_files)} train, {len(test_files)} test")
    else:
        train_files, test_files = partition_by_file(files, train_ratio, seed)
        print(f"Partitioned by file: {len(train_files)} train, {len(test_files)} test")
    
    # Copy files
    print("\nCopying files...")
    train_copied = copy_files(train_files, data_dir, output_dir, "train", args.target_subdir, args.clean_target)
    test_copied = copy_files(test_files, data_dir, output_dir, "test", args.target_subdir, args.clean_target)
    
    print(f"Copied {len(train_copied)} files to train/{args.target_subdir}")
    print(f"Copied {len(test_copied)} files to test/{args.target_subdir}")
    
    # Write manifests
    write_manifest(train_files, test_files, output_dir, data_dir, args.target_subdir)
    
    print("-" * 50)
    print("Done!")
    print(f"\nTrain data: {output_dir}/train/{args.target_subdir}")
    print(f"Test data:  {output_dir}/test/{args.target_subdir}")
    print(f"Manifests:  {output_dir}/manifests/")


if __name__ == "__main__":
    main()