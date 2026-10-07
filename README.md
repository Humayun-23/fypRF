# Dataset Setup Guide

This guide provides **exact, fixed commands** to unzip and partition the dataset. All teammates should run these commands verbatim to ensure **identical, reproducible results** across all machines.

---

## Prerequisites

- Python 3.7+
- Standard library only (no external packages required)

---

## Step 1: Unzip the Dataset (Recursive)

### Command

```bash
cd /path/to/your/project/root
python3 unzip_recursive.py
```

### What This Does

- Finds all `.zip` files recursively (including nested zips)
- Extracts each zip into a folder named after the zip file (without `.zip`)
- **Removes each zip file after successful extraction** (saves disk space)
- Protects against zip-slip attacks
- Tracks processed zips to avoid infinite loops

### Expected Output

```
Starting recursive unzip in: /path/to/your/project/root
--------------------------------------------------
Extracting: .../Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating Machine Under Varying Load Conditions for Fault Diagnosis.zip -> ...
  Successfully extracted to ...
  Removed zip file: ...
Extracting: .../current,temp.zip -> ...
  Successfully extracted to ...
  Removed zip file: ...
Extracting: .../acoustic.zip -> ...
  Successfully extracted to ...
  Removed zip file: ...
Extracting: .../vibration.zip -> ...
  Successfully extracted to ...
  Removed zip file: ...
--------------------------------------------------
Done!
```

### Verification

After running, verify no zip files remain:

```bash
find . -name "*.zip"
# Should return NO output (empty)
```

And verify extracted structure:

```bash
find . -type f \( -name "*.mat" -o -name "*.tdms" \) | wc -l
# Should output: 95
```

---

## Step 2: Partition into Train/Test (80:20)

### Command

```bash
cd /path/to/your/project/root
python3 partition_dataset.py "Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating Machine Under Varying Load Conditions for Fault Diagnosis/Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating Machine Under Varying Load Conditions for Fault Diagnosis" -o partitioned_dataset --strategy condition
```

### Fixed Parameters (DO NOT CHANGE)

| Parameter | Value | Reason |
|-----------|-------|--------|
| `--strategy` | `condition` | Groups by `{load}_{fault}_{severity}` for balanced fault types |
| `--seed` | `42` (default) | Fixed seed ensures identical split everywhere |
| `--ratio` | `0.8` (default) | 80% train, 20% test |
| `-o` | `partitioned_dataset` | Output folder name |

### What This Does

- Finds all `.mat` and `.tdms` files in the extracted dataset
- Groups files by **condition** (e.g., `0Nm_BPFO_03`, `2Nm_Misalign_05`)
- Shuffles conditions deterministically using seed `42`
- Assigns first 80% of conditions to **train**, remaining 20% to **test**
- Copies files preserving directory structure (`train/vibration/`, `train/acoustic/`, etc.)
- Generates manifests in `partitioned_dataset/manifests/`

### Expected Output

```
Dataset: /path/to/.../Fault Diagnosis/...
Output:  /path/to/.../partitioned_dataset
Strategy: condition
Seed: 42
Train ratio: 0.8
Extensions: ['.mat', '.tdms']
--------------------------------------------------
Found 95 data files
Partitioned by condition: 40 train conditions, 10 test conditions

Copying files...
Copied 75 files to train/
Copied 20 files to test/
Manifests written to: /path/to/.../partitioned_dataset/manifests
--------------------------------------------------
Done!

Train data: /path/to/.../partitioned_dataset/train/
Test data:  /path/to/.../partitioned_dataset/test/
Manifests:  /path/to/.../partitioned_dataset/manifests/
```

### Verification

Verify the split matches exactly:

```bash
# Check file counts
find partitioned_dataset/train -type f | wc -l  # Should be 75
find partitioned_dataset/test -type f | wc -l   # Should be 20

# Verify manifests exist
cat partitioned_dataset/manifests/summary.txt
```

Expected `summary.txt` content:

```
Dataset Partition Summary
========================
Seed: 42
Train ratio: 0.8
Total files: 95
Train files: 75
Test files:  20

Train conditions: 40
Test conditions:  10
```

---

## Step 3: Verify Deterministic Behavior (Optional but Recommended)

Run this to confirm the split is identical every time:

```bash
python3 partition_dataset.py "Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating Machine Under Varying Load Conditions for Fault Diagnosis/Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating Machine Under Varying Load Conditions for Fault Diagnosis" --verify-only
```

Expected output:

```
Verifying deterministic behavior...
✓ Deterministic verification PASSED - splits are identical
```

---

## Final Directory Structure

After both steps, your project root should look like:

```
project-root/
├── unzip_recursive.py              # Unzip script
├── partition_dataset.py            # Partition script
├── README.md                       # This file
└── partitioned_dataset/            # OUTPUT - Ready for training
    ├── train/
    │   ├── vibration/              # 50 .mat files (40 conditions)
    │   ├── acoustic/               # 5 .mat files
    │   └── current,temp/           # 20 .tdms files
    ├── test/
    │   ├── vibration/              # ~10 .mat files (10 conditions)
    │   ├── acoustic/               # ~1 .mat files
    │   └── current,temp/           # ~9 .tdms files
    └── manifests/
        ├── train_files.txt         # List of all train files (relative paths)
        ├── test_files.txt          # List of all test files (relative paths)
        └── summary.txt             # Split statistics
```

---

## Train/Test Conditions Breakdown

### Train Conditions (40) - 75 files
```
0Nm_BPFI_03, 0Nm_BPFI_30, 0Nm_BPFO_03, 0Nm_BPFO_10, 0Nm_Normal,
0Nm_Unbalance_0583mg, 0Nm_Unbalance_1169mg, 0Nm_Unbalance_1751mg, 0Nm_Unbalance_2239mg,
2Nm_BPFI_10, 2Nm_BPFO_03, 2Nm_BPFO_10, 2Nm_BPFO_30, 2Nm_Misalign_01, 2Nm_Misalign_03,
2Nm_Misalign_05, 2Nm_Normal, 2Nm_Unbalalnce_0583mg, 2Nm_Unbalalnce_1169mg,
2Nm_Unbalalnce_1751mg, 2Nm_Unbalalnce_2239mg, 2Nm_Unbalalnce_3318mg,
2Nm_Unbalance_0583mg, 2Nm_Unbalance_1169mg, 2Nm_Unbalance_1751mg, 2Nm_Unbalance_2239mg,
4Nm_BPFI_03, 4Nm_BPFI_10, 4Nm_BPFI_30, 4Nm_BPFO_03, 4Nm_BPFO_10,
4Nm_Misalign_01, 4Nm_Misalign_03, 4Nm_Misalign_05, 4Nm_Normal,
4Nm_Unbalance_0583mg, 4Nm_Unbalance_1169mg, 4Nm_Unbalance_1751mg,
4Nm_Unbalance_2239mg, 4Nm_Unbalance_3318mg
```

### Test Conditions (10) - 20 files
```
0Nm_BPFI_10, 0Nm_BPFO_30, 0Nm_Misalign_01, 0Nm_Misalign_03, 0Nm_Misalign_05,
0Nm_Unbalance_3318mg, 2Nm_BPFI_03, 2Nm_BPFI_30, 2Nm_Unbalance_3318mg, 4Nm_BPFO_30
```

---

## Troubleshooting

### "No zip files found"
- Ensure you're in the correct directory (where the main `.zip` file sits)
- Run `ls *.zip` to confirm the zip file exists

### "Data directory does not exist"
- The path in Step 2 must match the extracted folder name exactly
- Use tab completion or copy the exact path from `ls`

### Permission denied
```bash
chmod +x unzip_recursive.py partition_dataset.py
```

### Different file counts
- Re-run Step 1 (unzip) - may have missed nested zips
- Verify `find . -name "*.zip"` returns nothing

---

## For Teammates: Quick Copy-Paste

**Run these three commands in order (adjust the first `cd` path):**

```bash
# 1. Navigate to project root (CHANGE THIS PATH)
cd /home/yourusername/path/to/project/root

# 2. Unzip everything recursively
python3 unzip_recursive.py

# 3. Partition into train/test (80:20, deterministic)
python3 partition_dataset.py "Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating Machine Under Varying Load Conditions for Fault Diagnosis/Vibration, Acoustic, Temperature, and Motor Current Dataset of Rotating Machine Under Varying Load Conditions for Fault Diagnosis" -o partitioned_dataset --strategy condition
```

That's it. The `partitioned_dataset/` folder is now ready for training with **guaranteed identical splits** across all team members' machines.