"""
Data loading module for multi-sensor fault diagnosis dataset.
Handles .mat (vibration, acoustic) and .tdms (current, temp) files.
"""
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import scipy.io as sio
from nptdms import TdmsFile

from config import (
    TRAIN_DIR, TEST_DIR, SENSOR_DIRS, CHANNEL_NAMES,
    parse_filename, get_class_index, FAULT_CLASSES,
    WINDOW_SIZES, STEP_SIZES, SAMPLE_RATES
)


def load_mat_file(filepath: Path) -> np.ndarray:
    """
    Load .mat file and return data array of shape (n_samples, n_channels).
    
    For vibration: (1536000, 4) - x_A, y_A, x_B, y_B in g
    For acoustic: (3072000, 1) - pressure in Pa
    """
    mat = sio.loadmat(filepath)
    if 'signal' in mat:
        data = mat['signal']
        if data.ndim == 2 and data.shape[0] == 1:
            data = data.T
    elif 'Signal' in mat:
        sig = mat['Signal'][0, 0]
        y_values = sig['y_values'][0, 0]
        data = y_values['values']  # This IS the data array
    else:
        candidates = [v for k, v in mat.items() if not k.startswith('__')]
        data = candidates[0]
        if data.ndim == 2 and data.shape[0] == 1:
            data = data.T
    
    # Ensure 2D: (n_samples, n_channels)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    return data.astype(np.float32)


# Expected channel names in order (first 3 are common to all files)
EXPECTED_TDMS_CHANNELS = [
    "cDAQ9185-1F486B5Mod1/ai0",  # temp_A
    "cDAQ9185-1F486B5Mod1/ai1",  # temp_B
    "cDAQ9185-1F486B5Mod2/ai0",  # U_phase
    "cDAQ9185-1F486B5Mod2/ai2",  # V_phase
    "cDAQ9185-1F486B5Mod2/ai3",  # W_phase
]

# Common channels present in ALL files (first 3)
COMMON_TDMS_CHANNELS = EXPECTED_TDMS_CHANNELS[:3]

SENSOR_TDMS_CHANNELS = {
    "temperature": EXPECTED_TDMS_CHANNELS[:2],      # temp_A, temp_B
    "current": [EXPECTED_TDMS_CHANNELS[2]],         # U_phase
    "current_temp": COMMON_TDMS_CHANNELS,           # temp_A, temp_B, U_phase
}


def load_tdms_file(filepath: Path, sensor_name: str = "current_temp") -> np.ndarray:
    """
    Load .tdms file and return data array of shape (n_samples, n_channels).
    
    Loads requested channels based on sensor_name:
    - temperature: temp_A, temp_B
    - current: U_phase
    - current_temp: temp_A, temp_B, U_phase
    """
    tdms = TdmsFile.read(filepath)
    
    # Find the Log group with data channels
    data_channels = []
    for group in tdms.groups():
        channels = group.channels()
        if channels:
            data_channels = channels
            break
    
    if not data_channels:
        raise ValueError(f"No data channels found in {filepath}")
    
    # Build channel name to data mapping
    channel_map = {}
    for channel in data_channels:
        data = channel[:]
        if data.size > 0:
            channel_map[channel.name] = data
    
    # Extract common channels in consistent order
    channel_data = []
    target_channels = SENSOR_TDMS_CHANNELS.get(sensor_name, COMMON_TDMS_CHANNELS)
    for ch_name in target_channels:
        if ch_name in channel_map:
            channel_data.append(channel_map[ch_name])
        else:
            raise ValueError(f"Required channel {ch_name} not found in {filepath}")
    
    # Check all channels have same length
    lengths = [len(d) for d in channel_data]
    if len(set(lengths)) > 1:
        # Truncate to shortest
        min_len = min(lengths)
        print(f"  Warning: Channel lengths differ {lengths}, truncating to {min_len}")
        channel_data = [d[:min_len] for d in channel_data]
    
    # Stack as (n_samples, n_channels)
    data = np.column_stack(channel_data)
    
    return data.astype(np.float32)


def segment_signal(signal: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    """
    Segment a signal into overlapping windows.
    
    Args:
        signal: (n_samples, n_channels)
        window_size: Number of samples per window
        step_size: Step between windows
        
    Returns:
        windows: (n_windows, window_size, n_channels)
    """
    n_samples = signal.shape[0]
    
    if n_samples < window_size:
        # Pad if too short
        pad_width = window_size - n_samples
        signal = np.pad(signal, ((0, pad_width), (0, 0)), mode='reflect')
        n_samples = window_size
    
    n_windows = (n_samples - window_size) // step_size + 1
    
    if n_windows <= 0:
        return np.empty((0, window_size, signal.shape[1]), dtype=signal.dtype)
    
    windows = np.lib.stride_tricks.sliding_window_view(
        signal, window_shape=(window_size,), axis=0
    )[::step_size]
    
    # sliding_window_view returns (n_windows, n_channels, window_size) for 2D input
    # Transpose to (n_windows, window_size, n_channels)
    windows = windows.transpose(0, 2, 1)
    
    return windows[:n_windows]


def load_sensor_data(
    sensor_name: str,
    split: str = "train",
    max_files_per_class: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Load all data for a given sensor and split.
    
    Args:
        sensor_name: "vibration", "acoustic", or "current_temp"
        split: "train" or "test"
        max_files_per_class: Optional limit for debugging
        
    Returns:
        windows: (n_windows, window_size, n_channels)
        labels: (n_windows,) class indices
        metadata: List of dicts with file info per window
    """
    if sensor_name not in SENSOR_DIRS:
        raise ValueError(f"Unknown sensor: {sensor_name}")
    
    base_dir = TRAIN_DIR if split == "train" else TEST_DIR
    sensor_dir = base_dir / SENSOR_DIRS[sensor_name]
    
    if not sensor_dir.exists():
        raise FileNotFoundError(f"Sensor directory not found: {sensor_dir}")
    
    # Get file loader
    if sensor_name in ["vibration", "acoustic"]:
        loader = load_mat_file
        ext = ".mat"
    else:
        loader = lambda fp, s=sensor_name: load_tdms_file(fp, s)
        ext = ".tdms"
    
    window_size = WINDOW_SIZES[sensor_name]
    step_size = STEP_SIZES[sensor_name]
    
    all_windows = []
    all_labels = []
    all_metadata = []
    
    # Group files by class for balanced loading
    files_by_class = {cls: [] for cls in FAULT_CLASSES}
    
    for fname in sorted(os.listdir(sensor_dir)):
        if not fname.endswith(ext):
            continue
        load, condition, severity = parse_filename(fname)
        if condition in FAULT_CLASSES:
            files_by_class[condition].append((fname, load, severity))
    
    # Apply max_files_per_class limit if specified
    if max_files_per_class:
        for cls in files_by_class:
            files_by_class[cls] = files_by_class[cls][:max_files_per_class]
    
    # Load files
    for condition, file_list in files_by_class.items():
        class_idx = get_class_index(condition)
        
        for fname, load, severity in file_list:
            filepath = sensor_dir / fname
            
            try:
                signal = loader(filepath)
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")
                continue
            
            # Segment into windows
            windows = segment_signal(signal, window_size, step_size)
            
            if len(windows) == 0:
                print(f"Warning: No windows extracted from {fname}")
                continue
            
            all_windows.append(windows)
            all_labels.extend([class_idx] * len(windows))
            
            for i in range(len(windows)):
                all_metadata.append({
                    "file": fname,
                    "load": load,
                    "condition": condition,
                    "severity": severity,
                    "window_idx": i,
                    "sensor": sensor_name,
                    "split": split,
                })
    
    if not all_windows:
        raise ValueError(f"No data loaded for {sensor_name} {split}")
    
    windows = np.concatenate(all_windows, axis=0)
    labels = np.array(all_labels, dtype=np.int32)
    
    print(f"Loaded {sensor_name} {split}: {windows.shape[0]} windows, "
          f"shape={windows.shape}, classes={np.unique(labels)}")
    
    return windows, labels, all_metadata


def load_all_sensors(
    split: str = "train",
    max_files_per_class: Optional[int] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[Dict]]]:
    """Load data for all sensors."""
    data = {}
    for sensor in SENSOR_DIRS.keys():
        try:
            data[sensor] = load_sensor_data(sensor, split, max_files_per_class)
        except Exception as e:
            print(f"Warning: Failed to load {sensor} {split}: {e}")
            data[sensor] = None
    return data


def get_class_distribution(labels: np.ndarray) -> Dict[str, int]:
    """Get count of samples per class."""
    unique, counts = np.unique(labels, return_counts=True)
    return {FAULT_CLASSES[u]: int(c) for u, c in zip(unique, counts)}


if __name__ == "__main__":
    # Quick test
    for sensor in ["vibration", "acoustic", "current_temp"]:
        try:
            windows, labels, meta = load_sensor_data(sensor, "train", max_files_per_class=1)
            print(f"{sensor}: windows={windows.shape}, labels={labels.shape}")
            print(f"  Class dist: {get_class_distribution(labels)}")
            print(f"  Sample metadata: {meta[0] if meta else 'None'}")
        except Exception as e:
            print(f"{sensor} ERROR: {e}")