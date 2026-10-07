"""
Feature extraction module for multi-domain features:
- Time domain
- Frequency domain  
- Time-frequency domain (wavelet packet)
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats
from scipy.fft import rfft, rfftfreq
import pywt
import warnings
warnings.filterwarnings('ignore')

from config import (
    CHANNEL_NAMES, SAMPLE_RATES, FREQ_BANDS,
    WAVELET, WP_MAX_LEVEL
)


# ============================================================
# Time Domain Features
# ============================================================
def time_domain_features(x: np.ndarray) -> Dict[str, float]:
    """Extract time domain features from a 1D signal."""
    x = np.asarray(x).flatten()
    n = len(x)
    
    if n == 0:
        return {f: 0.0 for f in [
            'mean', 'std', 'var', 'rms', 'skewness', 'kurtosis',
            'peak_to_peak', 'crest_factor', 'shape_factor', 'impulse_factor'
        ]}
    
    mean_val = np.mean(x)
    std_val = np.std(x)
    var_val = np.var(x)
    rms_val = np.sqrt(np.mean(x**2))
    
    # Peak-to-peak
    ptp_val = np.max(x) - np.min(x)
    
    # Skewness and kurtosis
    skew_val = stats.skew(x)
    kurt_val = stats.kurtosis(x)
    
    # Crest factor = peak / rms
    peak_val = np.max(np.abs(x))
    crest_factor = peak_val / rms_val if rms_val > 0 else 0
    
    # Shape factor = rms / mean(abs)
    mean_abs = np.mean(np.abs(x))
    shape_factor = rms_val / mean_abs if mean_abs > 0 else 0
    
    # Impulse factor = peak / mean(abs)
    impulse_factor = peak_val / mean_abs if mean_abs > 0 else 0
    
    return {
        'mean': float(mean_val),
        'std': float(std_val),
        'var': float(var_val),
        'rms': float(rms_val),
        'skewness': float(skew_val),
        'kurtosis': float(kurt_val),
        'peak_to_peak': float(ptp_val),
        'crest_factor': float(crest_factor),
        'shape_factor': float(shape_factor),
        'impulse_factor': float(impulse_factor),
    }


# ============================================================
# Frequency Domain Features
# ============================================================
def frequency_domain_features(x: np.ndarray, fs: float) -> Dict[str, float]:
    """Extract frequency domain features from a 1D signal."""
    x = np.asarray(x).flatten()
    n = len(x)
    
    if n == 0:
        return {f: 0.0 for f in [
            'dominant_freq', 'spectral_centroid', 'spectral_entropy',
            'band_energy_0', 'band_energy_1', 'band_energy_2', 'band_energy_3',
            'total_power'
        ]}
    
    # Compute FFT
    fft_vals = rfft(x)
    freqs = rfftfreq(n, 1/fs)
    magnitude = np.abs(fft_vals)
    power = magnitude**2
    
    # Total power
    total_power = np.sum(power)
    
    if total_power == 0:
        return {
            'dominant_freq': 0.0,
            'spectral_centroid': 0.0,
            'spectral_entropy': 0.0,
            'band_energy_0': 0.0,
            'band_energy_1': 0.0,
            'band_energy_2': 0.0,
            'band_energy_3': 0.0,
            'total_power': 0.0,
        }
    
    # Normalized power spectrum
    psd = power / total_power
    
    # Dominant frequency
    dominant_idx = np.argmax(magnitude)
    dominant_freq = float(freqs[dominant_idx])
    
    # Spectral centroid
    spectral_centroid = float(np.sum(freqs * psd))
    
    # Spectral entropy
    # Avoid log(0)
    psd_nz = psd[psd > 0]
    spectral_entropy = float(-np.sum(psd_nz * np.log2(psd_nz)))
    
    # Band energy ratios
    band_energies = {}
    bands = FREQ_BANDS.get('default', [(0, fs/4), (fs/4, fs/2), (fs/2, 3*fs/4), (3*fs/4, fs/2)])
    
    for i, (f_low, f_high) in enumerate(bands):
        mask = (freqs >= f_low) & (freqs < f_high)
        band_energy = np.sum(power[mask]) / total_power if total_power > 0 else 0
        band_energies[f'band_energy_{i}'] = float(band_energy)
    
    return {
        'dominant_freq': dominant_freq,
        'spectral_centroid': spectral_centroid,
        'spectral_entropy': spectral_entropy,
        'total_power': float(total_power),
        **band_energies,
    }


# ============================================================
# Time-Frequency Domain Features (Wavelet Packet)
# ============================================================
def wavelet_packet_features(x: np.ndarray, wavelet: str = WAVELET, max_level: int = WP_MAX_LEVEL) -> Dict[str, float]:
    """Extract wavelet packet energy features."""
    x = np.asarray(x).flatten()
    n = len(x)
    
    n_subbands = 2**max_level
    empty_result = {f'wp_energy_{i}': 0.0 for i in range(n_subbands)}
    empty_result['wp_entropy'] = 0.0
    
    if n == 0:
        return empty_result
    
    try:
        # Wavelet packet decomposition
        wp = pywt.WaveletPacket(data=x, wavelet=wavelet, mode='symmetric', maxlevel=max_level)
        
        # Get nodes at max level
        nodes = wp.get_level(max_level, 'freq')
        
        energies = []
        for node in nodes:
            coeffs = node.data
            energy = np.sum(coeffs**2)
            energies.append(energy)
        
        energies = np.array(energies)
        total_energy = np.sum(energies)
        
        if total_energy == 0:
            return empty_result
        
        # Normalized energies
        norm_energies = energies / total_energy
        
        # Wavelet packet entropy
        nz = norm_energies[norm_energies > 0]
        wp_entropy = float(-np.sum(nz * np.log2(nz)))
        
        result = {f'wp_energy_{i}': float(e) for i, e in enumerate(norm_energies)}
        result['wp_entropy'] = wp_entropy
        
        return result
        
    except Exception as e:
        return empty_result


# ============================================================
# Combined Feature Extraction per Window
# ============================================================
def extract_window_features(
    window: np.ndarray,
    sensor_name: str
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract all features from a single window.
    
    Args:
        window: (window_size, n_channels)
        sensor_name: "vibration", "acoustic", or "current_temp"
        
    Returns:
        features: 1D array of features
        feature_names: List of feature names
    """
    fs = SAMPLE_RATES[sensor_name]
    n_channels = window.shape[1]
    channel_names = CHANNEL_NAMES[sensor_name]
    
    all_features = []
    all_names = []
    
    # Per-channel features
    for ch_idx in range(n_channels):
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"ch{ch_idx}"
        signal = window[:, ch_idx]
        
        # Time domain
        td_feats = time_domain_features(signal)
        for name, val in td_feats.items():
            all_features.append(val)
            all_names.append(f"{ch_name}_{name}")
        
        # Frequency domain
        fd_feats = frequency_domain_features(signal, fs)
        for name, val in fd_feats.items():
            all_features.append(val)
            all_names.append(f"{ch_name}_{name}")
        
        # Wavelet packet
        wp_feats = wavelet_packet_features(signal)
        for name, val in wp_feats.items():
            all_features.append(val)
            all_names.append(f"{ch_name}_{name}")
    
    # Cross-channel features for multi-channel sensors
    if n_channels > 1:
        # Correlation between channels
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                ch_i = channel_names[i] if i < len(channel_names) else f"ch{i}"
                ch_j = channel_names[j] if j < len(channel_names) else f"ch{j}"
                corr = np.corrcoef(window[:, i], window[:, j])[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                all_features.append(float(corr))
                all_names.append(f"{ch_i}_{ch_j}_corr")
        
        # Sum/difference for vibration (housing A vs B)
        if sensor_name == "vibration" and n_channels == 4:
            # x_A vs x_B, y_A vs y_B
            for axis_idx, (i, j) in enumerate([(0, 2), (1, 3)]):
                axis_name = ['x', 'y'][axis_idx]
                sum_sig = window[:, i] + window[:, j]
                diff_sig = window[:, i] - window[:, j]
                
                for sig, suffix in [(sum_sig, 'sum'), (diff_sig, 'diff')]:
                    td = time_domain_features(sig)
                    for name, val in td.items():
                        all_features.append(val)
                        all_names.append(f"{axis_name}_{suffix}_{name}")
    
    return np.array(all_features, dtype=np.float32), all_names


def extract_features_from_windows(
    windows: np.ndarray,
    sensor_name: str,
    n_jobs: int = -1
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract features from all windows.
    
    Args:
        windows: (n_windows, window_size, n_channels)
        sensor_name: Sensor name
        n_jobs: Number of parallel jobs (not used yet, for future)
        
    Returns:
        feature_matrix: (n_windows, n_features)
        feature_names: List of feature names
    """
    n_windows = windows.shape[0]
    
    if n_windows == 0:
        return np.empty((0, 0), dtype=np.float32), []
    
    # Extract features from first window to get feature names
    first_feats, feature_names = extract_window_features(windows[0], sensor_name)
    n_features = len(first_feats)
    
    if n_jobs != 1 and n_windows > 50:
        try:
            from joblib import Parallel, delayed
            import os
            n_workers = os.cpu_count() or 4
            n_chunks = min(n_windows, max(4, n_workers * 2))
            chunks = np.array_split(windows, n_chunks)
            
            def _proc_chunk(chunk):
                return [extract_window_features(w, sensor_name)[0] for w in chunk]
            
            res = Parallel(n_jobs=n_jobs)(delayed(_proc_chunk)(c) for c in chunks)
            feature_matrix = np.vstack(res)
        except Exception:
            feature_matrix = np.zeros((n_windows, n_features), dtype=np.float32)
            feature_matrix[0] = first_feats
            for i in range(1, n_windows):
                feats, _ = extract_window_features(windows[i], sensor_name)
                feature_matrix[i] = feats
    else:
        feature_matrix = np.zeros((n_windows, n_features), dtype=np.float32)
        feature_matrix[0] = first_feats
        for i in range(1, n_windows):
            feats, _ = extract_window_features(windows[i], sensor_name)
            feature_matrix[i] = feats
    # Handle NaN/Inf
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Extracted {n_features} features from {n_windows} windows for {sensor_name}")
    
    return feature_matrix, feature_names


if __name__ == "__main__":
    # Quick test with synthetic data
    from config import WINDOW_SIZES
    
    for sensor in ["vibration", "acoustic", "current_temp"]:
        ws = WINDOW_SIZES[sensor]
        n_ch = len(CHANNEL_NAMES[sensor])
        
        # Synthetic window
        window = np.random.randn(ws, n_ch).astype(np.float32)
        
        feats, names = extract_window_features(window, sensor)
        print(f"{sensor}: {len(feats)} features")
        print(f"  First 10: {names[:10]}")
        print(f"  Last 10: {names[-10:]}")