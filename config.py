"""
Configuration for Random Forest fault diagnosis pipeline.
"""
from pathlib import Path

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "partitioned_dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = BASE_DIR / "rf_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sensor directories
SENSOR_DIRS = {
    "vibration": "vibration",
    "acoustic": "acoustic",
    "current": "current,temp",
    "temperature": "current,temp",
    "current_temp": "current,temp",
}

# ============================================================
# Signal parameters (derived from data inspection)
# ============================================================
SAMPLE_RATES = {
    "vibration": 25600,    # Hz
    "acoustic": 51200,     # Hz
    "current": 25600,      # Hz
    "temperature": 25600,  # Hz
    "current_temp": 25600, # Hz (approximately)
}

CHANNEL_NAMES = {
    "vibration": ["x_housing_A", "y_housing_A", "x_housing_B", "y_housing_B"],
    "acoustic": ["pressure"],
    "current": ["U_phase"],
    "temperature": ["temp_A", "temp_B"],
    "current_temp": ["temp_A", "temp_B", "U_phase"],
}

# Units (for reference)
CHANNEL_UNITS = {
    "vibration": ["g", "g", "g", "g"],
    "acoustic": ["Pa"],
    "current": ["A"],
    "temperature": ["°C", "°C"],
    "current_temp": ["°C", "°C", "A"],
}

# ============================================================
# Segmentation parameters
# ============================================================
WINDOW_SECONDS = 1.0          # Window length in seconds
OVERLAP_RATIO = 0.5           # 50% overlap
MIN_WINDOWS_PER_FILE = 1      # Minimum windows to extract from a file

# Derived window sizes (samples)
WINDOW_SIZES = {
    sensor: int(WINDOW_SECONDS * SAMPLE_RATES[sensor])
    for sensor in SAMPLE_RATES
}
STEP_SIZES = {
    sensor: int(WINDOW_SIZES[sensor] * (1 - OVERLAP_RATIO))
    for sensor in SAMPLE_RATES
}

# ============================================================
# Feature extraction parameters
# ============================================================
# Wavelet packet decomposition
WAVELET = "db4"
WP_MAX_LEVEL = 3  # Gives 2^3 = 8 sub-bands

# Frequency bands for band-energy ratios (Hz)
FREQ_BANDS = {
    "vibration": [(0, 1000), (1000, 3000), (3000, 6000), (6000, 12800)],
    "acoustic": [(0, 2000), (2000, 6000), (6000, 12000), (12000, 25600)],
    "current": [(0, 100), (100, 500), (500, 2000), (2000, 12800)],
    "temperature": [(0, 100), (100, 500), (500, 2000), (2000, 12800)],
    "current_temp": [(0, 100), (100, 500), (500, 2000), (2000, 12800)],
}

# ============================================================
# Preprocessing
# ============================================================
SCALER_TYPE = "standard"  # "standard" or "robust"

# ============================================================
# Random Forest hyperparameter configuration
# ============================================================
RF_DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "criterion": "gini",
    "class_weight": "balanced",
    "bootstrap": True,
    "oob_score": True,
    "n_jobs": -1,
    "random_state": 42,
}

# Sensor-specific default parameters
RF_SENSOR_PARAMS = {
    "vibration": {},
    "acoustic": {},
    "current": {},
    "temperature": {"max_depth": 5, "max_features": None},
}

# Parameter grid for GridSearchCV
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
}

# Parameter distribution for RandomizedSearchCV
RF_PARAM_DIST = {
    "n_estimators": [50, 100, 150, 200, 300, 400],
    "max_depth": [None, 5, 10, 15, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "criterion": ["gini", "entropy"],
}

# Particle Swarm Optimization (PSO) for Random Forest
RF_PSO_CONFIG = {
    "n_particles": 20,
    "n_iterations": 30,
    "c1": 1.5,
    "c2": 1.5,
    "w": 0.7,
    "bounds": {
        "n_estimators": (50, 300),
        "max_depth": (5, 35),
        "min_samples_split": (2, 10),
        "min_samples_leaf": (1, 5),
    },
}

# Cross-validation
CV_FOLDS = 5
CV_SCORING = "f1_macro"
# ============================================================
# Evaluation
# ============================================================
N_PERMUTATION_REPEATS = 10
TOP_K_FEATURES = 20
RANDOM_STATE = 42

# ============================================================
# Fault class mapping (handles typo in filenames)
# ============================================================
FAULT_CLASSES = ["Normal", "BPFI", "BPFO", "Misalign", "Unbalance"]

def parse_filename(fname: str):
    """Parse filename into (load, condition, severity).
    Handles 'Unbalalnce' typo in 2Nm vibration files.
    """
    fname = fname.replace("Unbalalnce", "Unbalance")
    stem = fname.replace(".mat", "").replace(".tdms", "")
    parts = stem.split("_")
    load = parts[0]           # e.g., "0Nm", "2Nm", "4Nm"
    condition = parts[1]      # e.g., "Normal", "BPFI", "BPFO", "Misalign", "Unbalance"
    severity = "_".join(parts[2:]) if len(parts) > 2 else "none"
    return load, condition, severity

def get_class_index(condition: str) -> int:
    """Map condition string to class index."""
    return FAULT_CLASSES.index(condition)