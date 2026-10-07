#!/usr/bin/env python3
"""
Main pipeline for Random Forest fault diagnosis.

Runs separate Random Forest classifiers for each sensor:
- vibration (4 channels)
- acoustic (1 channel) 
- current_temp (5 channels: 2 temp + 3 current)

Handles acoustic data limitation (only 3 classes) by training on available classes.
"""
import os
import sys
import json
import time
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Set random seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)

# Local imports
from config import (
    OUTPUT_DIR, FAULT_CLASSES, RANDOM_STATE, SENSOR_DIRS,
    WINDOW_SIZES, STEP_SIZES, SAMPLE_RATES, RF_DEFAULT_PARAMS,
    RF_SENSOR_PARAMS
)
from data_loader import load_sensor_data, load_all_sensors, get_class_distribution
from feature_extractor import extract_features_from_windows
from random_forest import (
    optimize_grid, optimize_randomized, optimize_pso,
    train_final_model, compute_oob_curve, create_rf
)
from evaluation import run_full_evaluation

warnings.filterwarnings('ignore')


def run_sensor_pipeline(
    sensor_name: str,
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    train_metadata: List[Dict],
    test_metadata: List[Dict],
    output_dir: Path,
    tuning_method: str = "grid",
    pso_particles: int = 20,
    pso_iterations: int = 30,
    n_estimators_override: Optional[int] = None,
    max_depth_override: Optional[int] = None
) -> Dict:
    """
    Run complete Random Forest pipeline for a single sensor.
    
    Returns:
        Dictionary with results and artifacts
    """
    print(f"\n{'#'*70}")
    print(f"# PIPELINE: {sensor_name.upper()} (RANDOM FOREST)")
    print(f"{'#'*70}")
    
    sensor_output_dir = output_dir / sensor_name
    sensor_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Feature extraction
    print(f"\n[1/5] Extracting features for {sensor_name}...")
    start = time.time()
    
    X_train, feature_names = extract_features_from_windows(train_windows, sensor_name)
    X_test, _ = extract_features_from_windows(test_windows, sensor_name)
    
    print(f"    Train features: {X_train.shape}")
    print(f"    Test features:  {X_test.shape}")
    print(f"    Time: {time.time() - start:.1f}s")
    
    # Check for classes present in train and test
    train_classes = np.unique(train_labels)
    test_classes = np.unique(test_labels)
    print(f"    Train classes: {[FAULT_CLASSES[c] for c in train_classes]}")
    print(f"    Test classes:  {[FAULT_CLASSES[c] for c in test_classes]}")
    
    # For acoustic: only train on classes that appear in BOTH train and test
    common_classes = np.intersect1d(train_classes, test_classes)
    if len(common_classes) < len(train_classes):
        print(f"    WARNING: Some classes only in train: {[FAULT_CLASSES[c] for c in np.setdiff1d(train_classes, common_classes)]}")
    if len(common_classes) < len(test_classes):
        print(f"    WARNING: Some classes only in test: {[FAULT_CLASSES[c] for c in np.setdiff1d(test_classes, common_classes)]}")
    
    # Filter to common classes if needed (for acoustic)
    if len(common_classes) > 0 and len(common_classes) < len(FAULT_CLASSES):
        print(f"    Restricting to {len(common_classes)} common classes: {[FAULT_CLASSES[c] for c in common_classes]}")
        train_mask = np.isin(train_labels, common_classes)
        test_mask = np.isin(test_labels, common_classes)
        X_train = X_train[train_mask]
        train_labels = train_labels[train_mask]
        X_test = X_test[test_mask]
        test_labels = test_labels[test_mask]
        # Remap labels to contiguous indices
        label_map = {old: new for new, old in enumerate(sorted(common_classes))}
        train_labels = np.array([label_map[l] for l in train_labels])
        test_labels = np.array([label_map[l] for l in test_labels])
        class_names = [FAULT_CLASSES[c] for c in sorted(common_classes)]
    else:
        class_names = FAULT_CLASSES
    
    print(f"    Final train: {X_train.shape[0]} samples, {len(np.unique(train_labels))} classes")
    print(f"    Final test:  {X_test.shape[0]} samples, {len(np.unique(test_labels))} classes")
    
    # Step 2: Hyperparameter Tuning
    print(f"\n[2/5] Hyperparameter optimization ({tuning_method})...")
    convergence_curve = []
    
    if tuning_method == "grid":
        start = time.time()
        best_params, convergence_curve = optimize_grid(X_train, train_labels, sensor_name)
        print(f"    Tuning time: {time.time() - start:.1f}s")
    elif tuning_method == "randomized":
        start = time.time()
        best_params, convergence_curve = optimize_randomized(X_train, train_labels, sensor_name)
        print(f"    Tuning time: {time.time() - start:.1f}s")
    elif tuning_method == "pso":
        start = time.time()
        best_params, convergence_curve = optimize_pso(
            X_train, train_labels, sensor_name,
            n_particles=pso_particles,
            n_iterations=pso_iterations,
            verbose=True
        )
        print(f"    Tuning time: {time.time() - start:.1f}s")
    else:
        best_params = dict(RF_DEFAULT_PARAMS)
        if sensor_name in RF_SENSOR_PARAMS:
            best_params.update(RF_SENSOR_PARAMS[sensor_name])
        print(f"    Using default parameters: {best_params}")
    
    # Apply manual overrides if given
    if n_estimators_override is not None:
        best_params["n_estimators"] = n_estimators_override
    if max_depth_override is not None:
        best_params["max_depth"] = max_depth_override
    
    # Step 3: Train final model
    print(f"\n[3/5] Training final Random Forest model...")
    start = time.time()
    pipeline, rf_model = train_final_model(
        X_train, train_labels, best_params, sensor_name,
        save_path=sensor_output_dir
    )
    print(f"    Training time: {time.time() - start:.1f}s")
    
    # Compute OOB error progression for ensemble diagnostics
    oob_curve = []
    try:
        max_trees = min(best_params.get("n_estimators", 200), 200)
        _, oob_curve = compute_oob_curve(X_train, train_labels, best_params, min_trees=10, max_trees=max_trees, step=10)
    except Exception as e:
        print(f"    Note: OOB curve computation skipped: {e}")
    
    # Step 4: Evaluate
    print(f"\n[4/5] Evaluating on test set...")
    start = time.time()
    results = run_full_evaluation(
        pipeline,
        X_test,
        test_labels,
        feature_names,
        sensor_name,
        convergence_curve=convergence_curve,
        estimator_curve=oob_curve,
        y_train=train_labels,
        output_dir=output_dir,
        class_names=class_names
    )
    print(f"    Evaluation time: {time.time() - start:.1f}s")
    
    # Step 5: Save additional artifacts & metadata
    print(f"\n[5/5] Saving metadata & feature lists...")
    with open(sensor_output_dir / f"{sensor_name}_feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    
    metadata = {
        "model_type": "RandomForest",
        "sensor": sensor_name,
        "window_size": int(WINDOW_SIZES[sensor_name]),
        "step_size": int(STEP_SIZES[sensor_name]),
        "sample_rate": SAMPLE_RATES[sensor_name],
        "n_features": len(feature_names),
        "n_train_windows": int(len(train_labels)),
        "n_test_windows": int(len(test_labels)),
        "train_class_distribution": get_class_distribution(train_labels),
        "test_class_distribution": get_class_distribution(test_labels),
        "classes_used": class_names,
        "best_params": best_params,
        "tuning_method": tuning_method,
        "tuning_convergence": convergence_curve,
        "oob_score": results.get("oob_score"),
    }
    
    def convert_for_json(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj
    
    with open(sensor_output_dir / f"{sensor_name}_metadata.json", "w") as f:
        json.dump(convert_for_json(metadata), f, indent=2)
    
    print(f"\n{'#'*70}")
    print(f"# {sensor_name.upper()} COMPLETE")
    print(f"# Accuracy: {results['accuracy']:.4f}, Macro F1: {results['macro_f1']:.4f}")
    if results.get("oob_score") is not None:
        print(f"# OOB Score: {results['oob_score']:.4f}")
    print(f"{'#'*70}\n")
    
    return {
        "sensor": sensor_name,
        "pipeline": pipeline,
        "rf_model": rf_model,
        "best_params": best_params,
        "feature_names": feature_names,
        "results": results,
        "convergence_curve": convergence_curve,
        "class_names": class_names,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Random Forest Fault Diagnosis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--sensors", nargs="+", 
                        choices=["vibration", "acoustic", "current", "temperature", "current_temp", "all"],
                        default=["vibration", "acoustic", "current", "temperature"],
                        help="Sensors to process")
    parser.add_argument("--tune", choices=["grid", "randomized", "pso", "none"],
                        default="none",
                        help="Hyperparameter tuning method")
    parser.add_argument("--pso-particles", type=int, default=20,
                        help="Number of PSO particles (if --tune pso)")
    parser.add_argument("--pso-iterations", type=int, default=30,
                        help="Number of PSO iterations (if --tune pso)")
    parser.add_argument("--n-estimators", type=int, default=None,
                        help="Override number of trees")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Override maximum tree depth")
    parser.add_argument("--max-files-per-class", type=int, default=None,
                        help="Limit files per class (for rapid testing)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Determine sensors to process
    if "all" in args.sensors:
        sensors = ["vibration", "acoustic", "current", "temperature"]
    else:
        sensors = args.sensors
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"RANDOM FOREST FAULT DIAGNOSIS PIPELINE")
    print(f"{'='*70}")
    print(f"Sensors:       {sensors}")
    print(f"Tuning method: {args.tune}")
    print(f"Output:        {output_dir}")
    print(f"Random seed:   {RANDOM_STATE}")
    print(f"{'='*70}\n")
    
    # Load data for all sensors
    print("Loading data...")
    all_data = {}
    for sensor in sensors:
        print(f"  Loading {sensor}...")
        try:
            train_data = load_sensor_data(sensor, "train", args.max_files_per_class)
            test_data = load_sensor_data(sensor, "test", args.max_files_per_class)
            all_data[sensor] = {
                "train_windows": train_data[0],
                "train_labels": train_data[1],
                "train_metadata": train_data[2],
                "test_windows": test_data[0],
                "test_labels": test_data[1],
                "test_metadata": test_data[2],
            }
            print(f"    Train: {train_data[0].shape[0]} windows")
            print(f"    Test:  {test_data[0].shape[0]} windows")
        except Exception as e:
            print(f"    ERROR loading {sensor}: {e}")
            all_data[sensor] = None
    
    # Run pipeline for each sensor
    all_results = {}
    for sensor in sensors:
        if all_data[sensor] is None:
            print(f"\nSkipping {sensor} - no data loaded")
            continue
        
        try:
            result = run_sensor_pipeline(
                sensor,
                all_data[sensor]["train_windows"],
                all_data[sensor]["train_labels"],
                all_data[sensor]["test_windows"],
                all_data[sensor]["test_labels"],
                all_data[sensor]["train_metadata"],
                all_data[sensor]["test_metadata"],
                output_dir,
                tuning_method=args.tune,
                pso_particles=args.pso_particles,
                pso_iterations=args.pso_iterations,
                n_estimators_override=args.n_estimators,
                max_depth_override=args.max_depth
            )
            all_results[sensor] = result
        except Exception as e:
            print(f"\nERROR processing {sensor}: {e}")
            import traceback
            traceback.print_exc()
            all_results[sensor] = {"error": str(e)}
    
    # Summary report
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE - SUMMARY (RANDOM FOREST)")
    print(f"{'='*70}")
    
    for sensor, result in all_results.items():
        if "error" in result:
            print(f"  {sensor:15s}: FAILED - {result['error']}")
        else:
            r = result["results"]
            oob_str = f", OOB={r['oob_score']:.4f}" if r.get('oob_score') is not None else ""
            print(f"  {sensor:15s}: Acc={r['accuracy']:.4f}, Macro F1={r['macro_f1']:.4f}{oob_str}")
    
    print("\n" + "="*30)
    print("MODEL ACCURACY SUMMARY")
    print("="*30)
    for s_name in ["vibration", "acoustic", "current", "temperature"]:
        if s_name in all_results and "results" in all_results[s_name]:
            acc_pct = round(all_results[s_name]["results"]["accuracy"] * 100)
            print(f"{s_name.capitalize():<12} {acc_pct}")
    # Save combined summary
    summary = {}
    for sensor, result in all_results.items():
        if "error" not in result:
            summary[sensor] = {
                "accuracy": result["results"]["accuracy"],
                "macro_f1": result["results"]["macro_f1"],
                "macro_precision": result["results"]["macro_precision"],
                "macro_recall": result["results"]["macro_recall"],
                "oob_score": result["results"].get("oob_score"),
                "best_params": result["best_params"],
                "n_features": len(result["feature_names"]),
            }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {output_dir / 'summary.json'}")
    print(f"All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
