"""
Standalone Random Forest classifier module for fault diagnosis.

Provides:
- Model instantiation with balanced class weighting and OOB scoring
- Hyperparameter tuning via GridSearchCV, RandomizedSearchCV, and PSO
- Final model training, pipeline construction, and artifact serialization
- OOB error curve computation for tree ensemble diagnostics
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    RF_DEFAULT_PARAMS, RF_PARAM_GRID, RF_PARAM_DIST, RF_PSO_CONFIG,
    CV_FOLDS, CV_SCORING, RANDOM_STATE, OUTPUT_DIR
)
from evaluation import evaluate_model, save_results

# Optional PSO dependency
try:
    import pyswarms as ps
    HAS_PYSWARMS = True
except ImportError:
    HAS_PYSWARMS = False


def create_rf(params: Optional[Dict[str, Any]] = None, random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    """
    Create a RandomForestClassifier with specified or default hyperparameters.
    """
    merged_params = dict(RF_DEFAULT_PARAMS)
    if params:
        merged_params.update(params)
    
    # Ensure random_state is set
    merged_params["random_state"] = random_state
    
    return RandomForestClassifier(**merged_params)


def decode_rf_particle(position: np.ndarray) -> Dict[str, Any]:
    """
    Decode continuous PSO particle position to discrete Random Forest hyperparameters.
    Position layout: [n_estimators, max_depth, min_samples_split, min_samples_leaf]
    """
    bounds = RF_PSO_CONFIG["bounds"]
    n_est = int(np.clip(round(position[0]), bounds["n_estimators"][0], bounds["n_estimators"][1]))
    max_d = int(np.clip(round(position[1]), bounds["max_depth"][0], bounds["max_depth"][1]))
    min_split = int(np.clip(round(position[2]), bounds["min_samples_split"][0], bounds["min_samples_split"][1]))
    min_leaf = int(np.clip(round(position[3]), bounds["min_samples_leaf"][0], bounds["min_samples_leaf"][1]))
    
    return {
        "n_estimators": n_est,
        "max_depth": max_d,
        "min_samples_split": min_split,
        "min_samples_leaf": min_leaf,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "bootstrap": True,
        "oob_score": True,
        "n_jobs": -1,
    }


def optimize_grid(
    X: np.ndarray,
    y: np.ndarray,
    sensor_name: str,
    param_grid: Optional[Dict[str, list]] = None,
    cv_folds: int = CV_FOLDS,
    scoring: str = CV_SCORING,
    n_jobs: int = -1,
    verbose: bool = True
) -> Tuple[Dict[str, Any], List[float]]:
    """
    Hyperparameter optimization via GridSearchCV.
    """
    if param_grid is None:
        param_grid = RF_PARAM_GRID
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    base_rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        bootstrap=True,
        oob_score=True,
        n_jobs=n_jobs
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"GridSearchCV for {sensor_name}")
        print(f"Grid: {param_grid}")
        print(f"CV Folds: {cv_folds}, Scoring: {scoring}")
        print(f"{'='*60}")
    
    start_time = time.time()
    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1 if verbose else 0
    )
    grid_search.fit(X, y)
    elapsed = time.time() - start_time
    
    convergence = [-score for score in grid_search.cv_results_["mean_test_score"]]
    
    if verbose:
        print(f"GridSearchCV completed in {elapsed:.1f}s")
        print(f"Best CV {scoring}: {grid_search.best_score_:.4f}")
        print(f"Best params: {grid_search.best_params_}\n")
    
    return grid_search.best_params_, convergence


def optimize_randomized(
    X: np.ndarray,
    y: np.ndarray,
    sensor_name: str,
    param_dist: Optional[Dict[str, Any]] = None,
    n_iter: int = 25,
    cv_folds: int = CV_FOLDS,
    scoring: str = CV_SCORING,
    n_jobs: int = -1,
    verbose: bool = True
) -> Tuple[Dict[str, Any], List[float]]:
    """
    Hyperparameter optimization via RandomizedSearchCV.
    """
    if param_dist is None:
        param_dist = RF_PARAM_DIST
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    base_rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        bootstrap=True,
        oob_score=True,
        n_jobs=n_jobs
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RandomizedSearchCV for {sensor_name}")
        print(f"Iterations: {n_iter}, CV Folds: {cv_folds}, Scoring: {scoring}")
        print(f"{'='*60}")
    
    start_time = time.time()
    rand_search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
        verbose=1 if verbose else 0
    )
    rand_search.fit(X, y)
    elapsed = time.time() - start_time
    
    convergence = [-score for score in rand_search.cv_results_["mean_test_score"]]
    
    if verbose:
        print(f"RandomizedSearchCV completed in {elapsed:.1f}s")
        print(f"Best CV {scoring}: {rand_search.best_score_:.4f}")
        print(f"Best params: {rand_search.best_params_}\n")
    
    return rand_search.best_params_, convergence


def optimize_pso(
    X: np.ndarray,
    y: np.ndarray,
    sensor_name: str,
    n_particles: Optional[int] = None,
    n_iterations: Optional[int] = None,
    cv_folds: int = CV_FOLDS,
    scoring: str = CV_SCORING,
    verbose: bool = True
) -> Tuple[Dict[str, Any], List[float]]:
    """
    Hyperparameter optimization via Particle Swarm Optimization (PSO).
    """
    if not HAS_PYSWARMS:
        raise ImportError("pyswarms is not installed. Please install pyswarms or use --tune grid / --tune randomized.")
    
    pso_cfg = RF_PSO_CONFIG
    n_particles = n_particles or pso_cfg["n_particles"]
    n_iterations = n_iterations or pso_cfg["n_iterations"]
    
    bounds_dict = pso_cfg["bounds"]
    lower_bounds = np.array([bounds_dict[k][0] for k in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]])
    upper_bounds = np.array([bounds_dict[k][1] for k in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]])
    bounds = (lower_bounds, upper_bounds)
    
    options = {
        "c1": pso_cfg["c1"],
        "c2": pso_cfg["c2"],
        "w": pso_cfg["w"],
    }
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    def fitness_function(positions: np.ndarray) -> np.ndarray:
        fitness = np.zeros(positions.shape[0])
        for idx in range(positions.shape[0]):
            params = decode_rf_particle(positions[idx])
            rf = create_rf(params)
            try:
                scores = cross_val_score(rf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                fitness[idx] = -float(np.mean(scores))  # Minimize negative CV score
            except Exception as e:
                fitness[idx] = 1.0  # Worst fitness
        return fitness
    
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=4,
        options=options,
        bounds=bounds,
        ftol=1e-5,
        ftol_iter=10
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PSO Optimization for Random Forest ({sensor_name})")
        print(f"Particles: {n_particles}, Iterations: {n_iterations}")
        print(f"CV Folds: {cv_folds}, Scoring: {scoring}")
        print(f"{'='*60}")
    
    start_time = time.time()
    best_cost, best_pos = optimizer.optimize(fitness_function, iters=n_iterations, verbose=verbose)
    elapsed = time.time() - start_time
    
    best_params = decode_rf_particle(best_pos)
    best_score = -best_cost
    
    if verbose:
        print(f"PSO completed in {elapsed:.1f}s")
        print(f"Best CV {scoring}: {best_score:.4f}")
        print(f"Best params: {best_params}\n")
    
    return best_params, optimizer.cost_history


def compute_oob_curve(
    X_train: np.ndarray,
    y_train: np.ndarray,
    base_params: Dict[str, Any],
    min_trees: int = 10,
    max_trees: int = 200,
    step: int = 10
) -> Tuple[List[int], List[float]]:
    """
    Track Out-Of-Bag (OOB) error rate across increasing numbers of estimators.
    """
    tree_counts = list(range(min_trees, max_trees + 1, step))
    oob_errors = []
    
    params = dict(base_params)
    params["warm_start"] = True
    params["oob_score"] = True
    params["n_jobs"] = -1
    params["random_state"] = RANDOM_STATE
    
    rf = RandomForestClassifier(**params)
    
    for n in tree_counts:
        rf.set_params(n_estimators=n)
        rf.fit(X_train, y_train)
        oob_error = 1.0 - rf.oob_score_
        oob_errors.append(float(oob_error))
    
    return tree_counts, oob_errors


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    best_params: Dict[str, Any],
    sensor_name: str,
    save_path: Optional[Path] = None,
    include_scaler: bool = False
) -> Tuple[Pipeline, RandomForestClassifier]:
    """
    Train final Random Forest model on the complete training set and export artifacts.
    
    Args:
        X_train: Training feature matrix
        y_train: Training class labels
        best_params: Tuned or default hyperparameters
        sensor_name: Sensor identifier (e.g., 'vibration', 'acoustic', 'current_temp')
        save_path: Directory to export joblib models and param json
        include_scaler: Whether to prepend StandardScaler in pipeline
        
    Returns:
        (pipeline, trained_rf)
    """
    rf = create_rf(best_params)
    
    steps = []
    if include_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("rf", rf))
    
    pipeline = Pipeline(steps)
    
    print(f"\nTraining final Random Forest for {sensor_name}...")
    pipeline.fit(X_train, y_train)
    
    trained_rf = pipeline.named_steps["rf"]
    
    print("Training completed.")
    print(f"  Estimators: {trained_rf.n_estimators}")
    print(f"  Max Depth:  {trained_rf.max_depth}")
    if hasattr(trained_rf, "oob_score_"):
        print(f"  OOB Score:  {trained_rf.oob_score_:.4f}")
    
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline
        dump(pipeline, save_path / f"{sensor_name}_pipeline.joblib")
        
        # Save RF classifier
        dump(trained_rf, save_path / f"{sensor_name}_rf.joblib")
        
        # Save best hyperparameters JSON
        with open(save_path / f"{sensor_name}_best_params.json", "w") as f:
            clean_params = {}
            for k, v in best_params.items():
                if isinstance(v, (np.integer, np.int64, np.int32)):
                    clean_params[k] = int(v)
                elif isinstance(v, (np.floating, np.float64, np.float32)):
                    clean_params[k] = float(v)
                else:
                    clean_params[k] = v
            json.dump(clean_params, f, indent=2)
        
        print(f"Saved model artifacts to {save_path}")
    
    return pipeline, trained_rf
