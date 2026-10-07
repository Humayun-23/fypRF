"""
Evaluation and visualization module for Random Forest fault diagnosis.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

from config import FAULT_CLASSES, OUTPUT_DIR, TOP_K_FEATURES, N_PERMUTATION_REPEATS, RANDOM_STATE


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensor_name: str,
    class_names: List[str] = None
) -> Dict[str, Any]:
    """Evaluate model on test set and return comprehensive metrics."""
    y_pred = model.predict(X_test)
    
    if class_names is None:
        class_names = [str(i) for i in np.unique(y_test)]
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    # Check for OOB score on the model or pipeline step
    oob_score = None
    rf_estimator = model.named_steps["rf"] if hasattr(model, "named_steps") and "rf" in model.named_steps else model
    if hasattr(rf_estimator, "oob_score_"):
        oob_score = float(rf_estimator.oob_score_)
    
    results = {
        "sensor": sensor_name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "oob_score": oob_score,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "test_samples": int(len(y_test)),
    }
    
    print(f"\n{'='*60}")
    print(f"Test Evaluation - {sensor_name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"Macro Prec:{macro_precision:.4f}")
    print(f"Macro Rec: {macro_recall:.4f}")
    if oob_score is not None:
        print(f"OOB Score: {oob_score:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    return results


def save_results(results: Dict, sensor_name: str, output_dir: Path):
    """Save evaluation results to JSON and CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"{sensor_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    report = results["classification_report"]
    df = pd.DataFrame(report).T
    df.to_csv(output_dir / f"{sensor_name}_classification_report.csv")
    print(f"Results saved to {output_dir}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    sensor_name: str,
    save_path: Path,
    normalize: bool = True
):
    """Plot and save confusion matrix heatmap."""
    if normalize:
        with np.errstate(all='ignore'):
            row_sums = cm.sum(axis=1)[:, np.newaxis]
            cm_display = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        fmt = '.2f'
        title = f'{sensor_name} - Normalized Confusion Matrix'
    else:
        cm_display = cm
        fmt = 'd'
        title = f'{sensor_name} - Confusion Matrix'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_convergence(
    convergence_curve: List[float],
    sensor_name: str,
    save_path: Path,
    title: str = "Tuning Convergence"
):
    """Plot optimization convergence curve."""
    if not convergence_curve:
        return
    
    plt.figure(figsize=(10, 5))
    plt.plot(convergence_curve, 'b-', linewidth=2)
    plt.xlabel('Iteration / Evaluation', fontsize=12)
    plt.ylabel('Score / Fitness', fontsize=12)
    plt.title(f'{sensor_name} - {title}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    best_idx = int(np.argmin(convergence_curve))
    plt.plot(best_idx, convergence_curve[best_idx], 'ro', markersize=8, 
             label=f'Min: {convergence_curve[best_idx]:.4f} at #{best_idx}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved convergence plot to {save_path}")


def plot_oob_error_curve(
    error_curve: List[float],
    sensor_name: str,
    save_path: Path
):
    """Plot Out-Of-Bag (OOB) error progression across trees."""
    if not error_curve:
        return
    
    plt.figure(figsize=(10, 5))
    plt.plot(error_curve, 'g-', linewidth=2)
    plt.xlabel('Number of Estimators', fontsize=12)
    plt.ylabel('OOB Error Rate', fontsize=12)
    plt.title(f'{sensor_name} - Random Forest OOB Error vs Trees', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved OOB error curve to {save_path}")


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    sensor_name: str,
    save_path: Path,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
    top_k: int = TOP_K_FEATURES,
    use_permutation: bool = False
):
    """
    Compute and plot feature importance.
    Prefers native Random Forest Gini importance; falls back or optionally uses permutation importance.
    """
    print(f"Computing feature importance for {sensor_name}...")
    rf = model.named_steps["rf"] if hasattr(model, "named_steps") and "rf" in model.named_steps else model
    
    importances = None
    std = None
    importance_type = "MDI (Gini Importance)"
    
    if not use_permutation and hasattr(rf, "feature_importances_"):
        importances = rf.feature_importances_
        if hasattr(rf, "estimators_"):
            std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
        else:
            std = np.zeros_like(importances)
    elif X_test is not None and y_test is not None:
        try:
            result = permutation_importance(
                model, X_test, y_test,
                n_repeats=N_PERMUTATION_REPEATS,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                scoring='f1_macro'
            )
            importances = result.importances_mean
            std = result.importances_std
            importance_type = "Permutation Importance (F1 drop)"
        except Exception as e:
            print(f"Permutation importance failed: {e}")
            if hasattr(rf, "feature_importances_"):
                importances = rf.feature_importances_
                std = np.zeros_like(importances)
                importance_type = "MDI (Gini Importance)"
    
    if importances is None:
        print(f"No feature importances could be computed for {sensor_name}")
        return
    
    k = min(top_k, len(feature_names))
    top_indices = np.argsort(importances)[-k:][::-1]
    top_importances = importances[top_indices]
    top_std = std[top_indices] if std is not None else np.zeros_like(top_importances)
    top_names = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=(10, max(6, k * 0.3)))
    y_pos = np.arange(len(top_names))
    plt.barh(y_pos, top_importances, xerr=top_std, align='center', 
             capsize=3, color='steelblue', edgecolor='black', alpha=0.7)
    plt.yticks(y_pos, top_names)
    plt.xlabel(f'Feature Importance [{importance_type}]', fontsize=12)
    plt.title(f'{sensor_name} - Top {k} Feature Importance (Random Forest)', fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    importance_data = {
        "sensor": sensor_name,
        "type": importance_type,
        "top_features": [
            {"name": name, "importance": float(imp), "std": float(s)}
            for name, imp, s in zip(top_names, top_importances, top_std)
        ]
    }
    
    importance_path = save_path.with_suffix('.json')
    with open(importance_path, 'w') as f:
        json.dump(importance_data, f, indent=2)
    
    print(f"Saved feature importance to {save_path}")


def plot_class_distribution(
    y_train: np.ndarray,
    y_test: np.ndarray,
    sensor_name: str,
    save_path: Path,
    class_names: List[str] = None
):
    """Plot class distribution in train/test sets."""
    if class_names is None:
        class_names = FAULT_CLASSES
    
    train_counts = np.bincount(y_train, minlength=len(class_names))
    test_counts = np.bincount(y_test, minlength=len(class_names))
    
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, train_counts, width, label='Train', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, test_counts, width, label='Test', color='lightcoral', edgecolor='black')
    
    ax.set_xlabel('Fault Class', fontsize=12)
    ax.set_ylabel('Number of Windows', fontsize=12)
    ax.set_title(f'{sensor_name} - Class Distribution', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved class distribution to {save_path}")


def save_feature_list(feature_names: List[str], sensor_name: str, save_path: Path):
    """Save feature names to file."""
    with open(save_path, 'w') as f:
        for i, name in enumerate(feature_names):
            f.write(f"{i}: {name}\n")
    print(f"Saved feature list to {save_path}")


def create_summary_report(
    results: Dict,
    sensor_name: str,
    output_dir: Path,
    class_names: List[str] = None
):
    """Create a text summary report."""
    if class_names is None:
        class_names = FAULT_CLASSES
    
    report_path = output_dir / f"{sensor_name}_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"Random Forest Fault Diagnosis - {sensor_name.upper()} Sensor\n")
        f.write(f"{'='*60}\n\n")
        
        f.write(f"Test Set Performance:\n")
        f.write(f"  Accuracy:        {results['accuracy']:.4f}\n")
        f.write(f"  Macro F1:        {results['macro_f1']:.4f}\n")
        f.write(f"  Macro Precision: {results['macro_precision']:.4f}\n")
        f.write(f"  Macro Recall:    {results['macro_recall']:.4f}\n")
        if results.get("oob_score") is not None:
            f.write(f"  OOB Score:       {results['oob_score']:.4f}\n")
        f.write(f"\n")
        
        f.write(f"Test Samples: {results['test_samples']}\n\n")
        
        f.write(f"Per-Class Metrics:\n")
        report = results['classification_report']
        for cls in class_names:
            if cls in report:
                m = report[cls]
                f.write(f"  {cls:12s}: Prec={m['precision']:.4f}, Rec={m['recall']:.4f}, F1={m['f1-score']:.4f}, Support={m['support']}\n")
        
        f.write(f"\nConfusion Matrix:\n")
        cm = np.array(results['confusion_matrix'])
        f.write(f"  Rows=Actual, Cols=Predicted\n")
        f.write(f"  Classes: {class_names}\n")
        for i, row in enumerate(cm):
            f.write(f"  {class_names[i]:12s}: {row}\n")
    
    print(f"Saved summary report to {report_path}")


def run_full_evaluation(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    sensor_name: str,
    convergence_curve: List[float] = None,
    estimator_curve: List[float] = None,
    y_train: np.ndarray = None,
    output_dir: Path = None,
    class_names: List[str] = None
) -> Dict:
    """Run complete evaluation and generate all artifacts and plots."""
    if class_names is None:
        class_names = FAULT_CLASSES
    
    if output_dir is None:
        output_dir = OUTPUT_DIR / sensor_name
    else:
        output_dir = Path(output_dir) / sensor_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Test set evaluation
    results = evaluate_model(model, X_test, y_test, sensor_name, class_names)
    
    # 2. Save JSON and CSV
    save_results(results, sensor_name, output_dir)
    
    # 3. Confusion Matrix
    plot_confusion_matrix(
        np.array(results['confusion_matrix']),
        class_names,
        sensor_name,
        output_dir / f"{sensor_name}_confusion_matrix.png"
    )
    
    # 4. Convergence / Tuning plot (if applicable)
    if convergence_curve:
        plot_convergence(
            convergence_curve,
            sensor_name,
            output_dir / f"{sensor_name}_tuning_convergence.png",
            title="Tuning Convergence"
        )
    
    # 5. OOB / Tree curve (if applicable)
    if estimator_curve:
        plot_oob_error_curve(
            estimator_curve,
            sensor_name,
            output_dir / f"{sensor_name}_oob_error.png"
        )
    
    # 6. Feature Importance
    plot_feature_importance(
        model,
        feature_names,
        sensor_name,
        output_dir / f"{sensor_name}_feature_importance.png",
        X_test=X_test,
        y_test=y_test
    )
    
    # 7. Class Distribution
    if y_train is not None:
        plot_class_distribution(
            y_train,
            y_test,
            sensor_name,
            output_dir / f"{sensor_name}_class_distribution.png",
            class_names=class_names
        )
    
    # 8. Feature list text file
    save_feature_list(feature_names, sensor_name, output_dir / f"{sensor_name}_feature_list.txt")
    
    # 9. Summary text report
    create_summary_report(results, sensor_name, output_dir, class_names=class_names)
    
    return results
