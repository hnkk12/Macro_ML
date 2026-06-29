import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, f1_score

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) for binary classification."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if i == n_bins - 1:
            in_bin = in_bin | (y_prob == bin_upper)
            
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return ece

def compute_all_metrics(y_true, y_prob, thresholds=[0.15, 0.25, 0.35]):
    """Compute all evaluation metrics for predictions."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Handle edge case where there is only 1 class in y_true
    if len(np.unique(y_true)) < 2:
        roc_auc = np.nan
        pr_auc = np.nan
    else:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        
    brier = brier_score_loss(y_true, y_prob)
    
    # Avoid log loss infinity issues
    y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
    loss = log_loss(y_true, y_prob_clipped)
    
    ece = expected_calibration_error(y_true, y_prob)
    
    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "log_loss": loss,
        "ece": ece
    }
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        metrics[f"f1_at_{t}"] = f1_score(y_true, y_pred, zero_division=0)
        
    # Find best threshold for F1
    best_t = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = t
            
    metrics["best_f1_threshold"] = best_t
    metrics["best_f1_score"] = best_f1
    
    return metrics
