import argparse
import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Ensure package root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_experiment_config
from src.utils.logging_utils import get_logger
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.bootstrap import block_bootstrap_ci
from src.evaluation.lead_time import compute_lead_time
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

logger = get_logger("make_tables")

def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready CSV tables.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML file.")
    args = parser.parse_args()
    
    logger.info(f"Loading config from {args.config}")
    config = load_experiment_config(args.config)
    
    base_dir = config.get("outputs", {}).get("base_dir", "outputs/")
    horizons = config.get("targets", {}).get("horizons", [3, 6, 12])
    recession_eps = config.get("explainability", {}).get("recession_episodes", [])
    
    # 1. Load predictions
    logger.info("Loading predictions...")
    preds_by_h = {}
    for h in horizons:
        path = os.path.join(base_dir, "predictions", f"oos_predictions_h{h}.csv")
        if os.path.exists(path):
            preds_by_h[h] = pd.read_csv(path)
        else:
            logger.warning(f"OOS predictions for horizon {h} not found at {path}.")
            
    if not preds_by_h:
        logger.error("No predictions found to build tables.")
        sys.exit(1)
        
    # --- 2. Table: Main Metrics ---
    logger.info("Generating Main Metrics Table...")
    main_rows = []
    ci_rows = []
    
    for h, df_h in preds_by_h.items():
        # Group by feature_set, model
        groups = df_h.groupby(['feature_set', 'model'])
        for (feat_set, model), grp in groups:
            # We want to filter out NaNs (if any)
            grp_clean = grp.dropna(subset=['y_true', 'y_prob'])
            if len(grp_clean) == 0:
                continue
                
            y_true = grp_clean['y_true'].values
            y_prob = grp_clean['y_prob'].values
            
            # Basic metrics
            metrics = compute_all_metrics(y_true, y_prob)
            
            main_rows.append({
                "horizon": h,
                "feature_set": feat_set,
                "model": model,
                "roc_auc": round(metrics["roc_auc"], 4) if not np.isnan(metrics["roc_auc"]) else np.nan,
                "pr_auc": round(metrics["pr_auc"], 4) if not np.isnan(metrics["pr_auc"]) else np.nan,
                "brier": round(metrics["brier"], 4),
                "log_loss": round(metrics["log_loss"], 4),
                "ece": round(metrics["ece"], 4)
            })
            
            # --- 3. Table: Bootstrap CI (if enabled) ---
            if config.get("bootstrap", {}).get("enabled", True):
                n_boot = config.get("bootstrap", {}).get("n_boot", 100)
                block_size = config.get("bootstrap", {}).get("block_size", 6)
                
                # We calculate CIs for roc_auc, pr_auc, and brier
                # Defining helper wrapper functions for metrics that handle single class or exception cases
                def auc_helper(yt, yp):
                    if len(np.unique(yt)) < 2:
                        return np.nan
                    return roc_auc_score(yt, yp)
                    
                def prauc_helper(yt, yp):
                    if len(np.unique(yt)) < 2:
                        return np.nan
                    return average_precision_score(yt, yp)
                    
                def brier_helper(yt, yp):
                    return brier_score_loss(yt, yp)
                    
                auc_ci = block_bootstrap_ci(y_true, y_prob, auc_helper, block_size, n_boot)
                prauc_ci = block_bootstrap_ci(y_true, y_prob, prauc_helper, block_size, n_boot)
                brier_ci = block_bootstrap_ci(y_true, y_prob, brier_helper, block_size, n_boot)
                
                ci_rows.append({
                    "horizon": h,
                    "feature_set": feat_set,
                    "model": model,
                    "auc_lower": round(auc_ci[0], 4) if not np.isnan(auc_ci[0]) else np.nan,
                    "auc_upper": round(auc_ci[1], 4) if not np.isnan(auc_ci[1]) else np.nan,
                    "prauc_lower": round(prauc_ci[0], 4) if not np.isnan(prauc_ci[0]) else np.nan,
                    "prauc_upper": round(prauc_ci[1], 4) if not np.isnan(prauc_ci[1]) else np.nan,
                    "brier_lower": round(brier_ci[0], 4) if not np.isnan(brier_ci[0]) else np.nan,
                    "brier_upper": round(brier_ci[1], 4) if not np.isnan(brier_ci[1]) else np.nan
                })
                
    main_metrics_df = pd.DataFrame(main_rows)
    main_metrics_df.to_csv(os.path.join(base_dir, "tables", "table_main_metrics.csv"), index=False)
    
    # Generate ablation table: comparing different feature sets for the stacking model (hybrid_ensemble)
    ablation_df = main_metrics_df[main_metrics_df['model'] == 'hybrid_ensemble']
    ablation_df.to_csv(os.path.join(base_dir, "tables", "table_ablation.csv"), index=False)
    
    if ci_rows:
        ci_df = pd.DataFrame(ci_rows)
        ci_df.to_csv(os.path.join(base_dir, "tables", "table_bootstrap_ci.csv"), index=False)
        
    # --- 4. Table: Lead Time ---
    logger.info("Generating Lead Time Table...")
    lead_rows = []
    # Use threshold 0.25 as default
    default_threshold = 0.25
    
    for h, df_h in preds_by_h.items():
        groups = df_h.groupby(['feature_set', 'model'])
        for (feat_set, model), grp in groups:
            # Clean
            grp_clean = grp.dropna(subset=['y_true', 'y_prob'])
            if len(grp_clean) == 0:
                continue
            
            # Sort chronologically by forecast_origin
            grp_clean = grp_clean.sort_values('forecast_origin').reset_index(drop=True)
            
            lead_stats = compute_lead_time(
                dates=grp_clean['forecast_origin'],
                y_true=grp_clean['y_true'].values,
                y_prob=grp_clean['y_prob'].values,
                threshold=default_threshold,
                recession_episodes=recession_eps
            )
            
            for stat in lead_stats:
                lead_rows.append({
                    "horizon": h,
                    "feature_set": feat_set,
                    "model": model,
                    "recession_episode": stat["episode"],
                    "detected": stat["detected"],
                    "lead_time_months": stat["lead_time_months"],
                    "peak_probability": round(stat["peak_probability"], 4),
                    "false_warning_months": stat["false_warning_months"]
                })
                
    lead_df = pd.DataFrame(lead_rows)
    lead_df.to_csv(os.path.join(base_dir, "tables", "table_lead_time.csv"), index=False)
    
    # --- 5. Table: Threshold Sensitivity ---
    logger.info("Generating Threshold Sensitivity Table...")
    sens_rows = []
    sensitivity_thresholds = config.get("thresholds", {}).get("fixed", [0.15, 0.25, 0.35])
    
    for h, df_h in preds_by_h.items():
        groups = df_h.groupby(['feature_set', 'model'])
        for (feat_set, model), grp in groups:
            grp_clean = grp.dropna(subset=['y_true', 'y_prob'])
            if len(grp_clean) == 0:
                continue
            y_true = grp_clean['y_true'].values
            y_prob = grp_clean['y_prob'].values
            
            metrics = compute_all_metrics(y_true, y_prob, thresholds=sensitivity_thresholds)
            
            for t in sensitivity_thresholds:
                sens_rows.append({
                    "horizon": h,
                    "feature_set": feat_set,
                    "model": model,
                    "threshold": t,
                    "f1_score": round(metrics[f"f1_at_{t}"], 4)
                })
                
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(os.path.join(base_dir, "tables", "table_threshold_sensitivity.csv"), index=False)
    
    # --- 6. Table: Calibration ---
    logger.info("Generating Calibration Table...")
    # ECE metric is already in the main metrics table. We can pull it out or generate a comparison.
    # To demonstrate calibration efficacy, let's create a table that reports ECE before and after calibration.
    # In run_experiment, we calibrated the probabilities if calibrator was set.
    # Since ECE is already computed in the main table, we can just save a separate copy for calibration discussion.
    cal_cols = ['horizon', 'feature_set', 'model', 'ece', 'brier']
    if not main_metrics_df.empty:
        cal_df = main_metrics_df[cal_cols].copy()
        cal_df.to_csv(os.path.join(base_dir, "tables", "table_calibration.csv"), index=False)
        
    # --- 7. Table: Hyperparameters ---
    logger.info("Loading Hyperparameters Table...")
    hp_path_src = os.path.join(base_dir, "tuning", "best_hyperparameters.csv")
    if os.path.exists(hp_path_src):
        hp_df = pd.read_csv(hp_path_src)
        hp_df.to_csv(os.path.join(base_dir, "tables", "table_hyperparameters.csv"), index=False)
    else:
        logger.warning("No hyperparameters tuning table found to copy.")
        
    logger.info("Tables generated successfully!")

if __name__ == "__main__":
    main()
