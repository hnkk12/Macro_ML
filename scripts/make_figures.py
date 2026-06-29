import argparse
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

# Ensure package root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_experiment_config
from src.utils.logging_utils import get_logger

logger = get_logger("make_figures")

def plot_probability_paths(preds_df: pd.DataFrame, horizon: int, output_path: str) -> None:
    """Plot predicted recession probabilities over time with NBER recession shading."""
    plt.figure(figsize=(15, 6))
    
    # Filter predictions for core feature set or ensemble
    # Let's plot the hybrid_ensemble and standard baselines
    df_h = preds_df[preds_df['horizon'] == horizon].copy()
    df_h['forecast_origin'] = pd.to_datetime(df_h['forecast_origin'])
    
    # We want to find the true recession periods for shading
    # Group by date to get a single true label series
    true_series = df_h.groupby('forecast_origin')['y_true'].max().sort_index()
    dates = true_series.index
    y_true = true_series.values
    
    # Draw NBER shading (where y_true == 1)
    # We find contiguous periods of 1s
    in_recession = False
    start_date = None
    for idx, d in enumerate(dates):
        val = y_true[idx]
        if val == 1 and not in_recession:
            in_recession = True
            start_date = d
        elif val == 0 and in_recession:
            in_recession = False
            plt.axvspan(start_date, d, color='grey', alpha=0.3, label='Recession' if 'Recession' not in plt.gca().get_legend_handles_labels()[1] else "")
            
    # Add final period if sample ends in recession
    if in_recession:
        plt.axvspan(start_date, dates[-1], color='grey', alpha=0.3)
        
    # Plot models
    models_to_plot = ['hybrid_ensemble', 'yield_only_logit', 'xgboost']
    colors = {'hybrid_ensemble': 'red', 'yield_only_logit': 'blue', 'xgboost': 'green'}
    
    for m in models_to_plot:
        df_m = df_h[df_h['model'] == m].sort_values('forecast_origin')
        if not df_m.empty:
            # Group by date (in case there are multiple feature sets, take full or average)
            # Let's take the 'full' feature set if available, otherwise first
            if 'full' in df_m['feature_set'].values:
                df_plot = df_m[df_m['feature_set'] == 'full']
            else:
                df_plot = df_m[df_m['feature_set'] == df_m['feature_set'].iloc[0]]
                
            plt.plot(df_plot['forecast_origin'], df_plot['y_prob'], label=f"{m} (probs)", color=colors.get(m, None), linewidth=1.5)
            
    plt.title(f"Out-of-Sample Predicted Recession Probabilities (Horizon = {horizon} months)", fontsize=14)
    plt.xlabel("Forecast Origin", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved probability paths figure to {output_path}")

def plot_calibration_curves(preds_df: pd.DataFrame, horizon: int, output_path: str) -> None:
    """Plot reliability diagram (calibration curves)."""
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    df_h = preds_df[preds_df['horizon'] == horizon].copy()
    models = ['hybrid_ensemble', 'yield_only_logit', 'xgboost']
    
    for m in models:
        df_m = df_h[df_h['model'] == m]
        if not df_m.empty:
            # Filter feature set
            if 'full' in df_m['feature_set'].values:
                df_sub = df_m[df_m['feature_set'] == 'full']
            else:
                df_sub = df_m[df_m['feature_set'] == df_m['feature_set'].iloc[0]]
                
            y_true = df_sub['y_true'].values
            y_prob = df_sub['y_prob'].values
            
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            plt.plot(prob_pred, prob_true, "s-", label=f"{m}")
            
    plt.ylabel("Fraction of positives", fontsize=12)
    plt.xlabel("Mean predicted probability", fontsize=12)
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.title(f"Reliability Diagram (Horizon = {horizon} months)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved calibration curves to {output_path}")

def plot_shap_by_episode(shap_path: str, output_path: str) -> None:
    """Plot C3 SHAP analysis per recession episode."""
    if not os.path.exists(shap_path):
        logger.warning(f"SHAP data not found at {shap_path}, skipping SHAP figure.")
        return
        
    df = pd.read_csv(shap_path)
    if df.empty:
        return
        
    # Group by episode and feature, taking average across splits
    df_grouped = df.groupby(['episode', 'feature'])['mean_abs_shap'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_grouped,
        x='episode',
        y='mean_abs_shap',
        hue='feature'
    )
    plt.title("Feature Importance (Mean Absolute SHAP) per Recession Episode (Horizon = 6m)", fontsize=14)
    plt.xlabel("Recession Episode", fontsize=12)
    plt.ylabel("Mean Absolute SHAP Value", fontsize=12)
    plt.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved SHAP by episode figure to {output_path}")

def plot_pr_curves(preds_df: pd.DataFrame, horizon: int, output_path: str) -> None:
    """Plot Precision-Recall curves."""
    plt.figure(figsize=(8, 8))
    
    df_h = preds_df[preds_df['horizon'] == horizon].copy()
    models = ['hybrid_ensemble', 'yield_only_logit', 'xgboost']
    
    for m in models:
        df_m = df_h[df_h['model'] == m]
        if not df_m.empty:
            if 'full' in df_m['feature_set'].values:
                df_sub = df_m[df_m['feature_set'] == 'full']
            else:
                df_sub = df_m[df_m['feature_set'] == df_m['feature_set'].iloc[0]]
                
            y_true = df_sub['y_true'].values
            y_prob = df_sub['y_prob'].values
            
            # Skip if only 1 class present
            if len(np.unique(y_true)) < 2:
                continue
                
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            plt.plot(recall, precision, label=f"{m}")
            
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower left")
    plt.title(f"Precision-Recall Curves (Horizon = {horizon} months)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved PR curves to {output_path}")

def plot_feature_stability(coefs_path: str, output_path: str) -> None:
    """Plot coefficients stability over splits."""
    if not os.path.exists(coefs_path):
        logger.warning(f"Coefficients file not found at {coefs_path}, skipping stability figure.")
        return
        
    df = pd.read_csv(coefs_path)
    # Filter for low_revision or core econometric to avoid cluttering
    # Let's plot yield_only_logit model coefficients over splits
    df_model = df[df['model'] == 'yield_only_logit']
    if df_model.empty:
        df_model = df[df['model'] == 'logistic_l2']
        
    if df_model.empty:
        return
        
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_model,
        x='split_id',
        y='standardized_coefficient',
        hue='feature',
        marker='o'
    )
    plt.title("Stability of Standardized Coefficients Across Splits (Horizon = 6m)", fontsize=14)
    plt.xlabel("Split ID (Expanding Window)", fontsize=12)
    plt.ylabel("Standardized Coefficient Value", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved feature stability figure to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic and result figures.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML file.")
    args = parser.parse_args()
    
    logger.info(f"Loading config from {args.config}")
    config = load_experiment_config(args.config)
    
    base_dir = config.get("outputs", {}).get("base_dir", "outputs/")
    
    # Load horizon 6 predictions as main reference for curves
    pred_path = os.path.join(base_dir, "predictions", "oos_predictions_h6.csv")
    
    # If not found, look for any predictions file
    if not os.path.exists(pred_path):
        pred_files = [f for f in os.listdir(os.path.join(base_dir, "predictions")) if f.endswith('.csv')]
        if pred_files:
            pred_path = os.path.join(base_dir, "predictions", pred_files[0])
            logger.info(f"Using {pred_path} for figures.")
        else:
            logger.error("No predictions found to build figures.")
            sys.exit(1)
            
    preds_df = pd.read_csv(pred_path)
    
    logger.info("Generating figures...")
    
    # 1. Probability Paths
    plot_probability_paths(preds_df, 6, os.path.join(base_dir, "figures", "probability_paths.png"))
    
    # 2. Calibration Curves
    plot_calibration_curves(preds_df, 6, os.path.join(base_dir, "figures", "calibration_curves.png"))
    
    # 3. Precision-Recall Curves
    plot_pr_curves(preds_df, 6, os.path.join(base_dir, "figures", "pr_curves.png"))
    
    # 4. SHAP by episode (C3)
    shap_path = os.path.join(base_dir, "explainability", "shap_by_episode_h6.csv")
    plot_shap_by_episode(shap_path, os.path.join(base_dir, "figures", "shap_by_episode.png"))
    
    # 5. Feature Stability
    coefs_path = os.path.join(base_dir, "explainability", "logistic_coefficients_h6.csv")
    plot_feature_stability(coefs_path, os.path.join(base_dir, "figures", "feature_stability.png"))
    
    logger.info("Figures generated successfully!")

if __name__ == "__main__":
    main()
