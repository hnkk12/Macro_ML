import argparse
import sys
import os
import pandas as pd

# Ensure package root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_experiment_config
from src.utils.logging_utils import get_logger
from src.data.panel_builder import build_panel
from src.features.feature_sets import compute_features, save_feature_dictionary
from src.features.target_builder import make_forward_target
from src.evaluation.tables import generate_target_summary

logger = get_logger("build_dataset")

def main():
    parser = argparse.ArgumentParser(description="Process raw data, build features and targets.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML file.")
    args = parser.parse_args()
    
    logger.info(f"Loading config from {args.config}")
    config = load_experiment_config(args.config)
    
    # 1. Build panel from raw series
    logger.info("Merging raw series into panel_revised.csv...")
    panel_df = build_panel()
    if panel_df.empty:
        logger.error("Panel builder returned empty DataFrame. Make sure downloader ran successfully first.")
        sys.exit(1)
        
    # 2. Compute features
    logger.info("Computing macro features...")
    features_df = compute_features(panel_df)
    
    # 3. Build targets for each horizon
    logger.info("Building targets for horizons...")
    horizons = config.get("targets", {}).get("horizons", [3, 6, 12])
    
    final_df = features_df.copy()
    
    # Check if 'USREC' column is in panel_df
    if 'USREC' not in panel_df.columns:
        logger.error("USREC (recession indicator) is missing from raw data.")
        sys.exit(1)
        
    for h in horizons:
        # y_t is 1 if there is a recession in any of the next h months [t+1, t+h]
        # Store as Recession_within_Hmo
        logger.info(f"Building target for horizon {h} months...")
        target_series = make_forward_target(panel_df['USREC'], h)
        final_df[f"Recession_within_{h}mo"] = target_series
        # Also compute standard point target Recession_in_Hmo for legacy support
        final_df[f"Recession_in_{h}mo"] = panel_df['USREC'].shift(-h)
        
    # Save final dataset
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    csv_out_path = os.path.join("data", "processed", "final_dataset.csv")
    final_df.to_csv(csv_out_path, index=False)
    logger.info(f"Saved final combined dataset to {csv_out_path} (shape: {final_df.shape})")
    
    # 4. Save metadata outputs
    base_dir = config.get("outputs", {}).get("base_dir", "outputs/")
    
    dict_path = os.path.join(base_dir, "datasets", "feature_dictionary.csv")
    save_feature_dictionary(dict_path)
    
    summary_path = os.path.join(base_dir, "datasets", "target_summary.csv")
    generate_target_summary(final_df, horizons, summary_path)
    logger.info("Dataset construction finished successfully!")

if __name__ == "__main__":
    main()
