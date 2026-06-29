import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

# Ensure package root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_experiment_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed
from src.validation.expanding_window import ExpandingWindowSplitter
from src.models.baselines import StatsmodelWrapper
from src.models.boosting import get_xgboost
from src.models.ensembles import HybridProbitMLEnsemble

logger = get_logger("run_robustness_gap")

def main():
    set_seed(42)
    config_path = "configs/experiments/main.yaml"
    config = load_experiment_config(config_path)
    
    base_dir = config.get("outputs", {}).get("base_dir", "outputs/")
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. Load data
    logger.info("Loading dataset...")
    df_path = os.path.join("data", "processed", "final_dataset.csv")
    if not os.path.exists(df_path):
        logger.error(f"Processed dataset not found at {df_path}. Run build_dataset.py first.")
        sys.exit(1)
        
    df = pd.read_csv(df_path)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)
    
    start_date_str = config.get("data", {}).get("start_date", "1980-01-01")
    if start_date_str:
        start_date = pd.to_datetime(start_date_str)
        df = df[df['DATE'] >= start_date].reset_index(drop=True)
        
    # Feature configurations
    feature_defs = config.get("feature_sets_definitions", {})
    full_features = [f for f in feature_defs.get("full", []) if f in df.columns]
    yield_features = [f for f in feature_defs.get("yield_only", []) if f in df.columns]
    
    # Run robustness analysis for Horizon = 6 (representative case)
    horizon = 6
    target_col = f"Recession_within_{horizon}mo"
    
    # Test multiple gap sizes:
    # 0 (leakage case), 3 (half-gap), 6 (correct gap), 12 (conservative gap)
    gap_sizes = [0, 3, 6, 12]
    
    models_to_test = {
        "yield_only_logit": {"features": yield_features, "class": StatsmodelWrapper(model_type="logit")},
        "xgboost": {"features": full_features, "class": get_xgboost(max_depth=3, n_estimators=50, random_state=42)},
        "hybrid_ensemble": {"features": full_features, "class": HybridProbitMLEnsemble(random_state=42)}
    }
    
    robustness_records = []
    
    for gap in gap_sizes:
        logger.info(f"Running experiments with temporal Gap size: {gap} months...")
        
        # Instantiate splitter with specific gap size
        splitter = ExpandingWindowSplitter(
            horizon=horizon,
            test_size_months=12,
            initial_train_end="2005-12-01",
            gap_equals_horizon=False,
            gap_months=gap
        )
        
        splits = list(splitter.split(df))
        
        # Dictionary to store predictions for this gap size
        # format: {model_name: {"y_true": [], "y_prob": []}}
        preds = {m_name: {"y_true": [], "y_prob": []} for m_name in models_to_test}
        
        for split_idx, (train_idx, test_idx) in enumerate(splits):
            train_df = df.iloc[train_idx].dropna(subset=[target_col])
            test_df = df.iloc[test_idx].dropna(subset=[target_col])
            
            if len(train_df) < 10 or len(test_df) == 0:
                continue
                
            for m_name, m_info in models_to_test.items():
                feats = m_info["features"]
                X_tr, y_tr = train_df[feats], train_df[target_col].astype(int)
                X_te, y_te = test_df[feats], test_df[target_col].astype(int)
                
                # Scale
                scaler = StandardScaler()
                X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr), columns=feats)
                X_te_scaled = pd.DataFrame(scaler.transform(X_te), columns=feats)
                
                # Fit model
                if m_name == "hybrid_ensemble":
                    # Stacking needs out-of-fold level 0 predictions
                    from sklearn.model_selection import KFold
                    kf = KFold(n_splits=5, shuffle=False)
                    inner_splits = list(kf.split(X_tr_scaled))
                    # Create a fresh model instance to avoid cross-contamination
                    model = HybridProbitMLEnsemble(random_state=42)
                    model.fit(X_tr_scaled, y_tr, inner_cv_splits=inner_splits)
                elif m_name == "yield_only_logit":
                    model = StatsmodelWrapper(model_type="logit")
                    model.fit(X_tr_scaled, y_tr)
                elif m_name == "xgboost":
                    model = get_xgboost(max_depth=3, n_estimators=50, random_state=42)
                    model.fit(X_tr_scaled, y_tr)
                    
                prob = model.predict_proba(X_te_scaled)[:, 1]
                preds[m_name]["y_true"].extend(y_te.values)
                preds[m_name]["y_prob"].extend(prob)
                
        # Evaluate metrics for each model under this gap size
        for m_name in models_to_test:
            y_t = np.array(preds[m_name]["y_true"])
            y_p = np.array(preds[m_name]["y_prob"])
            
            if len(np.unique(y_t)) < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(y_t, y_p)
            brier = brier_score_loss(y_t, y_p)
            
            robustness_records.append({
                "gap_months": gap,
                "model": m_name,
                "roc_auc": round(auc, 4) if not np.isnan(auc) else np.nan,
                "brier": round(brier, 4)
            })
            logger.info(f"Gap: {gap}mo | Model: {m_name} | AUC: {auc:.4f} | Brier: {brier:.4f}")
            
    # Save results
    robustness_df = pd.DataFrame(robustness_records)
    out_path = os.path.join(base_dir, "tables", "table_robustness_gap.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    robustness_df.to_csv(out_path, index=False)
    logger.info(f"Saved Gap robustness analysis table to {out_path}")

if __name__ == "__main__":
    main()
