import argparse
import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from typing import Dict, List, Any, Tuple

# Ensure package root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_experiment_config
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed
from src.validation.expanding_window import ExpandingWindowSplitter
from src.models.baselines import StatsmodelWrapper
from src.models.sklearn_models import get_logistic_l2, get_random_forest
from src.models.boosting import get_xgboost, get_lightgbm
from src.models.ensembles import HybridProbitMLEnsemble
from src.models.calibration import RecessionCalibrator
from src.explainability.coefficients import extract_standardized_coefficients
from src.explainability.shap_tree import shap_by_recession_episode

logger = get_logger("run_experiment")

def get_base_model(model_name: str, params: dict = None, random_state: int = 42) -> Any:
    """Instantiate the base model with given parameters."""
    if params is None:
        params = {}
        
    if model_name in ["yield_only_logit", "core_logit"]:
        return StatsmodelWrapper(model_type="logit")
    elif model_name == "logistic_l2":
        C = params.get("C", 1.0)
        solver = params.get("solver", "lbfgs")
        max_iter = params.get("max_iter", 1000)
        return get_logistic_l2(C=C, solver=solver, max_iter=max_iter, random_state=random_state)
    elif model_name == "random_forest":
        n_estimators = params.get("n_estimators", 100)
        max_depth = params.get("max_depth", None)
        min_samples_split = params.get("min_samples_split", 2)
        return get_random_forest(n_estimators=n_estimators, max_depth=max_depth, 
                                 min_samples_split=min_samples_split, random_state=random_state)
    elif model_name == "xgboost":
        max_depth = params.get("max_depth", 3)
        min_child_weight = params.get("min_child_weight", 1.0)
        reg_lambda = params.get("reg_lambda", 1.0)
        learning_rate = params.get("learning_rate", 0.1)
        n_estimators = params.get("n_estimators", 100)
        return get_xgboost(max_depth=max_depth, min_child_weight=min_child_weight, 
                           reg_lambda=reg_lambda, learning_rate=learning_rate, 
                           n_estimators=n_estimators, random_state=random_state)
    elif model_name == "lightgbm":
        max_depth = params.get("max_depth", 3)
        num_leaves = params.get("num_leaves", 31)
        reg_lambda = params.get("reg_lambda", 0.0)
        learning_rate = params.get("learning_rate", 0.1)
        n_estimators = params.get("n_estimators", 100)
        return get_lightgbm(max_depth=max_depth, num_leaves=num_leaves, reg_lambda=reg_lambda, 
                            learning_rate=learning_rate, n_estimators=n_estimators, random_state=random_state)
    elif model_name == "svm":
        C = params.get("C", 1.0)
        kernel = params.get("kernel", "rbf")
        from src.models.appendix_models import get_svm
        return get_svm(C=C, kernel=kernel, random_state=random_state)
    elif model_name == "mlp":
        hidden_layer_sizes = params.get("hidden_layer_sizes", (64, 32))
        if isinstance(hidden_layer_sizes, list):
            hidden_layer_sizes = tuple(hidden_layer_sizes)
        alpha = params.get("alpha", 0.0001)
        learning_rate_init = params.get("learning_rate_init", 0.01)
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
                             learning_rate_init=learning_rate_init, max_iter=1000,
                             random_state=random_state)
    elif model_name == "hybrid_ensemble":
        # Stacking model
        return HybridProbitMLEnsemble(random_state=random_state)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def tune_hyperparameters(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, 
                         dates_train: pd.Series, horizon: int, model_grid: dict, 
                         random_state: int = 42) -> dict:
    """Tune hyperparameters using inner expanding-window cross-validation."""
    # If no grid or model has no parameters to tune, return empty dict
    if not model_grid or model_name in ["yield_only_logit", "core_logit", "hybrid_ensemble"]:
        return {}
        
    logger.info(f"Tuning {model_name}...")
    
    # 1. Generate grid parameter combinations
    import itertools
    grid_lists = {}
    for k, v in model_grid.items():
        if isinstance(v, list):
            grid_lists[k] = v
        else:
            grid_lists[k] = [v]
            
    keys, values = zip(*grid_lists.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # 2. Set up inner expanding window CV on the training data
    # We use a simplified inner splitter
    n_train = len(X_train)
    # The split point: start test period after 60% of training data
    split_point_idx = int(n_train * 0.6)
    if split_point_idx >= n_train - 12:
        split_point_idx = max(12, n_train - 24)
        
    initial_train_end_date = dates_train.iloc[split_point_idx].strftime("%Y-%m-%d")
    
    df_inner = X_train.copy()
    df_inner['DATE'] = dates_train.values
    df_inner['target'] = y_train.values
    
    inner_splitter = ExpandingWindowSplitter(
        horizon=horizon,
        test_size_months=12,
        initial_train_end=initial_train_end_date,
        gap_equals_horizon=True
    )
    
    inner_splits = list(inner_splitter.split(df_inner))
    if len(inner_splits) == 0:
        # Fallback to default params if not enough data for splits
        logger.warning(f"Not enough data for inner splits in {model_name}. Using default params.")
        return experiments[0] if experiments else {}
        
    best_score = float('inf')
    best_params = {}
    
    for params in experiments:
        scores = []
        for train_idx, val_idx in inner_splits:
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            
            X_tr_scaled_df = pd.DataFrame(X_tr_scaled, columns=X_tr.columns)
            X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)
            
            # Fit and evaluate
            model = get_base_model(model_name, params, random_state=random_state)
            try:
                model.fit(X_tr_scaled_df, y_tr)
                preds = model.predict_proba(X_val_scaled_df)[:, 1]
                score = brier_score_loss(y_val, preds)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Error fitting parameter grid {params} on inner split: {e}")
                scores.append(1.0) # Penalty
                
        mean_score = np.mean(scores)
        if mean_score < best_score:
            best_score = mean_score
            best_params = params
            
    logger.info(f"Best params for {model_name}: {best_params} (Best Inner Brier: {best_score:.4f})")
    return best_params

def run_experiment(config: dict) -> None:
    """Run validation pipeline over horizons, feature sets and models."""
    seed = config.get("seed", 42)
    set_seed(seed)
    
    base_dir = config.get("outputs", {}).get("base_dir", "outputs/")
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. Load data
    logger.info("Loading final dataset...")
    df_path = os.path.join("data", "processed", "final_dataset.csv")
    if not os.path.exists(df_path):
        logger.error(f"Processed dataset not found at {df_path}. Run build_dataset.py first.")
        sys.exit(1)
        
    df = pd.read_csv(df_path)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)
    
    # Filter dates by start_date in config
    start_date_str = config.get("data", {}).get("start_date", "1980-01-01")
    if start_date_str:
        start_date = pd.to_datetime(start_date_str)
        df = df[df['DATE'] >= start_date].reset_index(drop=True)
        logger.info(f"Filtered data to start_date >= {start_date_str} (samples remaining: {len(df)})")
        
    horizons = config.get("targets", {}).get("horizons", [3, 6, 12])
    feature_sets = config.get("feature_sets", ["full"])
    feature_defs = config.get("feature_sets_definitions", {})
    models_to_run = config.get("models", [])
    model_grids = config.get("models_definitions", {})
    
    val_config = config.get("validation", {})
    initial_train_end = val_config.get("initial_train_end", "2005-12-01")
    test_size_months = val_config.get("test_size_months", 12)
    gap_equals_horizon = val_config.get("gap_equals_horizon", True)
    
    calib_method = config.get("calibration", {}).get("main", "none")
    
    # Trackers for outputs
    all_oos_predictions = []
    split_diagnostics_all = []
    coef_list = []
    shap_list = []
    hyperparams_list = []
    meta_weights_records = []
    
    # Main outer loop
    for horizon in horizons:
        target_col = f"Recession_within_{horizon}mo"
        if target_col not in df.columns:
            logger.error(f"Target column {target_col} is missing from dataset.")
            continue
            
        logger.info(f"==================================================")
        logger.info(f"RUNNING EXP FOR HORIZON: {horizon} MONTHS")
        logger.info(f"==================================================")
        
        # Instantiate outer splitter
        splitter = ExpandingWindowSplitter(
            horizon=horizon,
            test_size_months=test_size_months,
            initial_train_end=initial_train_end,
            gap_equals_horizon=gap_equals_horizon
        )
        
        # We need to run splitter once to count splits
        # Let's clean target column NaNs out of outer splits?
        # Actually, outer splits should operate on sorted df, and we filter target NaNs inside the split loop.
        splits = list(splitter.split(df))
        logger.info(f"Total outer splits generated: {len(splits)}")
        
        # Save split diagnostics
        if config.get("outputs", {}).get("save_split_diagnostics", True):
            diag_df = splitter.get_split_diagnostics()
            split_diagnostics_all.append(diag_df)
            diag_path = os.path.join(base_dir, "splits", f"horizon_{horizon}_splits.csv")
            os.makedirs(os.path.dirname(diag_path), exist_ok=True)
            diag_df.to_csv(diag_path, index=False)
            
        for f_set_name in feature_sets:
            features = feature_defs.get(f_set_name, [])
            # Only keep features that exist in df
            features = [f for f in features if f in df.columns]
            
            logger.info(f"--- Feature Set: {f_set_name} (features: {len(features)}) ---")
            
            for model_name in models_to_run:
                # Handle baseline constraints: yield_only_logit is only run on yield_only feature set
                if model_name == "yield_only_logit" and f_set_name != "yield_only":
                    continue
                if model_name == "core_logit" and f_set_name != "core_econometric":
                    continue
                    
                logger.info(f"Fitting model: {model_name}...")
                
                # Loop splits
                for split_idx, (train_idx, test_idx) in enumerate(splits):
                    split_id = split_idx + 1
                    
                    # Filter out NaN targets from training
                    # At the end of sample, target can be NaN. We drop them from train.
                    train_df = df.iloc[train_idx]
                    train_clean = train_df.dropna(subset=[target_col])
                    
                    if len(train_clean) < 10:
                        logger.warning(f"Clean training set size too small ({len(train_clean)}) for split {split_id}. Skipping.")
                        continue
                        
                    test_df = df.iloc[test_idx]
                    test_clean = test_df.dropna(subset=[target_col])
                    
                    if len(test_clean) == 0:
                        logger.warning(f"No test targets in split {split_id}. Skipping.")
                        continue
                        
                    X_train, y_train = train_clean[features], train_clean[target_col].astype(int).reset_index(drop=True)
                    X_test, y_test = test_clean[features], test_clean[target_col].astype(int).reset_index(drop=True)
                    
                    # Fit StandardScaler
                    scaler = StandardScaler()
                    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
                    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)
                    
                    # Standard deviation of training features for coefficients scaling
                    std_x = scaler.scale_
                    
                    # Hyperparameter tuning
                    best_params = {}
                    if model_name in model_grids:
                        grid = model_grids[model_name]
                        best_params = tune_hyperparameters(
                            model_name=model_name,
                            X_train=X_train_scaled,
                            y_train=y_train,
                            dates_train=train_clean['DATE'],
                            horizon=horizon,
                            model_grid=grid,
                            random_state=seed
                        )
                        hyperparams_list.append({
                            "horizon": horizon,
                            "feature_set": f_set_name,
                            "model": model_name,
                            "split_id": split_id,
                            "best_params": json.dumps(best_params)
                        })
                        
                    # Fit main model
                    model = get_base_model(model_name, best_params, random_state=seed)
                    
                    # If ensemble, we can pass inner splits
                    if model_name == "hybrid_ensemble":
                        # Stacking needs out-of-fold level 0 predictions. We generate inner K-fold splits.
                        from sklearn.model_selection import KFold
                        kf = KFold(n_splits=5, shuffle=False)
                        inner_splits = list(kf.split(X_train_scaled))
                        model.fit(X_train_scaled, y_train, inner_cv_splits=inner_splits)
                        
                        # Save meta weights dynamics
                        if hasattr(model, "get_meta_weights"):
                            w_dict = model.get_meta_weights()
                            if w_dict:
                                last_train_date = train_clean['DATE'].iloc[-1].strftime("%Y-%m-%d")
                                w_dict["horizon"] = horizon
                                w_dict["feature_set"] = f_set_name
                                w_dict["split_id"] = split_id
                                w_dict["last_train_date"] = last_train_date
                                meta_weights_records.append(w_dict)
                    else:
                        model.fit(X_train_scaled, y_train)
                        
                    # Predict out-of-sample probabilities
                    pred_probs = model.predict_proba(X_test_scaled)[:, 1]
                    
                    # Calibrate probabilities if specified
                    if calib_method != "none":
                        # Generate validation set for calibrator: we use the last 20% of training data as val set
                        val_size = int(len(X_train_scaled) * 0.2)
                        X_tr_cal = X_train_scaled.iloc[:-val_size]
                        y_tr_cal = y_train.iloc[:-val_size]
                        X_val_cal = X_train_scaled.iloc[-val_size:]
                        y_val_cal = y_train.iloc[-val_size:]
                        
                        temp_model = get_base_model(model_name, best_params, random_state=seed)
                        if model_name == "hybrid_ensemble":
                            temp_model.fit(X_tr_cal, y_tr_cal)
                        else:
                            temp_model.fit(X_tr_cal, y_tr_cal)
                        val_probs = temp_model.predict_proba(X_val_cal)[:, 1]
                        
                        calibrator = RecessionCalibrator(method=calib_method)
                        calibrator.fit(val_probs, y_val_cal)
                        pred_probs = calibrator.calibrate(pred_probs)
                        
                    # Save test predictions
                    for i, idx in enumerate(test_clean.index):
                        row = test_clean.loc[idx]
                        # Target window start/end
                        lu_start = row['DATE'] + pd.DateOffset(months=1)
                        lu_end = row['DATE'] + pd.DateOffset(months=horizon)
                        
                        all_oos_predictions.append({
                            "forecast_origin": row['DATE'].strftime("%Y-%m-%d"),
                            "horizon": horizon,
                            "model": model_name,
                            "feature_set": f_set_name,
                            "y_true": int(row[target_col]),
                            "y_prob": float(pred_probs[i]),
                            "split_id": split_id,
                            "target_window_start": lu_start.strftime("%Y-%m-%d"),
                            "target_window_end": lu_end.strftime("%Y-%m-%d")
                        })
                        
                    # Extract Standardized Coefficients for interpretation
                    coef_df = extract_standardized_coefficients(model, features, std_x, horizon, split_id)
                    if not coef_df.empty:
                        coef_df["feature_set"] = f_set_name
                        coef_df["model"] = model_name
                        coef_list.append(coef_df)
                        
                    # Compute SHAP values on OOS set (Contribution C3)
                    # SHAP is done on XGBoost/boosting model for horizon=6 by default or the one specified
                    shap_horizon = config.get("explainability", {}).get("shap_horizon", 6)
                    shap_model_name = config.get("explainability", {}).get("shap_model", "xgboost")
                    if horizon == shap_horizon and model_name == shap_model_name:
                        logger.info(f"Computing SHAP values for split {split_id}...")
                        recession_eps = config.get("explainability", {}).get("recession_episodes", [])
                        shap_df = shap_by_recession_episode(
                            model=model,
                            X_oos=X_test_scaled,
                            forecast_origins=test_clean['DATE'],
                            recession_episodes=recession_eps
                        )
                        if not shap_df.empty:
                            shap_df["split_id"] = split_id
                            shap_df["feature_set"] = f_set_name
                            shap_list.append(shap_df)

    # 5. Save all OOS predictions and results
    predictions_df = pd.DataFrame(all_oos_predictions)
    for h in horizons:
        pred_h_df = predictions_df[predictions_df['horizon'] == h]
        pred_h_path = os.path.join(base_dir, "predictions", f"oos_predictions_h{h}.csv")
        os.makedirs(os.path.dirname(pred_h_path), exist_ok=True)
        pred_h_df.to_csv(pred_h_path, index=False)
        logger.info(f"Saved predictions for horizon {h} to {pred_h_path} (shape: {pred_h_df.shape})")
        
    # Save coefficients
    if coef_list:
        coefs_all = pd.concat(coef_list, ignore_index=True)
        for h in horizons:
            coefs_h = coefs_all[coefs_all['horizon'] == h]
            coefs_path = os.path.join(base_dir, "explainability", f"logistic_coefficients_h{h}.csv")
            os.makedirs(os.path.dirname(coefs_path), exist_ok=True)
            coefs_h.to_csv(coefs_path, index=False)
            
    # Save SHAP
    if shap_list:
        shap_all = pd.concat(shap_list, ignore_index=True)
        shap_horizon = config.get("explainability", {}).get("shap_horizon", 6)
        shap_path = os.path.join(base_dir, "explainability", f"shap_by_episode_h{shap_horizon}.csv")
        os.makedirs(os.path.dirname(shap_path), exist_ok=True)
        shap_all.to_csv(shap_path, index=False)
        logger.info(f"Saved SHAP values by episode to {shap_path}")
        
    # Save Tuning hyperparams
    if hyperparams_list:
        hyperparams_df = pd.DataFrame(hyperparams_list)
        hp_path = os.path.join(base_dir, "tuning", "best_hyperparameters.csv")
        os.makedirs(os.path.dirname(hp_path), exist_ok=True)
        hyperparams_df.to_csv(hp_path, index=False)
        
    # Save meta weights dynamics
    if meta_weights_records:
        meta_weights_df = pd.DataFrame(meta_weights_records)
        mw_path = os.path.join(base_dir, "tables", "table_meta_weights.csv")
        os.makedirs(os.path.dirname(mw_path), exist_ok=True)
        meta_weights_df.to_csv(mw_path, index=False)
        logger.info(f"Saved meta-learner weights to {mw_path}")
        
    logger.info("Out-of-sample experiments completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Run Out-Of-Sample prediction experiments.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML file.")
    args = parser.parse_args()
    
    logger.info(f"Loading config from {args.config}")
    config = load_experiment_config(args.config)
    
    run_experiment(config)

if __name__ == "__main__":
    main()
