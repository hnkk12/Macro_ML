import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import statsmodels.api as sm

warnings.filterwarnings("ignore")

CWD = r"D:\NCKH ML Macro\Recession-Predictor-master"
CORE_FEATURES = ["T10Y3M_level", "T10Y2Y_level", "UNRATE_level", "Credit_Spread_level"]

def get_hyperparameter_grids():
    """
    Returns search grids for hyperparameter tuning.
    Grids are kept concise to ensure fast execution during walk-forward backtests.
    """
    grids = {
        "rf": {
            "max_depth": [3, 5],
            "min_samples_split": [4, 8],
            "n_estimators": [100]
        },
        "xgb": {
            "max_depth": [2, 3],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [50]
        },
        "lgb": {
            "max_depth": [2, 3],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [50],
            "verbosity": [-1]
        }
    }
    return grids

def train_tuned_classifier(estimator, param_grid, X, y):
    """
    Tunes an estimator using TimeSeriesSplit cross-validation on the training set.
    """
    tscv = TimeSeriesSplit(n_splits=3)
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1
    )
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def run_walk_forward_backtest():
    """
    Executes walk-forward expanding-window cross-validation from 2006 to 2026.
    Fits Probit, tuned Logistic Regression, RF, XGB, LGB, and a Stacking Ensemble.
    """
    processed_path = os.path.join(CWD, "01_data", "processed_monthly_dataset.csv")
    df = pd.read_csv(processed_path, parse_dates=[0], index_col=0)
    df = df.sort_index()
    
    target_cols = ["target_3m", "target_6m", "target_12m"]
    exclude_cols = target_cols + ["USREC"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    models_dir = os.path.join(CWD, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    start_test_year = 2006
    end_test_year = 2026
    
    grids = get_hyperparameter_grids()
    prediction_results = {3: [], 6: [], 12: []}
    
    for horizon in [3, 6, 12]:
        print(f"\n[Backtest] Training models for horizon: {horizon}m")
        target_col = f"target_{horizon}m"
        
        for test_year in range(start_test_year, end_test_year + 1):
            train_end_date = pd.Timestamp(f"{test_year-1}-12-01")
            test_start_date = pd.Timestamp(f"{test_year}-01-01")
            test_end_date = pd.Timestamp(f"{test_year}-12-01")
            
            # Remove target overlap at the end of training set to prevent leakage
            max_train_target_date = train_end_date - pd.DateOffset(months=horizon)
            
            train_df = df.loc[:max_train_target_date].dropna(subset=feature_cols + [target_col])
            test_df = df.loc[test_start_date:test_end_date]
            
            if train_df.empty or test_df.empty:
                continue
                
            print(f"  Test Year {test_year} | Train range: {train_df.index[0].strftime('%Y-%m')} to {train_df.index[-1].strftime('%Y-%m')} | Test size: {len(test_df)}")
            
            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_test = test_df[feature_cols]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            # Use fillna(0) for test feature missing values (scaled mean)
            X_test_scaled = scaler.transform(X_test.fillna(0))
            
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=train_df.index)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=test_df.index)
            
            # 1. Probit Baseline (econometric, trained on core variables)
            X_train_probit = X_train_scaled_df[CORE_FEATURES]
            X_test_probit = X_test_scaled_df[CORE_FEATURES]
            
            X_train_probit_const = sm.add_constant(X_train_probit)
            X_test_probit_const = sm.add_constant(X_test_probit, has_constant="add")
            
            try:
                probit_model = sm.Probit(y_train, X_train_probit_const).fit(disp=0, maxiter=35)
                prob_probit = probit_model.predict(X_test_probit_const).values
            except Exception:
                # Fallback to sklearn Logistic Regression on core features if Probit fails to converge
                fallback_lr = LogisticRegression(C=1.0, random_state=42)
                fallback_lr.fit(X_train_probit, y_train)
                prob_probit = fallback_lr.predict_proba(X_test_probit)[:, 1]
                
            # 2. Logistic Regression (L2 regularized, trained on full feature panel)
            lr_model = LogisticRegression(C=0.1, penalty="l2", random_state=42)
            lr_model.fit(X_train_scaled, y_train)
            prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
            
            # 3. Random Forest (Tuned)
            rf_estimator = RandomForestClassifier(random_state=42, n_jobs=-1)
            rf_model = train_tuned_classifier(rf_estimator, grids["rf"], X_train.fillna(0), y_train)
            prob_rf = rf_model.predict_proba(X_test.fillna(0))[:, 1]
            
            # 4. XGBoost (Tuned)
            xgb_estimator = xgb.XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)
            xgb_model = train_tuned_classifier(xgb_estimator, grids["xgb"], X_train.fillna(0), y_train)
            prob_xgb = xgb_model.predict_proba(X_test.fillna(0))[:, 1]
            
            # 5. LightGBM (Tuned)
            lgb_estimator = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
            lgb_model = train_tuned_classifier(lgb_estimator, grids["lgb"], X_train.fillna(0), y_train)
            prob_lgb = lgb_model.predict_proba(X_test.fillna(0))[:, 1]
            
            # 6. Stacked Ensemble (Combining LR, RF, XGB, and LGB)
            stacking_clf = StackingClassifier(
                estimators=[
                    ('lr', lr_model),
                    ('rf', rf_model),
                    ('xgb', xgb_model),
                    ('lgb', lgb_model)
                ],
                final_estimator=LogisticRegression(C=1.0, penalty="l2", random_state=42),
                cv=3,
                n_jobs=-1
            )
            stacking_clf.fit(X_train_scaled, y_train)
            prob_stack = stacking_clf.predict_proba(X_test_scaled)[:, 1]
            
            # Save predictions
            for idx, date in enumerate(test_df.index):
                actual_rec = test_df.loc[date, "USREC"]
                actual_target = test_df.loc[date, target_col]
                
                prediction_results[horizon].append({
                    "DATE": date,
                    "actual_USREC": actual_rec,
                    "actual_target": actual_target,
                    "prob_probit": prob_probit[idx],
                    "prob_lr": prob_lr[idx],
                    "prob_rf": prob_rf[idx],
                    "prob_xgb": prob_xgb[idx],
                    "prob_lgb": prob_lgb[idx],
                    "prob_stack": prob_stack[idx]
                })
                
        # Export predictions
        horizon_df = pd.DataFrame(prediction_results[horizon])
        horizon_df.to_csv(os.path.join(models_dir, f"predictions_{horizon}m.csv"), index=False)
        print(f"[Backtest] Exported horizon {horizon}m predictions to models/predictions_{horizon}m.csv")

if __name__ == "__main__":
    run_walk_forward_backtest()
