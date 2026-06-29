import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from src.models.baselines import StatsmodelWrapper
from src.models.sklearn_models import get_logistic_l2, get_random_forest
from src.models.boosting import get_xgboost, get_lightgbm

class HybridProbitMLEnsemble(BaseEstimator, ClassifierMixin):
    """
    Stacked Ensemble model combining:
    - Level-0: Probit, Logistic Regression (L2), Random Forest, XGBoost, LightGBM
    - Level-1: Logistic Regression meta-learner
    
    Contribution C1:
    Combines econometric prior (Probit) with flexible machine learning models
    to balance interpretability and non-linear patterns.
    """
    def __init__(self, horizon: int = 6, random_state: int = 42):
        self.horizon = horizon
        self.random_state = random_state
        self.classes_ = np.array([0, 1])
        
        self.models = {
            "probit": StatsmodelWrapper(model_type="probit"),
            "logistic_l2": get_logistic_l2(C=1.0, random_state=self.random_state),
            "random_forest": get_random_forest(n_estimators=100, max_depth=5, random_state=self.random_state),
            "xgboost": get_xgboost(max_depth=2, n_estimators=50, random_state=self.random_state),
            "lightgbm": get_lightgbm(max_depth=2, n_estimators=50, random_state=self.random_state)
        }
        self.meta_learner = LogisticRegression(C=1.0, fit_intercept=True, random_state=self.random_state)
        
    def fit(self, X: pd.DataFrame, y: pd.Series, inner_cv_splits: list = None):
        """Fit Level-0 models and stack using Level-1 meta-learner."""
        # 1. Fit all models on the full training data
        for name, model in self.models.items():
            model.fit(X, y)
            
        # 2. Generate out-of-fold predictions for the meta-learner
        if inner_cv_splits is None:
            # Default KFold splits if not provided
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=5, shuffle=False)
            inner_cv_splits = list(kf.split(X))
            
        meta_features = np.zeros((len(X), len(self.models)))
        
        # Fill out-of-fold predictions
        for name_idx, (name, model) in enumerate(self.models.items()):
            for train_idx, val_idx in inner_cv_splits:
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                
                temp_model = self._get_model_instance(name)
                temp_model.fit(X_tr, y_tr)
                preds = temp_model.predict_proba(X_val)[:, 1]
                meta_features[val_idx, name_idx] = preds
                
        # 3. Fit meta-learner
        self.meta_learner.fit(meta_features, y)
        return self
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        level0_preds = []
        for name, model in self.models.items():
            level0_preds.append(model.predict_proba(X)[:, 1])
            
        meta_features = np.column_stack(level0_preds)
        return self.meta_learner.predict_proba(meta_features)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
        
    def _get_model_instance(self, name: str):
        if name == "probit":
            return StatsmodelWrapper(model_type="probit")
        elif name == "logistic_l2":
            return get_logistic_l2(C=1.0, random_state=self.random_state)
        elif name == "random_forest":
            return get_random_forest(n_estimators=100, max_depth=5, random_state=self.random_state)
        elif name == "xgboost":
            return get_xgboost(max_depth=2, n_estimators=50, random_state=self.random_state)
        elif name == "lightgbm":
            return get_lightgbm(max_depth=2, n_estimators=50, random_state=self.random_state)
            
    def get_meta_weights(self) -> dict:
        """Return the weights assigned to each base model by the meta-learner."""
        if not hasattr(self.meta_learner, "coef_"):
            return {}
        weights = self.meta_learner.coef_[0]
        model_names = list(self.models.keys())
        weights_dict = {model_names[i]: float(weights[i]) for i in range(len(model_names))}
        weights_dict["intercept"] = float(self.meta_learner.intercept_[0])
        return weights_dict
