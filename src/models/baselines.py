import statsmodels.api as sm
import numpy as np
import pandas as pd
from src.utils.logging_utils import get_logger

logger = get_logger("baselines")

class StatsmodelWrapper:
    """Wrapper class for Probit or Logit models using statsmodels."""
    def __init__(self, model_type: str = "logit"):
        self.model_type = model_type.lower()
        self.model = None
        self.results = None
        self.feature_names = None
        self.fallback_model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feature_names = list(X.columns)
        X_const = sm.add_constant(X, has_constant='add')
        
        # Check if X is empty or all constant
        if X_const.shape[1] == 0:
            logger.warning("Empty features in StatsmodelWrapper. Using fallback.")
            self._fit_sklearn_fallback(X, y)
            return

        try:
            if self.model_type == "probit":
                self.model = sm.Probit(y, X_const)
                # Try BFGS first as it's more stable
                self.results = self.model.fit(disp=0, method='bfgs', maxiter=1000)
            else:
                self.model = sm.Logit(y, X_const)
                self.results = self.model.fit(disp=0, method='bfgs', maxiter=1000)
        except Exception as e:
            logger.warning(f"Statsmodel {self.model_type} fit failed: {e}. Falling back to Logit (Newton).")
            try:
                self.model = sm.Logit(y, X_const)
                self.results = self.model.fit(disp=0, maxiter=1000)
            except Exception as ex:
                logger.error(f"Fallback Logit failed as well: {ex}. Using scikit-learn LogisticRegression.")
                self._fit_sklearn_fallback(X, y)
                
    def _fit_sklearn_fallback(self, X: pd.DataFrame, y: pd.Series):
        from sklearn.linear_model import LogisticRegression
        self.fallback_model = LogisticRegression(C=1e5, solver='lbfgs', max_iter=1000, random_state=42)
        self.fallback_model.fit(X, y)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.fallback_model is not None:
            return self.fallback_model.predict_proba(X)
            
        X_const = sm.add_constant(X, has_constant='add')
        # Ensure const column exists at the beginning
        if 'const' not in X_const.columns:
            X_const.insert(0, 'const', 1.0)
            
        # Reorder columns to match the training X_const columns
        expected_cols = self.results.params.index
        for col in expected_cols:
            if col not in X_const.columns:
                X_const[col] = 0.0
        X_const = X_const[expected_cols]
        
        p1 = self.results.predict(X_const).values
        p0 = 1 - p1
        return np.column_stack([p0, p1])
